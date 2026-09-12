Let me check the remaining facts I need: blend mode, MSAA, and the `pxStep` range.

Reviewed it. Three findings reframe what the deck needs to be, so let me lead with those.

## Review: what's actually in fade-frag.glsl

**1. `fade()` is a bare Gaussian; the `pdf()` scaffolding cancels.**

```glsl
pdfCoef = pdf(0., 0., sdev*sdev)        // = 1/sqrt(2πσ²)  — the peak height
dist    = pdf(dist, 0., sdev*sdev) / pdfCoef
```

The `1/sqrt(2πσ²)` appears in numerator and denominator and cancels exactly, so the whole function is

```
fade = exp(−½ · (dist/σ)²)
```

a Gaussian normalized to **peak 1**, not to unit area. That's the answer to "what's a Gaussian alpha filter": it's not filtering in the signal-processing sense at all. It's a bell-shaped opacity ramp — 1.0 at the column's mean, falling off with distance measured in standard deviations. At 1σ it's 0.61, at 2σ 0.14, at 3σ 0.011. The statistical meaning is the payoff: a pixel's opacity is (proportional to) the likelihood that a sample from that column landed at that height. Dense where samples cluster, faint in the tails.

The degenerate branch `variance == 0 → x == mean ? 9999 : 0` is a Dirac delta with 9999 standing in for infinity; the ratio 9999/9999 = 1 makes σ=0 collapse to a hard step.

**2. The `side` attribute makes the stats varyings *flat*. This is the centerpiece.**

I worked the strip topology out, and it resolves what `side` is for. Per column the buffer emits four vertices, but position ignores `side`, so they occupy only two distinct points — both at the same x. In a triangle strip that means **4 of every 6 triangles are degenerate** (zero area). The real quad between columns *i* and *i+1* is `v2,v3,v4,v5` = the `side=+1` pair of column *i* and the `side=−1` pair of column *i+1*. Evaluate lines 238-241 for both:

| | `statsLeft` | `statsRight` | `statsPrevRight` | `statsNextLeft` |
|---|---|---|---|---|
| left edge (`side=+1`, col *i*) | stats*ᵢ* | stats*ᵢ₊₁* | stats*ᵢ₋₁* | stats*ᵢ₊₂* |
| right edge (`side=−1`, col *i+1*) | stats*ᵢ* | stats*ᵢ₊₁* | stats*ᵢ₋₁* | stats*ᵢ₊₂* |

Identical. So all four varyings are **constant across each quad** — zero interpolation gradient. `side` exists to emulate `flat` varyings, which GLSL ES 1.00 doesn't have. The 2× vertex cost and the degenerate triangles are the price. The author even drew it in the `||` of the ASCII diagram at index.js:142-148.

That's also the answer to "account for the varyings": the fragment shader receives a clean 4-column window (*i−1, i, i+1, i+2*) with no interpolation, which is exactly what the local-extremum logic needs.

**3. Anti-aliasing: there is none, anywhere.** No `smoothstep`, no `fwidth`/`dFdx` (and `OES_standard_derivatives` is never enabled), no `discard` (line 99 commented out), `antialias` never requested on the context. The band boundary `y > avg + halfThickness` is a hard binary test, so alpha jumps discontinuously from 1.0 (inside) to `exp(−½(halfThickness/σ)²)` (just outside) — a 40% cliff when σ ≈ halfThickness. The *outer* falloff looks smooth only because the Gaussian reaches ~0 before the quad's edge, hiding that edge by accident. A smooth gradient is not AA; nothing here resolves sub-pixel edge coverage.

Two more worth plates: the four `if` blocks look like they could compound (`a *=` four times) but are provably mutually exclusive — pairwise contradictory on either the `y` comparison or the avg ordering. And `float x` at line 34 is computed and never used.

## The 0.5px question — this is the normal case, not an edge case

`index.js:547`: `this.pxStep || .5`, commented *"pxStep affects jittering on panning, .5 is good value"*. So **columns default to half a pixel wide** whenever zoomed out (`pxStep = max(width/span, 0.5)`). The range shader is selected exactly in that regime (`pxPerSample <= 1`, line 684).

Consequences, all worth their own plate:

- Two quads tile each pixel in x, and a pixel is shaded only if its **center** is covered — so exactly one of the two wins. It's decimation, not averaging: half the columns' stats are never sampled.
- The mitigation is finding #2. Because the varyings are flat and span a 4-column window, the surviving quad still carries stats*ᵢ₋₁…ᵢ₊₂*. The alpha envelope is computed from neighbours the rasterizer threw away.
- The join's x-component shears the quad (join.x ≠ 0 for steep segments, scaled by `thickness/viewport.z`), so quads are parallelograms that *can* overlap in x and both land — which is where overdraw comes from.

## Is the fragment shader run multiple times per pixel?

Yes, but not for the reason you might suspect. Per covering triangle, once per pixel — so:

- **Overdraw is the real multiplier.** `depth.enable = false` (line 264), no discard, blending on. Every triangle covering a pixel shades it and composites. Sheared quads, join overlaps, and the multipass x-shift (`passId`) all stack.
- The blend func (lines 255-260) is src-over for RGB with **alpha union** for alpha (`dstA = srcA(1−dstA) + dstA`). Two passes at α=a give `2a − a²`, not `2a` — it saturates rather than summing. That's the closest thing to AA in the whole pipeline, and it's approximate.
- **MSAA would not help even if enabled**: the fragment shader runs once per *pixel* and the result is broadcast to covered samples; only coverage is per-sample. The `y > avg + halfThickness` test is evaluated at one sample point regardless.
- Derivative helper invocations (the 2×2 quad) don't apply — no derivative instructions here.
- The degenerate triangles cost vertex work but produce zero fragments.

## Proposed plate sequence

Same visual identity as the existing deck, so they read as volumes I and II. But this subject needs one new tool: **actual canvases at extreme magnification** (one screen pixel = a ~24px cell, with the alpha value printed in it) for the pixel-grid plates. SVG can't honestly show per-pixel sampling. I'd reimplement `fade()` in JS and evaluate it per cell, same as the last deck reimplemented the join math.

1. **Hand-off.** Volume I ended at `gl_Position`. Show the finished ribbon and the five varyings crossing the boundary, with units: `dist`, `σ`, `halfThickness` are all fractions of viewport height — which is why `dist/σ` is dimensionless.
2. **The `side` trick.** The strip topology, the two coincident positions, the 4-of-6 degenerate triangles, and the table above showing both edges agree. Punchline: flat varyings without `flat`.
3. **What a Gaussian alpha ramp is.** Build `exp(−½(d/σ)²)` from scratch for a reader who hasn't met it: the bell, σ as the horizontal ruler, the 0.61/0.14/0.011 landmarks. Interactive σ slider is the obvious one here.
4. **The cancellation.** Side-by-side: the code's two `pdf()` calls and a divide, versus the one `exp` it reduces to. Include the σ=0 Dirac case and the 9999 sentinel.
5. **Geometry and alpha as a pair.** Volume I's `vertSdev = 2σ·H/thickness` sized the quad to ±2σ; this shader carves the density out of it. One figure, envelope and fill together — this is the thesis of both decks.
6. **The four blocks.** A 2×2 matrix (above/below × left/right) over a local-max figure, showing why a non-extremum defers to its neighbour's distribution, plus the exclusivity proof.
7. **The hard edge.** Plot alpha vs. y through a column: flat 1.0, then the cliff to `exp(−½(ht/σ)²)`, then the Gaussian tail. Slider on σ so you can watch the cliff grow. This is the "no AA" plate.
8. **Half-pixel columns.** Magnified pixel grid, `pxStep` slider from 2.0 down to 0.5, showing which quad owns each pixel center and which columns vanish. Then the same grid annotated with the 4-column window the survivor still carries.
9. **Overdraw and blending.** Same grid, per-pass contributions, `2a − a²` versus `2a`, and why MSAA wouldn't change the branch.
10. **Edges.** Dead `float x`, the `pdf` redundancy, the inner discontinuity, the decimation — as improvements, not complaints.

Two calls I'd like your read on before I build: whether plates 8-9 should use a real WebGL context (honest, but harder to annotate) or a JS reimplementation on canvas (annotatable, consistent with volume I), and whether 3-4 collapse into one — they're closely related, but §3 is the one thing you said you don't know yet, so I'd lean toward giving it room.