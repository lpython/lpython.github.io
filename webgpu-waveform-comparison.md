# `mrkev/webgpu-waveform` vs `dy/gl-waveform`

Top-level review and comparison. Reviewed at `mrkev/webgpu-waveform@3a0b5a5` (v3.4.0) against `dy/gl-waveform` v4.3.3.

---

## Summary

These look like the same project and are not. They render different things.

**`gl-waveform` renders a distribution.** Each pixel column is summarized as a *mean and standard deviation*, and the fragment shader modulates alpha by a Gaussian so opacity tracks how likely a sample landed at that height.

**`webgpu-waveform` renders an envelope.** Each column is summarized as a *min and max*, and the fragment shader does a binary inside/outside test filled with a flat color.

Almost every other difference between the two codebases follows from that one choice. It is worth stating up front because it means "which is better" is not well-posed — but it is also true that `webgpu-waveform` is the better-factored codebase, at roughly half the size, and that its architecture is the one `gl-waveform` would arrive at if rewritten today.

| | `gl-waveform` | `webgpu-waveform` |
|---|---|---|
| API | WebGL 1 (via regl) | WebGPU |
| Core size | ~1,600 lines (1,074 JS + 522 GLSL) | ~860 lines (681 TS + 174 WGSL) |
| Runtime deps | 18 | 1 (`color`) |
| Statistic | mean + σ | min / max |
| Aggregation | prefix sums of x and x² | min/max LOD pyramid + bounded loop |
| Where stats are computed | vertex shader, per vertex | compute shader, per column |
| Geometry | triangle strip, 4 verts/column | 2 triangles covering the viewport |
| Neighbour access | 4 flat varyings via a `side` attribute | direct indexing into a storage buffer |
| Streaming append | yes (`push`/`set`) | **no** — data fixed at construction |
| Anti-aliasing | none | none |
| Tests | `node test` | `echo 'no tests atm'` |

---

## 1. The statistic is the whole design

`gl-waveform` needs Σx and Σx² to get σ, so it keeps running prefix sums in textures and recovers a window with `sum[hi] − sum[lo]`. That single requirement drags in:

- a second "fractions" texture holding low-order bits, because f32 prefix sums lose precision;
- periodic texture rotation (`textureLength`) to reset the accumulation before it drifts too far;
- `prevSum`/`prevSum2`/`sum`/`sum2` uniforms to stitch windows across texture boundaries;
- `σ² = M(x²) − M(x)²` with an `abs()` guard, which catastrophically cancels when the data's magnitude dwarfs its variance — the source itself flags this as *"rangeDraw gives sdev error for high values dataLength"*.

`webgpu-waveform` needs none of it. **min/max is exactly associative and idempotent in floating point.** `min(min(a,b), c) == min(a, min(b,c))`, no rounding, no cancellation, no accumulation drift. A reduction pyramid over min/max is exact at every level, forever.

That is a genuinely large simplification, and it buys a second property the code exploits explicitly: because the envelope is a *union*, over-covering a range is harmless. The LOD comment makes the point well:

> Because pyramid bins are aligned to multiples of `binSize`, a column's min/max envelope can over-cover by up to one bin on each edge […] that slop is a tiny fraction of a column and is visually imperceptible (and conservative — the envelope only ever grows, never clips).

**This safety does not transfer to mean/σ.** Over-covering a range by a partial bin changes the mean. Any mean-based pyramid needs exact range decomposition — a segment-tree query with partial-edge handling — where min/max can be sloppy and still be correct-looking. That is a real, non-obvious advantage, and it is why the bounded-loop design here can be so much simpler than the equivalent would be for `gl-waveform`.

**What min/max costs you:** you cannot see density. A single-sample click and sustained loud content produce the same full-scale envelope. σ distinguishes them. For audio editing, peaks are what you want to see, so this is the right call; for noisy scientific data it is a real loss of information.

---

## 2. Architecture: compute-then-gather vs. encode-in-geometry

This is the most instructive contrast, and `webgpu-waveform` is decisively cleaner.

### `gl-waveform`

Stats are computed **in the vertex shader**, which then has to smuggle them to the fragment shader through varyings. WebGL 1 has no `flat` qualifier, so it emulates one: each column emits **four** vertices carrying a `side` attribute that never affects `gl_Position` and exists purely to make the stats varyings equal at every corner of a quad. Consequences:

- 4 vertex invocations per column for 2 distinct positions;
- half of all emitted triangles are degenerate (period-4 `D D R R` in the strip);
- ~45 vertex texture fetches *per vertex* (5 `stats()` calls × 9 `picki`), ~4× redundantly;
- vertex texture fetch is an optional capability in GL ES 2.0, which is a live portability risk (there is an iPhone `FIXME` in the source about exactly this);
- the ribbon is real geometry, so sub-pixel columns at the default `pxStep = 0.5` are decimated by pixel-centre coverage — every even column is dropped.

### `webgpu-waveform`

A compute pass, one invocation per pixel column, writes `colMinMax: array<vec2f>`. The render pass then draws two triangles over the viewport, and the fragment does:

```wgsl
let x = u32(input.fragCoord.x);
let mm = colMinMax[x];
```

That is it. The fragment shader derives its own column index from `fragCoord.x` and indexes the buffer directly. **There are no varyings carrying data, no attributes encoding structure, no `side`, no degenerate triangles, no joins, no miter limit, no sub-pixel decimation** — because there is no per-column geometry to rasterize. Every screen column is computed because every screen column *is* a fragment.

The entire apparatus of `gl-waveform`'s vertex shader exists to work around the fragment shader's inability to gather. Given storage buffers and integer math, it evaporates.

---

## 3. The LOD pyramid, as a runtime mode

`webgpu-waveform` ships two compute pipelines and picks between them per frame — which is precisely the "pyramid mode alongside the original mode" pattern, already built:

```ts
const PYRAMID_RATIO = 64;
const COLUMN_LOOP_BUDGET = 64;

private selectLevel(samplesPerPixel: number): number {
  if (samplesPerPixel <= COLUMN_LOOP_BUDGET || this.pyramidLevels.length === 0) {
    return -1;                                   // raw scan
  }
  const minBinSize = samplesPerPixel / COLUMN_LOOP_BUDGET;
  for (let i = 0; i < this.pyramidLevels.length; i++) {
    if (this.pyramidLevels[i].binSize >= minBinSize) return i;
  }
  return this.pyramidLevels.length - 1;
}
```

Bin sizes grow 64 → 4,096 → 262,144. The invariant is that a column's loop never exceeds ~64 iterations at any zoom: scan raw samples while there are few enough, otherwise pick the finest level that keeps bins-per-column under budget.

Worth noting the shape of the design: this is a **bounded linear scan over a chosen level**, not an O(log N) segment-tree range query. It is simpler, and for min/max the bin-alignment slop is free. A mean/σ version would need the segment-tree form.

**Build location is the notable weakness.** The pyramid is built on the CPU, once, in the constructor — `buildMinMaxPyramid` is a plain JS nested loop. It is O(N) total (each level reduces the one below), but for a long file that is tens of milliseconds of main-thread blocking at load. The source acknowledges this and notes it could move to a GPU reduction or a worker without touching the render path. That is an honest comment and the abstraction boundary really is clean enough to make it true.

---

## 4. Frame hygiene

`webgpu-waveform` is careful in a way that is easy to overlook:

```ts
const needsCompute =
  !this.colMinMaxValid ||
  scale !== this.lastComputeScale ||
  offset !== this.lastComputeOffset ||
  cwidth !== this.lastComputeWidth;
```

`colMinMax` depends only on `(channelData, scale, offset, width)`, and `channelData` is immutable for the renderer's lifetime — so resizing vertically or changing color skips the compute pass entirely. Uniform uploads are likewise byte-compared before writing. This is better frame discipline than `gl-waveform`'s coarser `dirty` flag, which recomputes `calc()` wholesale.

The catch is that this memoization is *why* streaming is not supported — the correctness argument depends on `channelData` never changing.

---

## 5. Functional gaps

Things `gl-waveform` does that `webgpu-waveform` does not:

- **Streaming / incremental append.** `push()`, `set()`, texture rotation, cross-texture sum stitching. `webgpu-waveform` fixes `channelData` at construction; live data means rebuilding the renderer and the whole pyramid. This is the single largest functional difference and rules `webgpu-waveform` out for live-signal use as written.
- **Line mode with real thickness and joins.** `webgpu-waveform` has no line rendering at all — no thickness, no joins, no `line-vert.glsl` equivalent.
- **Distribution rendering.** The Gaussian alpha fade has no counterpart; fill is a flat uniform color.
- **`pick()`** hit-testing.
- **Configurable amplitude range.** `webgpu-waveform` hardcodes PCM −1..1 (`yPosNorm`); `gl-waveform` has `amplitude`, auto-derived from running min/max when unset.
- **Opacity / multi-channel examples.**

Things `webgpu-waveform` does better:

- Half the code for the parts that are hard.
- Exact arithmetic where `gl-waveform` fights precision.
- One runtime dependency against eighteen.
- Real TypeScript types, ESM + UMD, changesets, pnpm workspace, a published docs site.
- A React wrapper as a separate package rather than a framework opinion in the core.

---

## 6. Shared shortcomings

**Neither anti-aliases.** `gl-waveform` uses hard comparisons against `±halfThickness`; `webgpu-waveform` uses

```wgsl
return select(vec4f(0.0), waveformColor, insideWaveform);
```

a binary test. The `epsilon = 1.0 / uniforms.height` is a minimum-visibility fudge so near-silent passages still draw a line — not AA. Both would improve markedly from a `smoothstep` over one pixel of distance. WebGPU makes this nearly free: the fragment already knows its exact distance to the envelope boundary in pixels.

**Neither has meaningful tests.** `webgpu-waveform`'s test script is literally `echo 'no tests atm'`. Given both are pure functions from (samples, scale, offset) to per-column summaries, the aggregation layer is very testable on the CPU without a GPU at all.

---

## 7. Minor findings in `webgpu-waveform`

- **Fractional `samplesPerPixel` truncates.** Both compute paths use `i32(samplesPerPixel)` for the span (`loopMax`, `endSample`). At `samplesPerPixel = 1.5` a column covers 1.5 samples but only 2 integer samples are examined starting at a truncated base, so coverage drifts slightly against the true column span. With min/max the failure mode is a missed peak, which is benign but non-conservative — mildly at odds with the "envelope only ever grows" guarantee stated for the pyramid path.
- **Two-triangle quad rather than a fullscreen triangle.** `TWO_TRIANGLES_COVERING_VIEWPORT` is 6 vertices; the standard 3-vertex fullscreen-triangle trick avoids redundant quad shading along the diagonal seam. Negligible, but free.
- **`selectLevel` linear-scans levels each frame.** Irrelevant in practice — there are only ~3–4 levels at `ratio = 64` — but it is called per frame.
- **No `destroy()`** for the pyramid buffers; `colMinMaxBuffer` is destroyed on resize but pyramid levels and the channel-data storage leak for the renderer's lifetime.

---

## 8. Verdict

They are not competitors. `gl-waveform` is a general data-visualization waveform: streaming, distribution-aware, amplitude-configurable, and portable to essentially any browser at the cost of considerable complexity fighting WebGL 1's limits. `webgpu-waveform` is a focused audio peak renderer: static buffers, exact envelopes, modern packaging, and a much cleaner pipeline bought partly by a newer API and partly by choosing an easier statistic.

The architectural lesson transfers cleanly in one direction. `webgpu-waveform` demonstrates that **compute-pass-per-column plus fragment-side gather** removes essentially all of `gl-waveform`'s vertex-stage complexity — the `side` attribute, the flat-varying emulation, the degenerate triangles, the join branch, and the sub-pixel decimation are all artifacts of not having that option available.

What does *not* transfer for free is the aggregation. Adopting min/max would make a port trivial but change the product. Keeping mean/σ means the pyramid must be exact rather than conservative, which means segment-tree range queries with partial-edge handling rather than a bounded scan over aligned bins — meaningfully more work than what is in this repo, though still far simpler than the prefix-sum machinery it would replace.

**If porting `gl-waveform` to WebGPU:** take the pipeline shape from `webgpu-waveform` wholesale, keep the mean/σ statistic, and budget the real effort for two things it does not have to solve — an exact (not conservative) pyramid query, and incremental append into that pyramid for streaming.
