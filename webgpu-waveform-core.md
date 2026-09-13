# `webgpu-waveform` core — a top-down read

Deep dive into `packages/webgpu-waveform` at `mrkev/webgpu-waveform@3a0b5a5` (v3.4.0). This is the bottom layer: the React wrapper and the docs site both sit on top of it.

Written for someone new to WebGPU. Each WebGPU concept is explained the first time the code uses it, in the order the code reaches it.

Starting point was [`webgpu-waveform-comparison.md`](webgpu-waveform-comparison.md). Where this read disagrees with it, the disagreement is called out and the evidence is given (§9).

---

## 0. The package at a glance

```
src/
  index.ts                 1 line    re-exports GPUWaveformRenderer
  GPUWaveformRenderer.ts   681 lines the class + CPU pyramid builder + color parsing
  waveformShader.ts        174 lines three WGSL programs, as template strings
  nullthrows.ts            6 lines   "throw if null" helper
```

Build is `vite build` in library mode (`vite.config.ts`). It emits an ES module plus a UMD bundle, and `vite-plugin-dts` writes `index.d.ts`. `minify: false`, so `dist/` is readable. The one runtime dependency is `color`, used only to parse CSS color strings.

**The job, in one sentence.** Given a `Float32Array` of PCM samples (−1…1), draw one vertical bar per pixel column from the **min** to the **max** of the samples that column covers.

**The shape of the solution.** Two GPU stages per frame:

1. A **compute pass** runs one tiny program per pixel column and writes `(min, max)` for that column into a GPU buffer called `colMinMax`.
2. A **render pass** draws a rectangle over the whole canvas. For every pixel, a second tiny program looks up `colMinMax[x]` and asks "is this pixel's height between min and max?" Yes → waveform color. No → transparent.

Everything else is setup, caching, or a way to keep stage 1 cheap when zoomed far out.

---

## 1. `index.ts` — the only door

```ts
export { GPUWaveformRenderer } from "./GPUWaveformRenderer";
```

The whole public surface is one class. Inside it, everything is either `private`, `protected`, or `readonly`. The only things a consumer calls are:

| Member | Kind | Purpose |
|---|---|---|
| `GPUWaveformRenderer.create(channelData)` | `static async` | Acquire a GPU and build a renderer |
| `GPUWaveformRenderer.createSync(device, channelData)` | `static` | Same, with a `GPUDevice` you already own |
| `renderer.render(destination, scale, offset, color?)` | method | Draw one frame |
| `renderer.device`, `.renderPipeline`, `.presentationFormat` | `readonly` fields | Exposed, but nothing needs them |

There is no `destroy()`, no way to change the data, and no event or error callback.

> **Stale docs.** The package README documents `create(canvas, channelData)`. `packages/site/src/Playground.tsx:57-59` calls `create(canvas, trimmed)` and `render(800, 0, w, h)`. Neither signature exists anymore. The code is the source of truth.

### The two numbers that define a view

`render` takes the view as two numbers, and every shader works in their units:

- **`scale`**: samples per pixel. The shaders call it `scaleFactor` or `samplesPerPixel`. `1` means one sample per pixel. `0.25` is zoomed in (4 px per sample). `5000` is zoomed out.
- **`offset`**: the index of the sample drawn at pixel column 0. It is an integer, and fractions are truncated with `| 0`.

So pixel column `x` covers samples `[offset + x·scale, offset + (x+1)·scale)`. Keep this formula in mind; the rest of the document keeps coming back to it.

---

## 2. Types and constants

### Module constants (`GPUWaveformRenderer.ts:9-34`)

```ts
const DEFAULT_WAVEFORM_COLOR = [0, 1, 0, 1];   // RGBA, 0..1 floats — opaque green
const COMPUTE_WORKGROUP_SIZE = 64;             // must match @workgroup_size(64) in WGSL
const PYRAMID_RATIO = 64;                      // each LOD level = 64 bins of the level below
const COLUMN_LOOP_BUDGET = 64;                 // max loop iterations per column (target)
const TWO_TRIANGLES_COVERING_VIEWPORT = new Float32Array([
  -1,-1,  1,-1,  1, 1,     // triangle 1
  -1,-1,  1, 1, -1, 1,     // triangle 2
]);
```

`COMPUTE_WORKGROUP_SIZE` is duplicated by hand. The TypeScript uses it to count workgroups, and the WGSL hardcodes `@workgroup_size(64)`. If one changes without the other, columns silently go missing or get computed twice. §6.2 covers workgroups.

### `PyramidLevel` (`:37-46`)

```ts
type PyramidLevel = {
  dataBuffer: GPUBuffer;  // array<vec2f>: (min,max) per bin
  lodBuffer:  GPUBuffer;  // uniform: (binSize, levelLength)
  binSize: number;        // raw samples per bin: 64, 4096, 262144, ...
  length:  number;        // bins in this level
};
```

This is one pre-reduced copy of the audio at a coarser resolution, already uploaded to the GPU. `binSize` and `length` are also kept on the CPU because `selectLevel` needs them without asking the GPU.

### Fields of the class, grouped by role

| Group | Fields | Lifetime |
|---|---|---|
| Geometry | `vertices`, `vertexBuffer`, `vertexCount` | Fixed at construction |
| View uniforms | `uniformArray`, `uniformBuffer`, `uniformBytesView`, `prevUniformBytes`, `prevUniformBytesValid` | Buffer fixed; contents per frame |
| Color uniform | `waveformColor`, `waveformColorBuffer`, `prevWaveformColor`, `prevWaveformColorValid`, `lastColorInput`, `lastColorOutput` | Buffer fixed; contents per frame |
| Audio | `channelData`, `channelDataStorage` | Fixed at construction |
| Pipelines | `renderPipeline`, `computePipeline`, `pyramidColumnPipeline` | Fixed at construction |
| LOD | `pyramidLevels[]` | Fixed at construction |
| Per-width | `colMinMaxBuffer`, `colMinMaxCapacity`, `rawComputeBindGroup`, `pyramidComputeBindGroups[]`, `renderBindGroup` | Rebuilt when the canvas gets wider |
| Memo | `lastComputeScale`, `lastComputeOffset`, `lastComputeWidth`, `colMinMaxValid` | Per frame |
| Canvas | `configuredContext` | Changes when the destination changes |

Only the **per-width** and **memo** groups ever change after construction. Everything else is set up once.

---

## 3. `create()` — getting a GPU (`:137-150`)

```ts
static async create(channelData) {
  if (!navigator.gpu) throw ...;
  const adapter = await navigator.gpu.requestAdapter();
  if (adapter == null) throw ...;
  const device = await adapter.requestDevice();
  return GPUWaveformRenderer.createSync(device, channelData);
}
```

> **WebGPU primer: `navigator.gpu` → adapter → device**
>
> - **`navigator.gpu`** exists only in browsers with WebGPU. This check is the "is WebGPU supported" test.
> - **`GPUAdapter`** represents a physical GPU (or a software fallback) plus what it can do. `requestAdapter()` can return `null` even when `navigator.gpu` exists, for example on a blocklisted driver.
> - **`GPUDevice`** is your *logical connection* to that adapter. Every resource (buffers, shaders, pipelines) is created from a device and belongs only to that device. You can't share a buffer between two devices.
>
> Both requests are `async` because the browser may need to talk to a separate GPU process.

**Limits.** `requestDevice()` is called with no `requiredLimits`, so the device gets WebGPU's *default* limits. One of those is `maxStorageBufferBindingSize = 128 MiB`. The raw audio is bound as a storage buffer (§5), so by spec the ceiling is 33,554,432 samples: **~12.7 min of mono at 44.1 kHz, ~11.6 min at 48 kHz**. Past that, `createBindGroup` fails validation. That failure is *not* a thrown exception (see the error-model note in §6). Not yet confirmed in a browser. The fix is to request `adapter.limits.maxStorageBufferBindingSize` when creating the device.

`createSync(device, channelData)` (`:152-164`) asks the browser for its preferred canvas pixel format with `navigator.gpu.getPreferredCanvasFormat()` (usually `"bgra8unorm"` or `"rgba8unorm"`) and hands off to `createPipeline`. The React wrapper uses `create`, so each `<GPUWaveform>` gets its own device.

---

## 4. `createPipeline()` — compiling the GPU programs (`:166-244`)

This static method builds everything that depends only on the device and the canvas format, not on the audio.

### 4.1 Vertex buffer layout

```ts
const vertexBufferLayout = {
  arrayStride: 8,                       // bytes per vertex
  attributes: [{ format: "float32x2", offset: 0, shaderLocation: 0 }],
};
```

> **Primer: vertex buffer layouts.** A GPU buffer is untyped bytes. The layout tells the pipeline how to slice them: every 8 bytes is one vertex, and the first 8 bytes of that vertex are two `f32`s. They feed the shader input marked `@location(0)`. This is the only place that knows `TWO_TRIANGLES_COVERING_VIEWPORT` holds `(x, y)` pairs.

### 4.2 Shader modules

```ts
device.createShaderModule({ code: waveformShader })               // vertex + fragment
device.createShaderModule({ code: waveformComputeShader })        // raw compute
device.createShaderModule({ code: waveformPyramidComputeShader }) // LOD compute
```

> **Primer: WGSL and shader modules.** WGSL is WebGPU's shading language. It's Rust-flavored, strictly typed, and has no implicit conversions (hence all the `f32(x)` and `i32(...)` casts). A shader module is a compiled WGSL source that can contain several **entry points** (functions marked `@vertex`, `@fragment`, or `@compute`). Compile errors don't throw here either. They show up in the console and make pipelines built from the module invalid.

### 4.3 Three pipelines

```ts
renderPipeline = device.createRenderPipeline({
  layout: "auto",
  vertex:   { module: renderShaderModule, entryPoint: "vertexMain", buffers: [vertexBufferLayout] },
  fragment: { module: renderShaderModule, entryPoint: "fragmentMain", targets: [{ format: canvasFormat }] },
});
computePipeline       = device.createComputePipeline({ layout: "auto", compute: { module: computeShaderModule,        entryPoint: "computeMain" } });
pyramidColumnPipeline = device.createComputePipeline({ layout: "auto", compute: { module: pyramidColumnShaderModule, entryPoint: "computeMain" } });
```

> **Primer: pipelines.** A pipeline is a shader bundled with everything the GPU needs to run it fast: which entry points, the vertex layout, and the output pixel format. Pipelines are expensive to create and cheap to use, so create once and reuse every frame. This code does exactly that.
>
> - A **render pipeline** has a vertex stage (runs per vertex, decides *where* triangles land) and a fragment stage (runs per covered pixel, decides *what color*). Unspecified options take defaults. Topology is `"triangle-list"` (every 3 vertices form a triangle), there's no depth test, and there's **no blending**, so the fragment output simply replaces the pixel.
> - A **compute pipeline** has one stage and produces no pixels. It runs a function N times in parallel and whatever it writes to storage buffers is the result.

> **Primer: `layout: "auto"` and bind groups.** Shaders declare their inputs as numbered slots:
> ```wgsl
> @group(0) @binding(0) var<uniform> uniforms: Uniforms;
> ```
> A **bind group layout** describes the slots (binding 0 is a uniform buffer, binding 1 is a read-only storage buffer…). A **bind group** fills those slots with actual buffers. `layout: "auto"` tells WebGPU to infer the layout from the shader. That's convenient, but every auto layout is **unique to its pipeline**: a bind group made for `computePipeline` can't be used with `pyramidColumnPipeline`, even if the slots look identical. That's why §6 Step 2 builds a separate bind group for each pipeline and for each pyramid level.

The three pipelines go to the private constructor along with `channelData`, `device`, and `canvasFormat`.

---

## 5. The constructor — uploading the data (`:259-315`)

> **Primer: `GPUBuffer` and usage flags.** `device.createBuffer({ size, usage })` allocates GPU memory. `usage` is a bitmask that says what the buffer may be used for, and the GPU enforces it:
>
> | Flag | Meaning |
> |---|---|
> | `VERTEX` | can be bound as vertex data |
> | `UNIFORM` | small, read-only params (fast; size-limited) |
> | `STORAGE` | large arrays; readable, and writable if the shader says `read_write` |
> | `COPY_DST` | the CPU may write into it with `queue.writeBuffer` |
>
> **`device.queue.writeBuffer(buffer, offset, data)`** copies bytes from a JS typed array into a GPU buffer. The copy is *queued*, not immediate. It's guaranteed to land before any command buffer submitted *after* the call. JS never gets a pointer to GPU memory.

The constructor makes, in order:

| # | Buffer | Size | Usage | Contents |
|---|---|---|---|---|
| 1 | `vertexBuffer` | 48 B | `VERTEX \| COPY_DST` | the 6 vertices (12 floats), written once |
| 2 | `uniformBuffer` | 20 B | `UNIFORM \| COPY_DST` | `Uniforms` struct, rewritten on change |
| 3 | `channelDataStorage` | 4·N B | `STORAGE \| COPY_DST` | every raw sample, written once |
| 4 | `waveformColorBuffer` | 16 B | `UNIFORM \| COPY_DST` | RGBA as 4×f32, rewritten on change |
| 5+ | pyramid level buffers | see §5.2 | `STORAGE`/`UNIFORM \| COPY_DST` | written once |

### 5.1 The `Uniforms` struct, byte by byte

`defaultUniformArray()` (`:246-257`) allocates a 20-byte `ArrayBuffer` and puts **two typed-array views** over the same bytes. That lets you write some slots as floats and others as integers:

```
byte   0      4      8      12     16     20
       ├──────┼──────┼──────┼──────┼──────┤
       │scale │width │height│offset│bufLen│
       │ f32  │ f32  │ f32  │ i32  │ i32  │
       └──────┴──────┴──────┴──────┴──────┘
         f32View[0..2]        i32View[3..4]
```

That matches the WGSL declaration, which is repeated verbatim in all three shaders:

```wgsl
struct Uniforms {
  scaleFactor: f32, width: f32, height: f32,
  offset: i32, bufferLength: i32,
};
```

> **Primer: memory layout.** WGSL structs follow alignment rules. Each scalar here is 4 bytes, aligned to 4, so the struct packs to exactly 20 bytes with no padding. The CPU side must produce *byte-identical* layout, and nothing checks it for you. Writing `offset` through the `Float32Array` view would store the float bit pattern of `1234.0`, which the shader reads as the integer 1,150,959,616. The git log has a commit for exactly that bug: `652344b fix issue with offset uniform recieving f32 instead of i32`.

`bufferLength` (slot 4) is written once in the constructor (`:300`) and never touched by `setOptions`. Every shader uses it for bounds checks.

### 5.2 Building the pyramid (`buildPyramid`, `:327-355`, and `buildMinMaxPyramid`, `:603-664`)

`buildMinMaxPyramid` is plain JS on the main thread.

- **Level 1:** walk the raw samples in blocks of 64 and store `(min, max)` for each block.
- **Level k > 1:** walk level k−1 in blocks of 64 bins, taking the min of mins and max of maxes.
- **Stop** once a level has a single bin.
- **Storage:** interleaved `[min0, max0, min1, max1, …]` in a `Float32Array`, which is byte-compatible with WGSL `array<vec2f>` (8 bytes per element).

Example with **3 min of 44.1 kHz mono = 7,938,000 samples**:

| Level | binSize | Bins | Bytes |
|---|---|---|---|
| raw (level 0, *not* in the pyramid) | 1 | 7,938,000 | 31.8 MB |
| L1 | 64 | 124,032 | 992 KB |
| L2 | 4,096 | 1,938 | 15.5 KB |
| L3 | 262,144 | 31 | 248 B |
| L4 | 16,777,216 | 1 | 8 B |

Total pyramid ≈ 3.2% of the raw data. The whole build is O(N): level 1 does all the real work, and each later level is 64× smaller.

For each level, `buildPyramid` uploads two buffers:

- **`dataBuffer`** (`STORAGE`): the `(min, max)` array.
- **`lodBuffer`** (`UNIFORM`, 16 B): `Uint32Array([binSize, length, 0, 0])`, which matches the WGSL `LodParams`.

The 16-byte padding comment overstates the requirement. Two `u32` fields make an 8-byte struct, which is valid. The padding is harmless.

The coarsest levels are never selected. At ratio 64, `selectLevel` would only pick L4 when zoomed past ~1.07 billion samples per pixel. They're tiny, so it doesn't matter.

---

## 6. `render()` — one frame, top to bottom (`:480-586`)

```ts
render(destination: HTMLCanvasElement | OffscreenCanvas | GPUCanvasContext,
       scale: number, offset: number,
       color?: string | [r, g, b, a])
```

Here's a full frame, step by step. The worked example uses **3 min of 44.1 kHz audio on a 1600 px wide canvas, fully zoomed out**, so `scale = 7,938,000 / 1600 = 4961.25`.

### Step 1: get a context and configure it (`:486-508`)

```ts
if (destination instanceof GPUCanvasContext) → use it
else → destination.getContext("webgpu")
if (this.configuredContext !== context) {
  context.configure({ device, format, alphaMode: "premultiplied" });
}
```

> **Primer: `GPUCanvasContext`.** A canvas becomes a WebGPU surface by calling `getContext("webgpu")` and then `configure({device, format})` once. After that, `context.getCurrentTexture()` returns a texture you draw into. When your JS task ends, the browser shows that texture on screen and hands out a *new* one next frame. Nothing you drew is kept between frames, which is why every `render()` must redraw fully even if nothing changed.

`getContext` returns the same object on every call for a given canvas, so the identity check means `configure` runs once per canvas. If one renderer alternates between two canvases, it reconfigures on every switch.

**`alphaMode: "premultiplied"`** makes the canvas transparent wherever the shader writes alpha 0, so the page background shows through. The catch: in premultiplied mode the browser expects every pixel's RGB to already be multiplied by its alpha. The fragment shader outputs `waveformColor` unchanged, and `ensureColorFormat` never premultiplies. That's fine for opaque colors (the default green is `a = 1`). A translucent color like `"rgba(0,255,0,0.5)"` becomes `(0, 1, 0, 0.5)`, which is invalid premultiplied data because RGB > alpha. The spec leaves the result undefined, and in practice it composites too bright. *Needs a browser check, but the expected fix is one line: `rgb * a`.*

### Step 2: size the per-column buffer (`ensureColMinMaxBuffer`, `:381-433`)

```ts
if (colMinMaxBuffer && colMinMaxCapacity >= width) return;   // fast path: nothing to do
colMinMaxBuffer?.destroy();
colMinMaxBuffer = createBuffer({ size: width * 8, usage: STORAGE });
// ...then rebuild every bind group that references it
colMinMaxValid = false;
```

`colMinMax` holds one `vec2f` per pixel column: 1600 × 8 = 12.8 KB. It only **grows**. A narrower canvas reuses the existing buffer, since the shaders ignore `x >= width`. Note there's no `COPY_DST` flag. The CPU never writes this buffer; only the compute shader does.

Every bind group that points at `colMinMax` has to be rebuilt when it's replaced:

```
rawComputeBindGroup           ← computePipeline:        {0: uniforms, 1: channelData,       2: colMinMax}
pyramidComputeBindGroups[i]   ← pyramidColumnPipeline:  {0: uniforms, 1: level[i].data,     2: colMinMax, 3: level[i].lod}
renderBindGroup               ← renderPipeline:         {0: uniforms, 1: colMinMax,         2: waveformColor}
```

One bind group per pyramid level means switching levels while zooming is just "pass a different bind group". Nothing gets uploaded.

`colMinMax` is the **hand-off point**. The compute shader writes it (`read_write`), the fragment shader reads it (`read`), and it never leaves the GPU.

### Step 3: resolve the color (`resolveColor`, `:467-478`)

Strings go through `Color(arg)` → `[r/255, g/255, b/255, alpha]`. Tuples pass through unchanged. The last input and output are memoized by `===`, so a constant string skips re-parsing each frame.

### Step 4: update uniforms, but only if they changed (`:517-529`)

```ts
setOptions(scale, cwidth, cheight, offset);  // writes into this.uniformArray (CPU)
setWaveformColor(waveformColor);             // writes into this.waveformColor (CPU)
if (uniformChanged()) queue.writeBuffer(uniformBuffer, 0, uniformArray);
if (colorChanged())   queue.writeBuffer(waveformColorBuffer, 0, waveformColor);
```

`uniformChanged` compares the 20 bytes against a saved copy, and `colorChanged` compares the 4 floats. An unchanged frame uploads nothing. This is a minor optimization, since `writeBuffer` for 20 bytes is cheap, but it's consistent with the rest of the design.

`setOptions` allocates two typed-array views on every call. That's a little garbage per frame. Negligible, and easy to hoist into fields.

### Step 5: decide whether the compute pass is needed (`:533-537`)

```ts
const needsCompute = !colMinMaxValid
  || scale  !== lastComputeScale
  || offset !== lastComputeOffset
  || cwidth !== lastComputeWidth;
```

`colMinMax` is a pure function of `(channelData, scale, offset, width)`. `channelData` can never change, so if the other three match the last frame, `colMinMax` on the GPU is still correct. Changing **height** or **color** only re-runs the cheap render pass.

This memo is only safe because the data is immutable. Adding streaming later without adding a "data version" to this key would show stale columns, and nothing would crash to tell you.

### Step 6: the compute pass (`:539-565`)

```ts
const encoder = device.createCommandEncoder();
if (needsCompute) {
  const level = selectLevel(scale);                 // -1 = raw, else pyramid index
  const pass = encoder.beginComputePass();
  pass.setPipeline(level < 0 ? computePipeline : pyramidColumnPipeline);
  pass.setBindGroup(0, level < 0 ? rawComputeBindGroup : pyramidComputeBindGroups[level]);
  pass.dispatchWorkgroups(Math.ceil(cwidth / 64));
  pass.end();
  // record lastCompute* and colMinMaxValid = true
}
```

> **Primer: command encoders, passes, and submission.** WebGPU calls don't run immediately. You **record** commands into a `GPUCommandEncoder`, grouped into **passes** (a compute pass or a render pass). `encoder.finish()` turns that recording into a `GPUCommandBuffer`, and `queue.submit([...])` sends it to the GPU. `render()` returns right after `submit`, and the GPU runs the work shortly after. Recording is cheap, validation happens in batches, and the CPU never blocks waiting for the GPU.

#### 6.1 `selectLevel(scale)` (`:361-379`)

```
if scale <= 64                → raw path (-1)
minBinSize = scale / 64
return first level with binSize >= minBinSize (else coarsest)
```

The goal is **"never loop more than ~64 times per column."** For the worked example, `scale = 4961.25`, so `minBinSize = 77.5`. L1 (bin 64) is too fine; L2 (bin 4096) is the first that fits. Each column then spans 4961 / 4096 ≈ 1.2 bins, which is 2–3 loop iterations.

Level **boundaries** (with `scale` in samples per pixel):

| scale range | Path | Bins per column |
|---|---|---|
| ≤ 64 | raw samples | 1–65 samples |
| 64 … 4,096 | L1 (bin 64) | ~1 … 64 |
| 4,096 … 262,144 | L2 (bin 4,096) | ~1 … 64 |
| 262,144 … 16.7 M | L3 (bin 262,144) | ~1 … 64 |

Within each band, the bin size ranges from **1/64 of a column up to nearly a whole column**. That matters in §7.2.

#### 6.2 Workgroups and dispatch

> **Primer: workgroups.** A compute shader declares `@workgroup_size(64)`. `dispatchWorkgroups(n)` runs `n × 64` invocations in total, and each one receives its own `global_invocation_id` (0, 1, 2, …). You can't ask for "exactly 1600 invocations", only a multiple of 64, so you round up and **the shader must ignore the extras**:
> ```wgsl
> if (f32(x) >= uniforms.width) { return; }
> ```
> 1600 / 64 = 25 workgroups = exactly 1600 invocations. A 1601 px canvas gets 26 workgroups (1664 invocations), and the last 63 return immediately. The invocations run in parallel and in no particular order, which is fine because each one writes only its own `colMinMax[x]`.

### Step 7: the render pass (`:568-585`)

```ts
const pass = encoder.beginRenderPass({
  colorAttachments: [{
    view: context.getCurrentTexture().createView(),
    loadOp: "clear", clearValue: {r:0,g:0,b:0,a:0},   // start fully transparent
    storeOp: "store",                                 // keep what we draw (so it can be shown)
  }],
});
pass.setPipeline(renderPipeline);
pass.setVertexBuffer(0, vertexBuffer);
pass.setBindGroup(0, renderBindGroup);
pass.draw(6);
pass.end();
device.queue.submit([encoder.finish()]);
```

The compute pass and the render pass go into **one command buffer**, compute first. The GPU finishes writing `colMinMax` before the fragment shader reads it, and the JS never has to wait between the two.

---

## 7. The shaders (`waveformShader.ts`)

### 7.1 Raw compute: `waveformComputeShader` (`:7-51`)

Bindings: `0` uniforms, `1` `channelData: array<f32>` (read), `2` `colMinMax: array<vec2f>` (read_write).

```wgsl
let x     = gid.x;                                   // pixel column
let base  = uniforms.offset + i32(floor(f32(x) * samplesPerPixel));
if (base < 0 || base >= bufferLength) { colMinMax[x] = vec2f(0,0); return; }
var minV = channelData[base]; var maxV = minV;
let loopMax = min(i32(samplesPerPixel), bufferLength - base - 1);
for (var i = 1; i <= loopMax; i++) { ... min/max channelData[base + i] ... }
colMinMax[x] = vec2f(minV, maxV);
```

- Columns outside the audio get `(0, 0)`, which the fragment stage draws as a thin silence line rather than nothing.
- The loop is **inclusive** (`i <= loopMax`), so it reads `floor(scale) + 1` samples: `base … base + floor(scale)`. That's one more sample than the column strictly owns, so neighboring columns share an edge sample. That's why no sample is ever skipped (§9).
- Zoomed in (`scale < 1`): `i32(0.25) = 0`, so the loop runs zero times, and each column is exactly one sample (`min == max`). Four adjacent columns map to the same sample, which produces the blocky "stair-step" look at extreme zoom. There's no interpolation between samples.

### 7.2 Pyramid compute: `waveformPyramidComputeShader` (`:64-122`)

Same bindings, but binding 1 is a pyramid level's `levelData: array<vec2f>`, plus binding `3` `lod: LodParams`.

```wgsl
let startSample = offset + i32(floor(f32(x) * samplesPerPixel));
let endSample   = min(startSample + i32(samplesPerPixel), bufferLength);   // exclusive
let b0 = clamp(startSample     / binSize, 0, lastBin);                     // integer division rounds down
let b1 = clamp((endSample - 1) / binSize, 0, lastBin);
min/max over levelData[b0 ..= b1]
```

It turns a sample range into the bins that contain it. Integer division rounds `startSample` **down** to its bin's start, and `endSample − 1` **up** to its bin's end. So the bins read cover a **superset** of the column. For min and max, reading extra samples can only make the envelope taller, never shorter. That's the "conservative" argument, and it holds.

**How big is the superset?** The code comment claims the extra coverage "is a tiny fraction of a column," and `pyramid-plates.html` puts it at "at most ~3% of a column." Both assume bins are ~1/64 of a column. But `selectLevel` switches levels in coarse ×64 steps, so right after a switch the bin is nearly as wide as the column. A CPU port of both shaders gives:

| scale | Level | Worst-case (samples read) / (column span) |
|---|---|---|
| 64 | raw | 1.02× |
| 64.5 | L1 bin 64 | **1.98×** |
| 100 | L1 bin 64 | 1.92× |
| 1,000 | L1 bin 64 | 1.09× |
| 4,095 | L1 bin 64 | 1.02× |
| 4,097 | L2 bin 4,096 | **2.00×** |
| 5,000 | L2 bin 4,096 | **2.46×** |
| 100,000 | L2 bin 4,096 | 1.06× |

So a column can pull in almost a full extra column's worth of samples on each side. **Visually, a single-sample transient can light up two adjacent pixel columns instead of one**, and the effect comes and goes as you zoom across a level boundary. It stays conservative and never hides a peak, but "imperceptible" is only true in the middle of a band. The ~3% figure is the best case. (Plate 06's "envelope error exactly zero at 1:64" used a column 64× the bin, the best case, so it doesn't contradict this.)

### 7.3 Render: `waveformShader` (`:124-174`)

**Vertex stage:**

```wgsl
@vertex fn vertexMain(@location(0) pos: vec2f) -> VertexOutput {
  output.pos = vec4f(pos, 0, 1);
}
```

> **Primer: clip space.** The vertex shader's `@builtin(position)` output is in *clip space*: x and y run from −1 to +1 across the viewport (y up), z from 0 to 1. The 6 vertices are exactly the corners (±1, ±1), so the two triangles cover every pixel. After this stage, the GPU works out which pixels each triangle covers and runs the fragment shader once for each. Nothing is passed from vertex to fragment except position.

**Fragment stage:**

```wgsl
@fragment fn fragmentMain(input: FragInput) -> @location(0) vec4f {
  let x  = u32(input.fragCoord.x);                     // which column am I in?
  let mm = colMinMax[x];                                // (min, max) for that column
  let yPosNorm = -1.0 * (2.0 * (input.fragCoord.y / uniforms.height) - 1.0);
  let epsilon  = 1.0 / uniforms.height;
  let inside   = yPosNorm <= mm.y + epsilon && yPosNorm >= mm.x - epsilon;
  return select(vec4f(0), waveformColor, inside);       // select(false_val, true_val, cond)
}
```

> **Primer: `@builtin(position)` in the fragment stage.** In a fragment shader, the same builtin means something different: **framebuffer pixel coordinates** of the pixel being shaded. The origin is the **top-left**, and values sit at pixel *centers* (0.5, 1.5, …). So `u32(fragCoord.x)` is the integer column index. That one line replaces all of `gl-waveform`'s vertex-stage machinery (see the comparison, §2).

Mapping y to amplitude:

```
fragCoord.y :  0 (top) ────────── height (bottom)
y / height  :  0      ────────── 1
2·that − 1  : −1      ────────── +1
× −1        : +1 (top) ───────── −1 (bottom)    ← matches PCM: +1 up, −1 down
```

`epsilon = 1/height` in normalized units is **half a pixel**, since the full −1…+1 range spans `height` pixels. It widens the band by half a pixel above and below, so a flat column (`min == max`, including silence and out-of-range columns) still covers at least one pixel center. Measured: a silent column lights **1 row** at odd heights (99, 101, 333, 1000) and **2 rows** at some even heights (100, 256, 300). At those, both pixel centers adjacent to zero land exactly on the ±ε boundary.

`select` is a hard yes/no, so there's **no anti-aliasing**. Pixels are either fully colored or fully transparent. The comparison (§6) covers the `smoothstep` fix. The render pipeline also has no blend state, so a partial alpha would simply replace what's there, and on a cleared canvas that's fine.

---

## 8. Error model and resources

**What throws:** `navigator.gpu` missing, no adapter, `getContext("webgpu")` returning null. That's all.

**What doesn't throw:** everything else. WebGPU reports validation errors (bad buffer sizes, failed bind groups, shader compile errors) *asynchronously*. They go to the console and to `device.onuncapturederror`, and the invalid object silently does nothing when used. The renderer never calls `device.pushErrorScope`, and it doesn't listen to `device.lost`, which fires when the GPU process resets (driver crash, laptop GPU switch). After a device loss this renderer stays dead until it's rebuilt.

**Edge inputs:**
- *Empty `channelData`:* the storage buffer is 0 bytes. Binding it fails validation (a runtime-sized array needs at least one element), so nothing renders and a console error appears.
- *Zero-width canvas:* `ensureColMinMaxBuffer` protects the buffer size with `max(width, 1)`, but `getCurrentTexture()` on a 0×0 canvas is itself a validation error.
- *Negative offset:* handled. Columns before sample 0 get `(0, 0)` and draw the silence line.
- *Very large `x · scale`:* `f32(x) * samplesPerPixel` loses integer precision past 2²⁴ (16.7 M). When zoomed out, the start sample can be off by a few samples, which doesn't matter since bins are ≥ 4,096 wide there.

**Resource lifetime:** Only `colMinMaxBuffer` is ever `destroy()`ed (when it's replaced). The vertex, uniform, color, channel-data, and pyramid buffers are released only when the renderer is garbage-collected. There's no `destroy()` method to free them sooner, which matters when the React wrapper rebuilds a renderer for each new `AudioBuffer`.

---

## 9. Checks against the comparison doc

A CPU port of both compute shaders, using `Math.fround` for f32 and truncation for i32, was swept across 3,000 random `(scale, offset)` pairs on a 10-minute buffer, plus the fixed cases in §7.2.

| Comparison claim | Verdict |
|---|---|
| Core ~681 TS + 174 WGSL, one runtime dep | ✅ Correct |
| Compute per column, fragment gathers `colMinMax[x]`, no varyings | ✅ Correct |
| Bin sizes 64 → 4,096 → 262,144, loop ≤ ~64 | ✅ Correct (1–65 iterations observed) |
| Compute skipped on height or color change | ✅ Correct |
| **§7: fractional `samplesPerPixel` truncates → "missed peak … non-conservative"** | ❌ **Not observed.** Zero skipped samples in either path across the whole sweep. The raw loop reads `floor(scale) + 1` samples, so neighbors overlap by one. The pyramid path's bin rounding covers the one-sample gap its exclusive `endSample` would otherwise leave. Coverage is **gap-free and conservative**, just not exact. |
| "Slop is a tiny fraction of a column" (source comment, plate deck ~3%) | ⚠️ **Best case only.** Up to ~2.5× coverage just past each level switch (§7.2). |
| §7: Two-triangle quad → "redundant quad shading along the diagonal seam" | ⚠️ Rasterization rules shade each pixel center exactly once. The only redundancy is GPU-internal helper invocations along the diagonal. Negligible, as the comparison says. |
| §7: "No `destroy()` … leak for renderer lifetime" | ✅ Accurate as worded ("live as long as the renderer"). It isn't a true leak, because GC eventually reclaims them. |
| §6: No anti-aliasing | ✅ Correct |
| Not in the comparison | New: **128 MiB default storage limit** (~12 min mono) · **premultiplied alpha not applied** to translucent colors · **no `device.lost` handling** · stale README/Playground API |

---

## 10. Where a mean + σ mode would plug in (for later)

The layering makes a second statistic a matter of adding pieces rather than changing existing ones:

| Layer | Min/max today | Mean + σ mode |
|---|---|---|
| Per-column output | `colMinMax: array<vec2f>` | `colStats: array<vec2f>` = (mean, σ), same size |
| Compute (raw) | min/max loop | Accumulate n, Σx, Σx² (or a running mean/variance) over the same range. Exact: no overlap allowed |
| Compute (LOD) | Scan aligned bins (conservative) | Needs **exact** range decomposition, with partial edges read from raw samples. See `pyramid-plates.html` Plates 04–07 |
| Pyramid storage | (min, max) per bin | (n, mean, M2) per bin, merged exactly |
| Fragment | `select(inside)` | `exp(−½((y − mean)/σ)²)` alpha (see `fade_frag.md`), which also needs premultiplied output |
| Memo key | (scale, offset, width) | + mode |
| Pipelines and bind groups | 3 pipelines | +2 compute pipelines and +1 render pipeline, or one shader with a mode uniform |

Two things in the current code are **not** reusable as-is for mean/σ. First, the raw loop's inclusive read of `floor(scale) + 1` samples: harmless overlap for min/max, but it biases a mean. Second, the aligned-bin pyramid scan (§7.2). Everything else carries over: buffers, the hand-off, dispatch, the memo, and the canvas handling.

---

## Appendix A: vocabulary

| Term | One-line meaning here |
|---|---|
| Adapter | A physical GPU the browser offers |
| Device | Your connection to it; creates everything |
| Queue | Where command buffers and `writeBuffer` calls are sent |
| Buffer | Untyped GPU memory with a usage bitmask |
| Uniform buffer | Small, read-only parameters (`Uniforms`, `LodParams`, color) |
| Storage buffer | Large arrays; the only kind a shader can write (`colMinMax`) |
| Shader module | Compiled WGSL containing entry points |
| Pipeline | Shader + fixed configuration, compiled once |
| Bind group layout | Slot types a pipeline expects (`auto`-inferred here) |
| Bind group | Actual buffers plugged into those slots |
| Command encoder | Records passes into a command buffer |
| Compute pass | Runs `@compute` entry point in parallel workgroups |
| Render pass | Rasterizes triangles into a texture, running vertex and fragment stages |
| Workgroup | A batch of compute invocations (64 here); dispatch counts workgroups |
| Clip space | Vertex output coordinates, −1…+1 |
| `fragCoord` | Fragment-stage pixel coordinates, top-left origin, pixel centers at .5 |
| Premultiplied alpha | RGB already multiplied by A; required by the canvas `alphaMode` used here |

## Appendix B: candidate plates (for the HTML pass)

1. **The two numbers.** `scale` and `offset` as a ruler of samples under a row of pixel columns.
2. **Adapter → device → buffers.** The resource tree the constructor builds, with sizes for the worked example.
3. **Twenty bytes.** The `Uniforms` layout with f32 and i32 views over one `ArrayBuffer`, plus the `652344b` bug.
4. **The pyramid.** Levels as stacked strips with byte costs (already partly in `pyramid-plates.html` Plate 01, so link to it).
5. **`selectLevel` bands.** Log-scale `scale` axis, level bands, bins-per-column sawtooth. Interactive `scale` slider.
6. **Bind groups as wiring.** Three pipelines, their slots, and which buffer plugs in where; `colMinMax` as the shared node.
7. **Workgroups.** 1600 px → 25 × 64, the 1601 px case, and the `x >= width` guard.
8. **One frame.** Command encoder timeline: `writeBuffer`s, the optional compute pass, the render pass, submit, present. Toggle "height changed" versus "scale changed".
9. **Column coverage.** Raw inclusive overlap versus pyramid bin rounding, with the §7.2 table as a live readout. Correct the ~3% figure here.
10. **Clip space → `fragCoord` → `yPosNorm`.** The coordinate systems, and the ε band at 1 versus 2 rows.
11. **Edges.** Storage limit, premultiplied alpha, `device.lost`, no `destroy()`.

---

*Sources: `packages/webgpu-waveform/src/{index,GPUWaveformRenderer,waveformShader,nullthrows}.ts`, `vite.config.ts`, `package.json` at `3a0b5a5`; `packages/webgpu-waveform-react/src/GPUWaveform.tsx`; `packages/site/src/Playground.tsx`. The coverage and ε figures come from a CPU port of the WGSL, not a GPU run. The storage limit and premultiplied-alpha behavior come from the WebGPU spec and haven't been confirmed in a browser yet.*
