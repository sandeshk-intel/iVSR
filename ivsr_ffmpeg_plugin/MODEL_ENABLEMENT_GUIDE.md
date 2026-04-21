# iVSR Model Enablement Guide

This guide is for **developers** who want to add new inference models to the iVSR FFmpeg plugin. For running existing models, see [README.md](README.md).

---

## Table of contents

1. [How the backend dispatches models](#1-how-the-backend-dispatches-models)
2. [Built-in models reference](#2-built-in-models-reference)
3. [Path A — Add a model via JSON config (no C changes)](#3-path-a--add-a-model-via-json-config-no-c-changes)
4. [Path B — Add a model with custom C I/O](#4-path-b--add-a-model-with-custom-c-io)
5. [JSON config field reference](#5-json-config-field-reference)
6. [Conversion scripts reference](#6-conversion-scripts-reference)
7. [Worked examples](#7-worked-examples)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. How the backend dispatches models

### The `ModelDesc` table

Every model the iVSR backend knows about is described by a `ModelDesc` struct row in `model_table[]` inside `dnn_backend_ivsr.c`. The struct captures everything the dispatch logic needs to handle that model — tensor layout, precision, frame window size, alignment, normalization convention — without any `if (model_type == X)` chains in the hot paths.

```
┌─────────────────────────────────────────────────────────────┐
│ ff_dnn_load_model_ivsr()                                    │
│   └─ get_model_desc()                                       │
│        ├─ model_type >= 0  →  &model_table[model_type]      │ ← built-in
│        └─ model_type == -1 →  dynamic_desc (parsed JSON)    │ ← config file
│                                                             │
│   Apply from descriptor:                                    │
│     • alignment (pad W/H to multiple)                       │
│     • tensor layout + precision (iVSR SDK config)           │
│     • color format (RGB / I420)                             │
│     • nif override (force number of input frames)           │
└─────────────────────────────────────────────────────────────┘
            │
            ▼  per frame
┌─────────────────────────────────────────────────────────────┐
│ fill_model_input_ivsr()                                     │
│   pack_input != NULL  → call custom function  (e.g. RIFE)  │
│   pack_input == NULL  → generic: ff_proc_from_frame_to_dnn  │
└─────────────────────────────────────────────────────────────┘
            │
            ▼  after inference
┌─────────────────────────────────────────────────────────────┐
│ infer_completion_callback()                                 │
│   unpack_output != NULL → call custom function              │
│   unpack_output == NULL → generic: ff_proc_from_dnn_to_frame│
└─────────────────────────────────────────────────────────────┘
```

### Five questions to ask before writing any code

Before touching a single file, answer these about your model:

| # | Question | Drives |
|---|---|---|
| 1 | How many input frames does one inference call consume? | `nif`, `window_type` |
| 2 | What tensor layout and element type does the model expect? | `in_layout`, `in_precision` |
| 3 | Is input normalisation standard (divide by 255) or raw [0, 255]? | `normalize_input` |
| 4 | Does output have the same spatial size as input, or is it upscaled? | `output_passthrough_dims` |
| 5 | Does the model require a second conditioning input? | Pre-fuse at export time (see §3) |

If questions 1–4 all have straightforward answers and question 5 is "no" or "yes, but bicubic/fixed transform", use **Path A** (JSON config). If the model has truly novel frame queuing, multi-frame output tensors, or packed multi-resolution data, use **Path B** (C code).

---

## 2. Built-in models reference

| model_type | Name | Task | Input shape | Window | In precision | Align | FFmpeg pix_fmt |
|---|---|---|---|---|---|---|---|
| 0 | BasicVSR | Multi-frame VSR ×4 | [1,F,H,W,C] NFHWC | `in_queue` nif=3 | u8/u16 | 32px | rgb24 |
| 1 | VideoProc | SVP preprocessing | [1,H,W,C] NHWC | single | u8/u16 | 64px | rgb24 or yuv420p |
| 2 | EDSR | Single-frame SR | [1,H,W,C] NHWC | single | u8/u16 | none | rgb24 |
| 3 | CustVSR | Custom single-frame VSR | [1,H,W,C] NHWC | single | u8/u16 | 64px | yuv420p |
| 4 | TSENet | 3-frame temporal SR | [1,9,H,W] NCHW | sliding nif=3 | u8/u16 | none | rgb24 |
| 5 | RIFE | Frame interpolation | [1,6,H,W] NCHW | sliding nif=2 | f32 [0,1] | 128px | rgb24 |
| 6 | VideoSeal | Invisible watermarking | [1,3,H,W] NCHW | single | f32 [0,255] | none | rgb24 |
| -1 | Custom (JSON) | Any | from JSON | from JSON | from JSON | from JSON | rgb24 |

> **model_type=-1** activates the JSON config path. Must be paired with `model_config=path/to/config.json`.

---

## 3. Path A — Add a model via JSON config (no C changes)

Use this path when your model fits into the generic dispatch (single or sliding window, standard normalization, one input tensor, one output tensor).

### Step 1 — Characterise your model's tensor contract

You need to know:

- **Input tensor shape**: use [Netron](https://netron.app) on the `.onnx`, or run `benchmark_app -m model.xml` from OpenVINO
- **Normalization**: does the PyTorch `forward()` start with `x = x / 255`? Check the source
- **Output size**: same as input (super-res produces larger output — check the ONNX output node shape)
- **Frame count**: how many frames does the `forward()` take?

### Step 2 — Export to OpenVINO IR

Every model needs a conversion script that produces `.xml` + `.bin` files compatible with the iVSR SDK.

**For single-input models (most cases):**

```bash
python3 mymodel_to_openvino.py --checkpoint mymodel.pth --scale 4 --output mymodel_x4
# → mymodel_x4.xml  mymodel_x4.bin
```

**For models with a second conditioning input (e.g. HDRTVNet++ AGCM):**

Wrap the model before tracing so the conditioning transform is baked into the graph:

```python
class ModelWithFusedCond(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Bake the conditioning (bicubic ×1/4) into the graph
        cond = F.interpolate(x, scale_factor=0.25, mode='bicubic', align_corners=False)
        return self.model(x, cond)

wrapped = ModelWithFusedCond(original_model)
torch.onnx.export(wrapped, dummy_input, "model_fused.onnx", ...)
```

The exported graph has a single input node. The iVSR backend sees it as a standard single-input model.

### Step 3 — Write the JSON config

Copy the template and fill in your values:

```bash
cp ivsr_model_config.template.json models/mymodel_config.json
# edit with IDE — schema validation is automatic if .vscode/settings.json is present
```

Example for a SPAN 4× SR model:

```json
{
  "name":        "SPAN-x4",
  "nif":         1,
  "align":       0,
  "in_layout":   "NCHW",
  "in_precision": "f32",
  "out_layout":  "NCHW",
  "out_precision": "fp32",
  "model_color": "RGB",
  "out_order":   "RGB",
  "window_type": "single",
  "window_init_dup":             false,
  "normalize_input":             true,
  "normalize_output":            true,
  "output_passthrough_dims":     false,
  "out_precision_depth_derived": false
}
```

See §5 for the full field reference.

### Step 4 — Run with FFmpeg

```bash
./ffmpeg -i input.mp4 \
  -vf "format=rgb24,dnn_processing=dnn_backend=ivsr:\
model=models/mymodel_x4.xml:input=input:output=output:\
model_type=-1:model_config=models/mymodel_config.json:device=GPU" \
  -pix_fmt yuv420p output.mp4
```

### Step 5 — Validate the config

```bash
pip install jsonschema
jsonschema -i models/mymodel_config.json ivsr_model_config.schema.json
# No output = valid
```

---

## 4. Path B — Add a model with custom C I/O

Use this path when your model needs behaviour the generic dispatch cannot express:
- Multi-frame output (like BasicVSR writing to `task->out_queue`)
- Non-standard frame queuing (copy-pad first frame multiple times, etc.)
- Packed multi-resolution or multi-scale tensors
- Truly custom normalization unlike anything in `normalize_input`/`normalize_output`

### Step 1 — Add an enum value

In `dnn_backend_ivsr.c`, extend `ModelType`:

```c
typedef enum {
    UNKNOWN_MODEL = -1,
    BASICVSR,
    VIDEOPROC,
    EDSR,
    CUSTVSR,
    TSENET,
    RIFE,
    VIDEOSEAL,
    MYMODEL,       // ← add here
    MODEL_TYPE_NUM // always last
} ModelType;
```

> The compile-time assertion `model_table_wrong_size` will immediately fail at the next build until you also add the table row (Step 3).

### Step 2 — Forward-declare the I/O functions

Near the existing forward declarations:

```c
static int  pack_input_mymodel  (IVSRModel *m, void *base, DNNData *in,  TaskItem *t);
static void unpack_output_mymodel(IVSRModel *m, TaskItem *t, DNNData *out);
```

### Step 3 — Add a row to `model_table[]`

```c
/* [MYMODEL] */ { "MyModel", 1, 1, 0, "NCHW", "f32", "NCHW", "fp32",
                  "RGB", DCO_RGB, pack_input_mymodel, unpack_output_mymodel },
```

Fields left-to-right: `name`, `nif_override`, `channel_divisor`, `align`, `in_layout`, `in_precision`, `out_layout`, `out_precision`, `model_color`, `out_order`, `pack_input`, `unpack_output`. See §5 for semantics.

After adding this row, the compile-time assertion passes again.

### Step 4 — Implement the I/O functions

**Function contracts:**

`pack_input` must:
- Return `0` on success, `DNN_MORE_FRAMES` if not enough frames yet, or `AVERROR(ENOMEM)` on allocation failure
- Reset `input->data = base` before returning `0` (the buffer pointer must point to the start of the tensor)
- Not free `base` — the request item owns it

`unpack_output` must:
- Write directly into `task->out_frame->data[0]`
- Not allocate persistent memory (use `av_malloc`/`av_free` for temporaries)

**Single-frame [0,1] float32 NCHW skeleton** (copy-and-adapt):

```c
static int pack_input_mymodel(IVSRModel *m, void *base, DNNData *input, TaskItem *task)
{
    (void)m;
    AVFrame *frame = task->in_frame;
    float   *dst   = (float *)input->data;
    int w = input->width, h = input->height, ps = w * h;

    for (int y = 0; y < h; y++) {
        uint8_t *row = frame->data[0] + y * frame->linesize[0];
        for (int x = 0; x < w; x++) {
            dst[0*ps + y*w + x] = row[x*3 + 0] / 255.0f;  // R plane
            dst[1*ps + y*w + x] = row[x*3 + 1] / 255.0f;  // G plane
            dst[2*ps + y*w + x] = row[x*3 + 2] / 255.0f;  // B plane
        }
    }
    input->data = base;  // REQUIRED: reset to buffer start
    return 0;
}

static void unpack_output_mymodel(IVSRModel *m, TaskItem *task, DNNData *output)
{
    (void)m;
    const float *src = (const float *)output->data;
    int fw = task->out_frame->width, fh = task->out_frame->height;

    for (int h = 0; h < fh; h++) {
        uint8_t *row = task->out_frame->data[0] + h * task->out_frame->linesize[0];
        for (int w = 0; w < fw; w++) {
            int ps = output->height * output->width;
            float r = src[0*ps + h*output->width + w];
            float g = src[1*ps + h*output->width + w];
            float b = src[2*ps + h*output->width + w];
            row[w*3 + 0] = (uint8_t)av_clip((int)(r * 255.0f + 0.5f), 0, 255);
            row[w*3 + 1] = (uint8_t)av_clip((int)(g * 255.0f + 0.5f), 0, 255);
            row[w*3 + 2] = (uint8_t)av_clip((int)(b * 255.0f + 0.5f), 0, 255);
        }
    }
}
```

**Useful internal helpers:**

| Helper | Purpose |
|---|---|
| `av_fifo_write(m->frame_queue, &frame, 1)` | Push a frame onto the sliding window queue |
| `av_fifo_read(m->frame_queue, &frame, 1)` | Pop oldest frame from queue |
| `av_fifo_peek(m->frame_queue, frames, n, 0)` | Read n frames without consuming |
| `av_fifo_can_read(m->frame_queue)` | Number of frames currently queued |
| `ff_proc_from_frame_to_dnn(frame, &input, filter_ctx)` | Generic NHWC packer (handles u8/u16 bit depths) |
| `convert_nhwc_to_nchw(data, N,C,H,W, type)` | In-place layout conversion |
| `convert_nchw_to_nhwc(data, N,C,H,W, type)` | In-place layout conversion |
| `m->sliding_window_frame_num` | Init counter for first-frame duplication logic |
| `DNN_MORE_FRAMES` | Return this from `pack_input` to signal "need more frames, skip inference" |

### Step 5 — Build and verify

```bash
make -C /path/to/ffmpeg -j$(nproc) 2>&1 | grep -E "error:|model_table"
# No errors expected. model_table_wrong_size fires immediately if row count mismatches.
```

---

## 5. JSON config field reference

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | yes | — | Human-readable label used in log messages |
| `nif` | integer ≥ 1 | yes | — | Number of input frames consumed per inference call |
| `align` | integer ≥ 0 | no | `0` | Pad input W and H up to a multiple of this (0 = none; RIFE uses 128) |
| `in_layout` | `"NCHW"` \| `"NHWC"` \| `"NFHWC"` | yes | — | Tensor layout for the iVSR SDK input tensor |
| `in_precision` | `"f32"` \| `"u8"` \| `"u16"` \| `null` | yes | — | Input element type; `null` = derive from frame bit depth |
| `out_layout` | `"NCHW"` \| `"NHWC"` \| `"NFHWC"` | yes | — | Output tensor layout |
| `out_precision` | `"fp32"` \| `"u8"` \| `"u16"` \| `null` | yes | — | Output element type; `null` = default `"fp32"` |
| `model_color` | `"RGB"` \| `"I420_Three_Planes"` \| `null` | no | `null` | Color space string passed to iVSR SDK; `null` = fallback auto-detect |
| `out_order` | `"RGB"` \| `"BGR"` \| `"NONE"` | no | `"RGB"` | Color order for `ff_proc_from_dnn_to_frame` output path |
| `window_type` | `"single"` \| `"sliding"` \| `"in_queue"` | yes | — | Frame queuing strategy (see table below) |
| `window_init_dup` | boolean | no | `false` | `sliding` only — duplicate first frame to prime the window (TSENet behaviour) |
| `normalize_input` | boolean | no | `true` | `true` = divide rgb24 uint8 by 255 before packing (RIFE/SPAN); `false` = raw [0,255] (VideoSeal) |
| `normalize_output` | boolean | no | `true` | `true` = multiply float32 output by 255 then clip to uint8; `false` = direct round+clip |
| `output_passthrough_dims` | boolean | no | `false` | `true` = output W/H = input W/H (VideoProc-style passthrough models) |
| `out_precision_depth_derived` | boolean | no | `false` | `true` = derive output precision from frame bit depth (EDSR-style 8/10/16-bit models) |

**`window_type` values explained:**

| Value | Behaviour | Use when |
|---|---|---|
| `"single"` | One input frame in, one output frame out, no queuing | SPAN, VideoSeal, EDSR, HDRTVNet++ LE |
| `"sliding"` | N-frame sliding window via `frame_queue`; first frame optionally duplicated | RIFE (nif=2), TSENet (nif=3, `window_init_dup=true`) |
| `"in_queue"` | Reads `nif` frames from `task->in_queue` (batched) | BasicVSR (nif=3) |

---

## 6. Conversion scripts reference

| Script | Model | Key flags | Output files | Notes |
|---|---|---|---|---|
| `rife_to_openvino.py` | RIFE | `--checkpoint`, `--height`, `--width` | `rife.xml` + `.bin` | Re-export if resolution changes |
| `videoseal/export_videoseal_openvino.py` | VideoSeal embedder | `--text`, `--height`, `--width` | `videoseal_baked_<H>p.xml` + `.bin` | Re-export per payload text and resolution |
| `videoseal/export_videoseal_detector_openvino.py` | VideoSeal detector | none | `videoseal_detector.xml` + `.bin` | Export once; resolution-independent |
| `span_to_openvino.py` *(planned)* | SPAN | `--checkpoint`, `--scale 2\|3\|4`, `--feature_channels` | `span_x<N>.xml` + `.bin` | One export per scale factor |
| `agcm_to_openvino.py` *(planned)* | HDRTVNet++ AGCM | `--checkpoint` | `agcm_fused.xml` + `.bin` | Wraps bicubic cond into the graph |
| `hdrtv_le_to_openvino.py` *(planned)* | HDRTVNet++ LE | `--checkpoint` | `hdrtv_le.xml` + `.bin` | Straightforward single-input export |

---

## 7. Worked examples

### Example 1 — SPAN 4× super-resolution (Path A)

```bash
# 1. Export
python3 span_to_openvino.py --checkpoint SPAN_x4.pth --scale 4 --output models/span_x4

# 2. Config already provided: models/span_x4.json

# 3. Run (GPU)
./ffmpeg -i input.mp4 \
  -vf "format=rgb24,dnn_processing=dnn_backend=ivsr:\
model=models/span_x4.xml:input=input:output=output:\
model_type=-1:model_config=models/span_x4.json:device=GPU" \
  -pix_fmt yuv420p output_4x.mp4
```

### Example 2 — HDRTVNet++ LE local enhancement (Path A)

LE is a standard single-frame model. Tensor convention is the same as VideoSeal: input raw float [0, 255], output raw float [0, 255]. The 16-bit HDR output is handled by setting `"out_precision_depth_derived": true` so u16 is used for 10-bit source.

```json
{
  "name": "HDRTVNet-LE",
  "nif": 1,
  "align": 0,
  "in_layout": "NCHW",  "in_precision": "f32",
  "out_layout": "NCHW", "out_precision": "fp32",
  "model_color": "RGB",  "out_order": "RGB",
  "window_type": "single",
  "normalize_input": false,
  "normalize_output": false,
  "output_passthrough_dims": false,
  "out_precision_depth_derived": true
}
```

### Example 3 — HDRTVNet++ full 3-stage pipeline

Chain three `dnn_processing` filters — no extra backend changes needed:

```bash
./ffmpeg -i input_sdr.mp4 \
  -vf "format=rgb24,\
    dnn_processing=dnn_backend=ivsr:model=agcm_fused.xml:\
      model_type=-1:model_config=models/agcm_config.json:device=GPU,\
    dnn_processing=dnn_backend=ivsr:model=hdrtv_ensemble.xml:\
      model_type=-1:model_config=models/ensemble_config.json:device=GPU,\
    dnn_processing=dnn_backend=ivsr:model=hdrtv_hg.xml:\
      model_type=-1:model_config=models/hg_config.json:device=CPU" \
  -pix_fmt yuv420p10le -color_primaries bt2020 -color_trc smpte2084 \
  -color_space bt2020nc output_hdr10.mp4
```

> **AGCM note:** `agcm_fused.xml` is exported using the `AGCMWithFusedCond` wrapper in `agcm_to_openvino.py`. The bicubic ×¼ conditioning is baked into the graph — no second input needed at runtime.

### Example 4 — Adding a future tiled-output model (Path B)

If your model produces `[1, N, 3, h, w]` tiled output (multiple sub-regions) the generic unpacker cannot handle it. Write a custom `unpack_output_tiled()` iterating over tile indices, accumulating into `task->out_frame` at the correct spatial offsets, and point the `model_table[]` row at it. The rest of the dispatch (load, pack_input, SDK call) can still use the generic path.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Output is grayscale / all one color | `normalize_output: true` but model outputs [0,255]; float×255 overflows | Set `"normalize_output": false` |
| Checkerboard / tiling artefacts | NCHW output fed to NHWC unpacker | Set `"out_layout": "NCHW"` in JSON or correct `in_layout` |
| `DNN_MORE_FRAMES` loops forever, no output | `window_type: "sliding"` but `nif` too high, or queue never primed | Verify `nif` matches model's actual input frame count |
| Colors shifted (wrong hue) | `model_color` mismatch — model trained on BGR, fed RGB | Try `"model_color": null` and inspect model's training preprocessing |
| Output dimensions unexpectedly 1×1 | `output_passthrough_dims: true` for an SR model | Remove or set to `false` |
| Compile error: negative array size `model_table_wrong_size` | `ModelType` enum and `model_table[]` row count differ | Add or remove the matching table row |
| Low quality at scene cuts (sliding window models) | Stale frames from previous scene remain in `frame_queue` | Insert a `null` frame flush or `select` filter before the DNN filter |
| `Failed to initialize ivsr engine` | OV model input shape does not match `align`-adjusted resolution | Double-check `align` value matches what the model was exported with |
| `Failed to get input dimensions` | Wrong `.xml` path or OV version mismatch | Verify model path and that OpenVINO version matches the one used at export |
