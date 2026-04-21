# iVSR FFmpeg Plugin — Port to FFmpeg n8.1

**Date:** April 7, 2026  
**Base:** FFmpeg n8.1 (released 2026-03-16)  
**Prior base:** FFmpeg n7.1  

---

## Background

The iVSR FFmpeg plugin ships three patch files targeting FFmpeg n7.1:

| Patch | Description |
|---|---|
| `0001-This-commit-ports-ivsr-ffmpeg-plugin-from-n6.1-to-n7.patch` | Core port: adds `dnn_backend_ivsr.c/.h`, registers the iVSR DNN backend across the FFmpeg DNN subsystem, adds 10/16-bit pixel I/O, and configures the build system |
| `0002-Fix-10-bit-data-type-support.patch` | Fixes 10-bit data type handling in `dnn_backend_ivsr.c` |
| `0003-For-BasicVSR-make-the-inference-input-width-32-align.patch` | Aligns inference input width to 32 for the BasicVSR model |

When applying these patches to n8.1, three files in patch 0001 fail due to API changes between n7.1 and n8.1:

| File | Reason for failure |
|---|---|
| `configure` | Context lines changed substantially; the EXTERNAL_LIBRARY_LIST and `dnn_deps_any` sections were restructured |
| `libavfilter/dnn/dnn_interface.c` | n8.1 already includes the `av_opt_set_defaults()` call that the patch tries to add, causing a context mismatch |
| `libswscale/swscale_unscaled.c` | `SwsContext` renamed to `SwsInternal`; function signatures gained `const` qualifiers; `c->srcW` renamed to `c->opts.src_w` |

---

## Solution

A fourth patch file was created that covers n8.1 exactly, so no manual edits or `sed` commands are required at build time.

### New file: `patches/0004-n8.1-compat-dnn_interface-swscale.patch`

Contains three hunks targeting n8.1's exact source context:

#### 1. `configure`

Four additions to register `libivsr` in the FFmpeg build system:

- **Help text** (after the `--enable-libopenvino` entry):
  ```
  --enable-libivsr         enable iVSR SDK as a DNN module backend
                           for DNN based Super Resolution filters [no]
  ```
- **`EXTERNAL_LIBRARY_LIST`** — adds `libivsr` after `libopenvino`
- **`dnn_deps_any`** — adds `libivsr` as an optional DNN backend dependency:
  ```
  dnn_deps_any="libtensorflow libopenvino libtorch libivsr"
  ```
- **Library detection** (before the `libopus` block):
  ```
  enabled libivsr && require libivsr ivsr.h ivsr_init -livsr
  ```

#### 2. `libavfilter/dnn/dnn_interface.c`

- Adds `extern const DNNModule ff_dnn_backend_ivsr;`
- Registers the iVSR backend in `dnn_backend_info_list` under `#if CONFIG_LIBIVSR`
- Adds `ff_dnn_uninit_child_class()` function for releasing per-backend option memory

#### 3. `libswscale/swscale_unscaled.c`

Adds four pixel format conversion functions using n8.1's `SwsInternal` type and `c->opts.src_w` field:

| Function | Conversion |
|---|---|
| `uint16_y_to_float_y_wrapper` | `GRAY16` → `GRAYF32` (normalise by 1/65535) |
| `float_y_to_uint16_y_wrapper` | `GRAYF32` → `GRAY16` (scale × 65535, clamp) |
| `uint10_y_to_float_y_wrapper` | `GRAY10` → `GRAYF32` (normalise by 1/1023) |
| `float_y_to_uint10_y_wrapper` | `GRAYF32` → `GRAY10` (scale × 1023, clamp to [0,1023]) |

Dispatch entries for each conversion are added in `ff_get_unscaled_swscale()`.

---

## Changes to `build.sh` — `build_ffmpeg()`

### FFmpeg target version

Changed from `n7.1` to `n8.1`:
```bash
ffmpeg_tag=n8.1
```

### Patch application strategy

Patch 0001's three incompatible files are excluded via `filterdiff`, and patch 0004 handles them instead. The order of application is:

```
filterdiff -x '*/configure'         \
           -x '*/dnn_interface.c'   \
           -x '*/swscale_unscaled.c' \
           0001-*.patch | git apply --3way --ignore-whitespace -

git apply --ignore-whitespace 0004-*.patch

git apply --3way --whitespace=fix 0002-*.patch
git apply --3way --whitespace=fix 0003-*.patch
git apply --3way --whitespace=fix 0005-*.patch
git apply --3way --whitespace=fix 0006-*.patch
git apply --3way --whitespace=fix 0007-*.patch
git apply --3way --whitespace=fix 0008-*.patch
```

All `sed` commands that had previously been used to patch `configure` at build time have been removed — that logic is now fully encoded in patch 0004.

### Configure flags

```bash
./configure \
    --enable-gpl \
    --enable-nonfree \
    --disable-static \
    --disable-doc \
    --enable-shared \
    --enable-version3 \
    --enable-libivsr \
    --enable-libx264 \
    --enable-libx265
```

---

## Verification

After a clean build from the updated `build.sh`:

| Check | Result |
|---|---|
| `ffmpeg -version` | `n8.1` |
| `--enable-libivsr` in configuration | ✓ |
| `CONFIG_LIBIVSR=yes` in `config.mak` | ✓ |
| `-livsr` in `EXTRALIBS-avfilter` | ✓ |
| `dnn_processing` filter registered | ✓ |
| Build exit code | `0` |

---

## Patch Application Summary

| Patch | Applied via | Files touched |
|---|---|---|
| `0001` (filtered) | `filterdiff` + `git apply --3way` | `Makefile`, `dnn_backend_common.c/h`, `dnn_backend_ivsr.c` *(new)*, `dnn_backend_ivsr.h` *(new)*, `dnn_io_proc.c`, `dnn_filter_common.c`, `dnn_interface.h`, `vf_dnn_processing.c` |
| `0004` | `git apply` | `configure`, `dnn_interface.c`, `swscale_unscaled.c` |
| `0002` | `git apply --3way` | `dnn_backend_ivsr.c` |
| `0003` | `git apply --3way` | `dnn_backend_ivsr.c` |
| `0005` | `git apply --3way` | `dnn_backend_ivsr.c` |
| `0006` | `git apply --3way` | `dnn_backend_ivsr.c` |
| `0007` | `git apply --3way` | `dnn_backend_ivsr.c`, `dnn_interface.h` |
| `0008` | `git apply --3way` | `ivsr_model_config.template.json` *(new)*, `ivsr_model_config.schema.json` *(new)*, `models/span_x4.json` *(new)* |

---

## VideoSeal Watermarking + ModelDesc Refactor — Patch 0006

### New file: `patches/0006-Add-VideoSeal-watermarking-model-support.patch`

Adds invisible watermarking support for the VideoSeal model and refactors `dnn_backend_ivsr.c` to use a centralised per-model descriptor table, making future model additions a single-location change.

#### Changes summary

**`ModelType` enum**
- Adds `VIDEOSEAL` (invisible watermarking, 1-frame 3-channel input, baked payload) before `MODEL_TYPE_NUM`.

**`ModelDesc` descriptor table (new)**
- Introduces `struct ModelDesc` with fields: `name`, `nif_override`, `channel_divisor`, `align`, `in_layout`, `in_precision`, `out_layout`, `out_precision`, `model_color`, `out_order`, `pack_input` (fn ptr), `unpack_output` (fn ptr).
- A static `model_table[]` array indexed by `ModelType` holds one row per model — all per-model configuration is in one place.
- A compile-time `sizeof` assert ensures the table stays in sync with the enum (missing a row is a build error).

**Per-model I/O functions (new)**
- `pack_input_rife` / `unpack_output_rife` — extracted from the previous inline RIFE code.
- `pack_input_videoseal` — packs `rgb24` → NCHW float32 **without** `/255` (the model has `/255` baked in via `VideoSealFinalWrapper`).
- `unpack_output_videoseal` — converts NCHW float32 `[0, 255]` → packed `rgb24` (model already multiplied by 255 internally); clips with `av_clip(r + 0.5f)`.

**`fill_model_input_ivsr()`**
- Channel divisor now read from `model_table[model_type].channel_divisor` — replaces the TSENET/RIFE if/else chain.
- Model-specific input packing replaced by a single `model_table[model_type].pack_input(...)` dispatch; BASICVSR and TSENet retain their inline paths (they use `ff_proc_from_frame_to_dnn`).

**`infer_completion_callback()`**
- `output.order` now read from `model_table[model_type].out_order` — replaces the switch statement.
- Output unpacking replaced by `model_table[model_type].unpack_output(...)` dispatch; generic `ff_proc` path is the fallback for models with a `NULL` function pointer.

**`get_input_ivsr()`**
- Channel divisor read from `model_table[model_type].channel_divisor`.

**`ff_dnn_load_model_ivsr()`**
- Layout/precision, alignment, model colour format, and `nif` override all driven by `model_table` — replaces five separate if/else chains.

#### Adding a new model after this patch

1. Add an enum value to `ModelType`.
2. Add one row to `model_table[]`.
3. Optionally write `pack_input_<name>` / `unpack_output_<name>` if the generic `ff_proc` path is insufficient; otherwise set the pointers to `NULL`.

No other changes are needed anywhere in the file.

#### Usage

Export the model (bakes payload + normalisation):
```bash
cd sandesh/iVSR
python3 videoseal/export_videoseal_openvino.py --text "YOUR_WATERMARK" --height 720 --width 1280
```

Run watermarking (input must be 1280×720; scale beforehand if needed):
```bash
LD_LIBRARY_PATH=<build_dir>/libavfilter \
<build_dir>/ffmpeg \
  -i input.mp4 \
  -vf "scale=1280:720,format=rgb24,dnn_processing=dnn_backend=ivsr:model=/path/to/videoseal_baked_720p.xml:model_type=6:nif=1:device=CPU:normalize_factor=1.0" \
  -c:v libx264 -crf 18 -pix_fmt yuv420p output_watermarked.mp4
```

`model_type=6` corresponds to the `VIDEOSEAL` enum value.

> **Note:** `LD_LIBRARY_PATH` is required when the system `/usr/local/lib/libavfilter.so.11` is an older build. Run `sudo make install && sudo ldconfig` from the FFmpeg build directory to install the updated library system-wide and avoid it.

---

## RIFE Model Support — Patch 0005

### New file: `patches/0005-Add-RIFE-model-support-to-dnn_backend_ivsr.patch`

Adds frame interpolation support for the RIFE model to `libavfilter/dnn/dnn_backend_ivsr.c`.

#### Changes summary

**`ModelType` enum**
- Adds `RIFE` (frame interpolation, 2-frame 6-channel input) before `MODEL_TYPE_NUM`.

**`IVSRModel` struct**
- Adds `rife_frame_num` sliding-window init counter (parallel to the existing `tsenet_frame_num` which replaces the former `static int frame_num`).
- The pre-existing `static int frame_num` variable in `fill_model_input_ivsr` is replaced by the instance field `tsenet_frame_num`, fixing a data-race / multi-instance bug.

**`fill_model_input_ivsr()`**
- For `RIFE`: divides the reported channel count by `nif` (2) so each `ff_proc_from_frame_to_dnn` call fills exactly one frame's worth of channels.
- Adds RIFE sliding-window input path: queues 2 consecutive frames, then manually packs them into the NCHW float tensor as normalized `[0, 1]` RGB planes — bypassing `sws_scale` which would collapse RGB to luma and produce incorrect value ranges.

**`infer_completion_callback()`**
- Adds `RIFE` to the `DCO_RGB` output order switch.
- Adds RIFE-specific output path: converts float32 NCHW/NHWC `[0, 1]` tensor back to packed `rgb24` — bypassing `ff_proc_from_dnn_to_frame` which applies luma range expansion causing wave/colour artefacts.
- Fixes a latent bug for other models: the `DL_NCHW → DL_NONE` layout reset after `convert_nchw_to_nhwc` is now applied so downstream sws stride calculations are correct.

**`get_input_ivsr()`**
- For `RIFE`: reports 3 channels (halves the model's 6-channel input) so the filter graph accepts standard `rgb24` frames.

**`ff_dnn_load_model_ivsr()`**
- Sets `NCHW` / `f32` layout and precision for both input and output tensors when model type is `RIFE`.
- Enforces 128-aligned `frame_h` / `frame_w` for the RIFE shape string (model was exported with 128-aligned dimensions).
- Adds `RIFE` to the `RGB` color format switch.
- Sets `ivsr_model->nif = 2` for RIFE (hard-coded, same pattern as TSENet's `nif = 3`).

#### Usage

```
ffmpeg -i input.mp4 \
  -vf "scale,format=rgb24,split[a][b]; \
       [b]dnn_processing=dnn_backend=ivsr:model=rife.xml:model_type=5[ri]; \
       [ri]setpts=PTS+1/(2*FRAME_RATE*TB)[interp]; \
       [a][interp]interleave=nb_inputs=2:duration=longest" \
  output.mp4
```

`model_type=5` corresponds to the `RIFE` enum value.

---

## Complete model_table refactor + JSON config system — Patches 0007 & 0008

### New files

| File | Description |
|---|---|
| `patches/0007-Complete-model-table-refactor-and-JSON-config.patch` | All C source changes (both phases below) |
| `patches/0008-Add-JSON-model-config-template-schema-and-SPAN-example.patch` | New JSON artifact files shipped alongside the plugin |
| `ivsr_model_config.template.json` | Copy-and-edit template with inline `_comment_*` annotations for every supported field |
| `ivsr_model_config.schema.json` | JSON Schema draft-07 for IDE validation and autocomplete |
| `models/span_x4.json` | First ready-to-use config: SPAN 4× super-resolution |

---

### Phase 1 — Complete the model_table refactor (Patch 0007 Part A)

Patches 0005 and 0006 introduced the `model_table[]` pattern for RIFE and VideoSeal. Patch 0007 Part A migrates the five remaining hardcoded `if (model_type == X)` chains for BasicVSR, TSENet, VideoProc, CustVSR, and EDSR into the table.

#### Extended `ModelDesc` — 4 new fields

| Field | Type | Purpose |
|---|---|---|
| `out_precision_depth_derived` | `int` | `1` = derive output tensor precision from frame bit-depth (replaces EDSR hardcode: u8 for 8-bit, u16 for 10/16-bit) |
| `output_passthrough_dims` | `int` | `1` = report output W/H = input W/H regardless of tensor (replaces VideoProc `get_output_ivsr` hardcode) |
| `color_format_auto` | `int` | `0` = use `model_color` string; `1` = VideoProc-auto (RGB vs I420 from pixel format); `2` = CustVSR-auto (always I420) |
| `sliding_window_init_dup` | `int` | `1` = duplicate first frame to prime the queue on init (replaces TSENet-specific dup logic) |

#### New per-model I/O functions

| Function | Replaces |
|---|---|
| `pack_input_basicvsr` | BASICVSR inline branch in `fill_model_input_ivsr()` (reads `nif` frames from `task->in_queue`) |
| `pack_input_tsenet` | TSENET inline branch in `fill_model_input_ivsr()` (sliding window with first-frame dup) |
| `unpack_output_basicvsr` | BASICVSR inline loop in `infer_completion_callback()` (multi-frame `task->out_queue` iteration) |

#### Counter unification

`tsenet_frame_num` and `rife_frame_num` in `IVSRModel` collapsed into a single `sliding_window_frame_num` (safe: only one sliding-window model can be active per filter instance).

#### Updated `model_table[]` (post-Patch-0007)

| Enum | name | nif_override | channel_divisor | align | in_layout | in_precision | out_layout | pack_input | unpack_output | Notable flags |
|---|---|---|---|---|---|---|---|---|---|---|
| BASICVSR | BasicVSR | 0 (SDK) | 1 | 32 | NFHWC | null | NFHWC | `pack_input_basicvsr` | `unpack_output_basicvsr` | — |
| VIDEOPROC | VideoProc | 0 (SDK) | 1 | 64 | NHWC | null | NHWC | null (generic) | null (generic) | `color_format_auto=1`, `output_passthrough_dims=1` |
| EDSR | EDSR | 0 (SDK) | 1 | 0 | NHWC | null | NHWC | null (generic) | null (generic) | `out_precision_depth_derived=1` |
| CUSTVSR | CustVSR | 0 (SDK) | 1 | 64 | NHWC | null | NHWC | null (generic) | null (generic) | `color_format_auto=2` |
| TSENET | TSENet | 3 | 3 | 0 | NCHW | null | NHWC | `pack_input_tsenet` | null (generic) | `sliding_window_init_dup=1` |
| RIFE | RIFE | 2 | 2 | 128 | NCHW | f32 | NCHW | `pack_input_rife` | `unpack_output_rife` | — |
| VIDEOSEAL | VideoSeal | 1 | 1 | 0 | NCHW | f32 | NCHW | `pack_input_videoseal` | `unpack_output_videoseal` | — |

**Result:** Zero `if (model_type == X)` chains remain in any hot-path function. Adding a new built-in model = one enum value + one `model_table[]` row + optional I/O functions. Compile-time assert (`model_table_wrong_size`) still guards row count.

---

### Phase 2 — JSON config system (Patch 0007 Part B)

Adds the ability to run a new model without any C changes or recompilation: supply a `.json` descriptor file and set `model_type=-1`.

#### `iVSROptions` changes (`dnn_interface.h`)

New field: `char *model_config` — path to a JSON model descriptor file.

#### `ModelType` enum change

`UNKNOWN_MODEL = -1` renamed to `CUSTOM = -1` to reflect its new purpose.

#### New `AVOption`

```c
{ "model_config",
  "Path to a JSON model descriptor file. Use with model_type=-1. "
  "Specifies tensor layout, precision, nif, align, window_type and normalisation "
  "for models not built into the iVSR backend. "
  "See ivsr_model_config.template.json for all supported fields.",
  OFFSET(model_config), AV_OPT_TYPE_STRING, {.str=NULL}, 0, 0, FLAGS }
```

The `model_type` option's minimum value changed from `0` to `-1`.

#### New `WindowType` enum

| Value | Meaning |
|---|---|
| `WINDOW_SINGLE` | One frame per inference, no queue (RIFE/VideoSeal/SPAN style) |
| `WINDOW_SLIDING` | N-frame sliding window via `frame_queue` |
| `WINDOW_IN_QUEUE` | Read `nif` frames from `task->in_queue` (BasicVSR style) |

#### Extended `ModelDesc` (Patch 0008 additions)

| Field | Type | Purpose |
|---|---|---|
| `window_type` | `WindowType` | Frame queuing strategy for JSON-loaded models |
| `normalize_input` | `int` | `1` = divide uint8 by 255 before packing into float32 (RIFE/SPAN); `0` = raw [0,255] (VideoSeal) |
| `normalize_output` | `int` | `1` = multiply float32 by 255 before clipping to uint8; `0` = direct round-and-clip |

#### New functions

| Function | Purpose |
|---|---|
| `parse_model_config_json(ctx, path, &desc)` | Flat JSON parser (~200 lines, no external deps, uses `avio_open`). Keys beginning with `_` are silently ignored (comment support). Returns `AVERROR` on missing required fields. |
| `pack_input_window(m, base, input, task)` | Generic input packing dispatcher — branches on `md->window_type` and applies `normalize_input`. Shared by all JSON-loaded models. |
| `unpack_output_generic_fp32(m, task, output)` | Generic output unpacking — NCHW or NHWC float32 → rgb24; applies `normalize_output`. |
| `free_dynamic_desc(&desc)` | Frees heap-allocated strings in a `ModelDesc` created by the JSON parser, then the struct itself. |
| `get_model_desc(ivsr_model)` | **Central lookup helper** — returns `dynamic_desc` when present (JSON-loaded), else `&model_table[model_type]`. This is the only place `model_table` is accessed; all hot paths call through it. |

#### `IVSRModel` struct additions

- `ModelDesc *dynamic_desc` — heap-allocated descriptor for JSON-loaded models; `NULL` for built-in models.

#### Load flow for `model_type=-1`

```
ff_dnn_load_model_ivsr()
  └─ model_type == CUSTOM?
       └─ parse_model_config_json()  →  ivsr_model->dynamic_desc
  └─ get_model_desc()  ──►  dynamic_desc          (JSON-loaded)
                       └──►  &model_table[type]    (built-in)
  └─ apply: align, layout, precision, color_format, nif_override
```

#### JSON config field reference

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | string | — | **Required.** Used in log messages |
| `nif` | integer | 1 | Frames consumed per inference |
| `align` | integer | 0 | W/H alignment in pixels (0 = none) |
| `in_layout` | string | — | **Required.** `NCHW` \| `NHWC` \| `NFHWC` |
| `in_precision` | string\|null | null | `f32` \| `u8` \| `u16` \| null (inherit from bit-depth) |
| `out_layout` | string | `NCHW` | `NCHW` \| `NHWC` \| `NFHWC` |
| `out_precision` | string\|null | null | `fp32` \| `u8` \| `u16` \| null |
| `model_color` | string\|null | null | `RGB` \| `I420_Three_Planes` \| null |
| `out_order` | string | `RGB` | `RGB` \| `BGR` \| `NONE` |
| `window_type` | string | `single` | `single` \| `sliding` \| `in_queue` |
| `window_init_dup` | boolean | false | Dup first frame on init (`sliding` only, TSENet behaviour) |
| `normalize_input` | boolean | true | Divide uint8 by 255 → float32 [0,1] |
| `normalize_output` | boolean | true | Multiply float32 by 255 → uint8 |
| `output_passthrough_dims` | boolean | false | Output W/H = input W/H (VideoProc style) |
| `out_precision_depth_derived` | boolean | false | Derive output precision from frame bit-depth (EDSR style) |
| `color_format_auto` | integer | 0 | 0 = fixed; 1 = VideoProc-auto; 2 = CustVSR-auto |

#### Example usage — SPAN 4× super-resolution

```bash
# Run with JSON config (no recompile, no C changes)
ffmpeg -i input.mp4 \
  -vf "format=rgb24,dnn_processing=dnn_backend=ivsr:model=span_x4.xml:\
model_type=-1:model_config=models/span_x4.json" \
  output.mp4
```

#### Model configs for existing built-in models

| Model | `model_type` | Usable via JSON? | Blocker |
|---|---|---|---|
| Enhanced EDSR | 2 | ✅ Yes | None — `out_precision_depth_derived: true`, generic ff_proc path |
| SVP / VideoProc | 1 | ✅ Yes | None — `color_format_auto: 1`, `output_passthrough_dims: true` |
| Enhanced BasicVSR | 0 | ⚠️ Input only | `unpack_output_basicvsr` multi-frame `out_queue` loop has no generic JSON equivalent yet |
| TSENet | 4 | ✅ Yes | `window_type: sliding`, `window_init_dup: true`, `nif: 3` |
| RIFE | 5 | ✅ Yes | `window_type: sliding`, `normalize_input: true`, `normalize_output: true`, `nif: 2` |
| VideoSeal | 6 | ✅ Yes | `window_type: single`, `normalize_input: false`, `normalize_output: false` |

