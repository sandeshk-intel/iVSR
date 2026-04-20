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
