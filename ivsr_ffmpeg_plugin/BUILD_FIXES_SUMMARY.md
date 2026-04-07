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
