# iVSR FFmpeg plugin - iVSR SDK based
The folder `ivsr_ffmpeg_plugin` enables model inference using FFmpeg with iVSR SDK as backend. It provides additional `ivsr` backend for the DNN interface called by the `dnn_processing` filter.<br>
The patches included in `patches` folder are specifically for FFmpeg n8.1.<br>

<div align=center>
<img src="./figs/ffmpeg_ivsr_sdk_backend.png" width = 80% height = 80% />
</div>

## How to run inference with FFmpeg-plugin
To run inference with iVSR SDK, you need to specify `ivsr` as the backend for the `dnn_processing` filter. Here is an example of how to do it: `dnn_processing=dnn_backend=ivsr`. <br>
Additionally, there are other parameters that you can use. These parameters are listed in the table below:<br>

|AVOption name|Description|Default value|Recommended value(s)|
|:--|:--|:--|:--|
|dnn_backend|DNN backend framework name|native|ivsr|
|model|path to model file|NULL|Available full path of the released model files|
|input|input name of the model|NULL|input|
|output|output name of the model|NULL|output|
|device|device for inference task|CPU|CPU or GPU|
|model_type|type for built-in models; `-1` = external JSON config via `model_config`|0|0 Enhanced BasicVSR, 1 SVP, 2 Enhanced EDSR, 3 CustVSR, 4 TSENet, 5 RIFE, 6 VideoSeal, **-1 JSON config (e.g. SPAN)**|
|model_config|path to a JSON model descriptor file — use with `model_type=-1`; see `ivsr_model_config.template.json` for all fields|NULL|path to a `.json` config file|
|normalize_factor|normalizing factor for models that do not require input normalization to [0, 1]|1.0|255.0 for Enhanced EDSR, 1.0 for all other models|
|num_streams|number of execution streams for throughput mode (valid only for GPU devices)|1|use `benchmark_app` to determine the best value|
|extension|extension lib file full path, required for Enhanced BasicVSR|—|—|
|op_xml|custom op xml file full path, required for Enhanced BasicVSR|—|—|
|nif|number of input frames in batch sent to the DNN backend|1|3 for Enhanced BasicVSR, 1 for other models|
|nireq|number of infer requests|0|leave as default or set to match CPU core count|

Here are some examples of FFmpeg command lines to run inference with the supported models using the `ivsr` backend.<br>

- Command sample to run Enhanced BasicVSR inference, the input pixel format supported by the model is `rgb24`.
```
cd <iVSR project path>/ivsr_ffmpeg_plugin/ffmpeg
./ffmpeg -i <your test video> -vf format=rgb24,dnn_processing=dnn_backend=ivsr:model=<basic_vsr_model.xml>:input=input:output=output:nif=3:device=<CPU or GPU>:extension=<iVSR project path>/ivsr_ov/based_on_openvino_2022.3/openvino/bin/intel64/Release/libcustom_extension.so:op_xml=<iVSR project path>/ivsr_ov/based_on_openvino_2022.3/openvino/flow_warp_cl_kernel/flow_warp.xml test_out.mp4
```
Please note that for the Enhanced BasicVSR model, you need to set the `extension` and `op_xml` options (with `backend_configs`) in the command line. After applying OpenVINO's patches and building OpenVINO, the extension lib file is located in `<OpenVINO folder>/openvino/bin/intel64/Release/libcustom_extension.so`, and the op xml file is located in `<OpenVINO folder>/openvino/flow_warp_cl_kernel/flow_warp.xml`.<br>

- Command sample to run SVP models inference. If the supported input pixel format of the model variance is `rgb24`, set the preceeding format as is to avoid unnecessary layout conversion:
```
cd <iVSR project path>/ivsr_ffmpeg_plugin/ffmpeg
./ffmpeg -i <your test video> -vf format=rgb24,dnn_processing=dnn_backend=ivsr:model=<svp_model.xml>:input=input:output=output:nif=1:device=<CPU or GPU>:model_type=1 -pix_fmt yuv420p test_out.mp4
```
If the model variance supports Y-input, set the preceeding format as YUV:
```
./ffmpeg -i <your test video> -vf format=yuv420p,dnn_processing=dnn_backend=ivsr:model=<svp_model.xml>:input=input:output=output:nif=1:device=<CPU or GPU>:model_type=1 -pix_fmt yuv420p test_out.mp4
```
- Command sample to run Enhanced EDSR inference, the input pixel format supported by the model is `rgb24`.
```
cd <iVSR project path>/ivsr_ffmpeg_plugin/ffmpeg
./ffmpeg -i <your test video> -vf format=rgb24,dnn_processing=dnn_backend=ivsr:model=<edsr_model.xml>:input=input:output=output:nif=1:device=<CPU or GPU>:model_type=2:normalize_factor=255.0 -pix_fmt yuv420p test_out.mp4
```
- Command sample to run CUSTOM VSR inference. Note the input pixel format supported by this model is `yuv420p`, and its input shape is `[1, (Y channel)1, H, W]`, output shape is `[1, 1, 2xH, 2xW]`.
```
cd <iVSR project path>/ivsr_ffmpeg_plugin/ffmpeg
./ffmpeg -i <your test video> -vf format=yuv420p,dnn_processing=dnn_backend=ivsr:model=<customvsr_model.xml>:input=input:output=output:nif=1:nireq=1:device=CPU:model_type=3 -pix_fmt yuv420p test_out.mp4
```
- Command sample to run TSENet model, the input pixel format supported by the model is `rgb24`.
```
cd <iVSR project path>/ivsr_ffmpeg_plugin/ffmpeg
./ffmpeg -i <your test video> -vf format=rgb24,dnn_processing=dnn_backend=ivsr:model=<tsenet_model.xml>:input=input:output=output:nif=1:device=<CPU or GPU>:model_type=4 -pix_fmt yuv420p test_out.mp4
```
- Command sample to run RIFE frame interpolation inference (doubles frame rate), the input pixel format supported by the model is `rgb24`. Export the model with `export_rife_openvino.py` first.
```
cd <iVSR project path>/ivsr_ffmpeg_plugin/ffmpeg
./ffmpeg -i <your test video> \
  -vf "scale,format=rgb24,split[a][b]; \
       [b]dnn_processing=dnn_backend=ivsr:model=<rife_model.xml>:model_type=5:nif=2:device=CPU[ri]; \
       [ri]setpts=PTS+1/(2*FRAME_RATE*TB)[interp]; \
       [a][interp]interleave=nb_inputs=2:duration=longest" \
  test_out.mp4
```
- Command sample to run VideoSeal invisible watermarking, the input pixel format supported by the model is `rgb24`. Export the model with `videoseal/export_videoseal_openvino.py` first; the model has the watermark payload and normalization baked in so **`normalize_factor=1.0`** must be used and the video must match the exported resolution.
```
cd <iVSR project path>/ivsr_ffmpeg_plugin/ffmpeg
./ffmpeg -i <your test video> \
  -vf "scale=1280:720,format=rgb24,dnn_processing=dnn_backend=ivsr:model=<videoseal_baked_720p.xml>:model_type=6:nif=1:device=CPU:normalize_factor=1.0:nireq=4" \
  -c:v libx264 -crf 18 -pix_fmt yuv420p test_out_watermarked.mp4
```
For bit-exact lossless output (recommended when extraction accuracy is critical), use FFV1 with `rgb24` pixel format instead.
`yuv420p` performs chroma subsampling which degrades the subtle per-pixel watermark signal even when using a lossless codec:
```
  -c:v ffv1 -level 3 -pix_fmt rgb24 test_out_watermarked.mkv
```
- Command sample to run **SPAN single-image super-resolution** (2× or 4×) using the JSON config system (`model_type=-1`).

  **Step 1 — Export the SPAN checkpoint to OpenVINO IR** (run once per scale factor):
  ```bash
  cd <iVSR project path>/span
  python3 export_span_openvino.py --weights spanx4_ch48.pth --scale 4 --channels 48
  # → spanx4_ch48_ir/spanx4_ch48.xml + spanx4_ch48_ir/spanx4_ch48.bin

  # For 2× upscaling:
  python3 export_span_openvino.py --weights spanx2_ch48.pth --scale 2 --channels 48
  # → spanx2_ch48_ir/spanx2_ch48.xml + .bin
  ```

  **Step 2 — Write the JSON config** (copy `ivsr_model_config.template.json` or use this minimal version):
  ```json
  {
    "name": "SPAN-x4",
    "nif": 1,
    "align": 0,
    "in_layout": "NCHW",
    "in_precision": "f32",
    "out_layout": "NCHW",
    "out_precision": "fp32",
    "model_color": "RGB",
    "out_order": "RGB",
    "window_type": "single",
    "window_init_dup": false,
    "normalize_input": true,
    "normalize_output": true,
    "output_passthrough_dims": false,
    "out_precision_depth_derived": false
  }
  ```
  For 2× change `"name"` to `"SPAN-x2"` only; all other fields are identical.

  **Step 3 — Run inference:**
  ```bash
  cd <iVSR project path>/ivsr_ffmpeg_plugin/ffmpeg
  ./ffmpeg -i <your test video> \
    -vf "format=rgb24,dnn_processing=dnn_backend=ivsr:\
  model=<path>/spanx4_ch48.xml:input=input:output=output:\
  model_type=-1:model_config=<path>/span_x4.json:device=CPU" \
    -pix_fmt yuv420p test_out_span4x.mp4
  ```
  The output resolution is automatically set by the model (e.g. 480p → 1920p for 4×).

  > **Note:** The Streamlit demo app (*Single-Image SR (SPAN)* tab) auto-generates the JSON config and runs
  > this command — just upload the `.xml` / `.bin` pair and input video.

See [JSON Model Config](#json-model-config-system) section below for full details.

## VideoSeal — Two Separate OpenVINO IR Models

VideoSeal uses two completely independent neural networks that are exported as separate OpenVINO IR files.

### Why two separate IRs?

| | Embedder IR | Detector IR |
|:--|:--|:--|
| **Purpose** | Embed watermark into video frames | Recover watermark from watermarked video |
| **Architecture** | Custom encoder-decoder (image restoration network) | ConvNeXt-V2 backbone + PixelDecoder + Linear head |
| **Input shape** | `[1, 3, H, W]` NCHW float32, values `[0, 255]` | `[1, 3, 256, 256]` NCHW float32, values `[0, 1]` |
| **Output shape** | `[1, 3, H, W]` float32, values `[0, 255]` | `[1, 257]` raw logits |
| **Resolution** | Fixed at export time (e.g. 720p, 1080p) | Always 256×256 — resolution-independent |
| **Payload** | Baked in at export time (text → 256-bit string) | Not applicable — reads bits from any watermarked frame |
| **Used by** | FFmpeg `dnn_processing` filter (model_type=6) | Python extractor / Streamlit Extract tab |
| **Export script** | `videoseal/export_videoseal_openvino.py` | `videoseal/export_videoseal_detector_openvino.py` |
| **Output files** | `videoseal_baked_<H>p.xml` + `.bin` | `videoseal_detector.xml` + `.bin` (~64 MB) |

The two networks are **never used together at inference time** — the embedder runs inside FFmpeg per-frame during watermark insertion; the detector runs as a standalone Python/OpenVINO process during watermark extraction.

### Embedder IR — export and use

The embedder bakes the payload text and the `÷255` / `×255` normalization permanently into the graph, so FFmpeg can pass raw `[0, 255]` pixel values directly with `normalize_factor=1.0`.

The exported model is **resolution-specific** and **payload-specific**: it must be re-exported whenever the target resolution or watermark text changes.

```bash
# Export for 720p with payload "IVSR"
python3 videoseal/export_videoseal_openvino.py --text "IVSR" --height 720 --width 1280
# → videoseal_baked_720p.xml + videoseal_baked_720p.bin

# Export for 1080p with a different payload
python3 videoseal/export_videoseal_openvino.py --text "MyStudio" --height 1080 --width 1920
# → videoseal_baked_1080p.xml + videoseal_baked_1080p.bin
```

**Re-export is required when:**
- Changing the watermark text (e.g. per-content or per-customer identifier)
- Changing the output resolution (the model input shape is fixed at export time)
- The same detector IR (`videoseal_detector.xml`) works for all of the above — only the embedder changes

**Payload encoding:** the `--text` string is UTF-8 encoded to binary at 8 bits per character and zero-padded or truncated to 256 bits. Maximum payload is 32 ASCII characters (256 bits ÷ 8).

### Detector IR — export and use

The detector always resizes its input to 256×256 internally (as in the original `model.detect()` API), so the exported model uses a **static `[1, 3, 256, 256]` input** and works on watermarked video at any source resolution.

Output `[1, 257]` raw logits:
- Index `0`: watermark-presence confidence bit (sigmoid → probability)
- Indices `1–256`: payload bits (sign → 0/1 → decode to ASCII)

```bash
# Export once — works for all resolutions
python3 videoseal/export_videoseal_detector_openvino.py
# → videoseal_detector.xml + videoseal_detector.bin
```

### Extraction — multi-frame averaging

H.264/H.265 compression in high-motion content (e.g. sports video) can flip individual watermark bits in a single frame.
The extraction script averages raw logit scores across up to 20 evenly sampled frames before thresholding.
The watermark signal is identical in every frame and reinforces; compression noise is random and cancels out.

```bash
python3 videoseal/extract_videoseal_ov.py \
    --model videoseal_detector.xml \
    --video test_out_watermarked.mp4
```

**Important**: resize source frames to 256×256 with `cv2.INTER_AREA` (area averaging) before feeding the detector, not `INTER_LINEAR`. `INTER_AREA` closely matches PyTorch's `antialias=True` bilinear downscale used in the original `model.detect()` implementation, reducing per-frame bit-sign mismatches from ~16 to ~2 at 720p, which multi-frame averaging then eliminates.

---

## Backend Architecture — model_table dispatcher

All model-specific behaviour in `dnn_backend_ivsr.c` is driven by a static descriptor table. There are no `if (model_type == X)` chains in any hot-path function after patch 0007.

### ModelDesc struct

Each row in `model_table[]` is a `ModelDesc` that fully describes one model:

| Field | Purpose |
|:--|:--|
| `name` | Human-readable label used in log messages |
| `nif_override` | Override SDK-reported frame count (0 = use SDK value) |
| `channel_divisor` | Divide the reported channel count so the filter graph sees standard 3-ch frames (RIFE ÷2, TSENet ÷3) |
| `align` | Round input W and H up to this alignment before reshape (0 = none) |
| `in_layout` / `out_layout` | Tensor memory layout string passed to iVSR SDK |
| `in_precision` / `out_precision` | Tensor element type (`f32`, `u8`, `u16`; NULL = inherit from bit-depth) |
| `model_color` | Colour space string for the iVSR SDK (NULL = use `color_format_auto`) |
| `out_order` | Output channel order for `ff_proc_from_dnn_to_frame` |
| `pack_input` | Function pointer — custom rgb→tensor packing; NULL = generic `ff_proc` path |
| `unpack_output` | Function pointer — custom tensor→rgb unpacking; NULL = generic path |
| `out_precision_depth_derived` | Derive output precision from frame bit-depth (EDSR style) |
| `output_passthrough_dims` | Report output W/H = input W/H (VideoProc/SVP passthrough) |
| `color_format_auto` | `0` = fixed; `1` = auto RGB vs I420 from pixel format (SVP); `2` = always I420 (CustVSR) |
| `sliding_window_init_dup` | Duplicate first frame to prime the sliding-window queue (TSENet) |
| `window_type` | Frame queuing strategy for JSON-loaded models: `WINDOW_SINGLE` / `WINDOW_SLIDING` / `WINDOW_IN_QUEUE` |
| `normalize_input` | `1` = divide uint8 by 255 before packing into float32 (RIFE/SPAN); `0` = raw [0,255] |
| `normalize_output` | `1` = multiply float32 by 255 before clipping to uint8; `0` = direct round-and-clip |

### Dispatch flow

```
ff_dnn_load_model_ivsr()
  └─ model_type == CUSTOM (-1)?
       └─ parse_model_config_json()  →  ivsr_model->dynamic_desc
  └─ get_model_desc()  ──►  dynamic_desc            (JSON-loaded model)
                       └──►  &model_table[type]      (built-in model)
  └─ apply: align, layout, precision, color_format, nif_override
            ↓
fill_model_input_ivsr()  [per input frame]
  ├─ pack_input != NULL  →  call custom function  (BasicVSR, TSENet, RIFE, VideoSeal)
  └─ pack_input == NULL  →  pack_input_window() for JSON models;
                             ff_proc_from_frame_to_dnn() for built-in generic path
            ↓
infer_completion_callback()  [per output frame]
  ├─ unpack_output != NULL  →  call custom function
  └─ unpack_output == NULL  →  unpack_output_generic_fp32() for JSON models;
                                ff_proc_from_dnn_to_frame() for built-in generic path
```

### Built-in model reference

| model_type | Enum | name | nif | align | in_layout | pack / unpack | Notable flags |
|:--|:--|:--|:--|:--|:--|:--|:--|
| 0 | BASICVSR | BasicVSR | SDK | 32 | NFHWC | custom both | — |
| 1 | VIDEOPROC | VideoProc | SDK | 64 | NHWC | generic both | `output_passthrough_dims`, `color_format_auto=1` |
| 2 | EDSR | EDSR | SDK | — | NHWC | generic both | `out_precision_depth_derived` |
| 3 | CUSTVSR | CustVSR | SDK | 64 | NHWC | generic both | `color_format_auto=2` |
| 4 | TSENET | TSENet | 3 | — | NCHW | custom input | `sliding_window_init_dup` |
| 5 | RIFE | RIFE | 2 | 128 | NCHW f32 | custom both | — |
| 6 | VIDEOSEAL | VideoSeal | 1 | — | NCHW f32 | custom both | — |
| **−1** | **CUSTOM** | from JSON | from JSON | from JSON | from JSON | `pack_input_window` / `unpack_output_generic_fp32` | all fields from JSON |

### Adding a new built-in model (Path B — C code)

1. Add an enum value before `MODEL_TYPE_NUM` in `ModelType`.
2. Add one row to `model_table[]` — the compile-time assert (`model_table_wrong_size`) will fail at build time if this is forgotten.
3. Optionally write `pack_input_<name>` / `unpack_output_<name>` if the generic path is insufficient; otherwise set the pointers to `NULL`.

For a full guide including annotated code skeletons, see `MODEL_ENABLEMENT_GUIDE.md`.

---

## JSON Model Config System

Patches 0007/0008 add a zero-C-change model registration path. Any model whose I/O follows a standard pattern (single-frame or sliding-window, float32 [0,1] or [0,255] NCHW/NHWC) can be enabled via a JSON file.

### Quick start

```bash
# 1. Export your model to OpenVINO IR (example: SPAN)
python3 span_to_openvino.py --scale 4 --input span_x4.pth

# 2. Write a config (or copy and edit the template)
cp ivsr_model_config.template.json models/mymodel.json
# Edit mymodel.json — IDE validates against ivsr_model_config.schema.json

# 3. Run
./ffmpeg -i input.mp4 \
  -vf "format=rgb24,dnn_processing=dnn_backend=ivsr:model=mymodel.xml:\
model_type=-1:model_config=models/mymodel.json" \
  output.mp4
```

### JSON field reference

| Field | Type | Required | Default | Description |
|:--|:--|:--|:--|:--|
| `name` | string | ✅ | — | Label used in log messages |
| `in_layout` | string | ✅ | — | `NCHW` \| `NHWC` \| `NFHWC` |
| `nif` | integer | — | 1 | Frames consumed per inference call |
| `align` | integer | — | 0 | W/H alignment in pixels (0 = none) |
| `in_precision` | string\|null | — | null | `f32` \| `u8` \| `u16` \| null (inherit from bit-depth) |
| `out_layout` | string | — | `NCHW` | `NCHW` \| `NHWC` \| `NFHWC` |
| `out_precision` | string\|null | — | null | `fp32` \| `u8` \| `u16` \| null |
| `model_color` | string\|null | — | null | `RGB` \| `I420_Three_Planes` \| null |
| `out_order` | string | — | `RGB` | `RGB` \| `BGR` \| `NONE` |
| `window_type` | string | — | `single` | `single` — no queue; `sliding` — N-frame sliding window; `in_queue` — read from task queue |
| `window_init_dup` | boolean | — | false | Duplicate first frame when priming a sliding window (TSENet behaviour) |
| `normalize_input` | boolean | — | true | `true` = divide uint8 by 255 → float32 [0,1] |
| `normalize_output` | boolean | — | true | `true` = multiply float32 by 255 → uint8 |
| `output_passthrough_dims` | boolean | — | false | Output W/H = input W/H (VideoProc/SVP style) |
| `out_precision_depth_derived` | boolean | — | false | Derive output precision from frame bit-depth (EDSR style) |
| `color_format_auto` | integer | — | 0 | 0 = use `model_color`; 1 = auto RGB/I420; 2 = always I420 |

Keys beginning with `_` are treated as comments and silently skipped by the parser. Use `_comment_*` keys for inline documentation exactly as shown in `ivsr_model_config.template.json`.

### Validate before running

```bash
jsonschema -i models/mymodel.json ivsr_model_config.schema.json
```

### Existing models expressed as JSON configs

| Model | `model_type` | Usable via JSON? | Key JSON fields |
|:--|:--|:--|:--|
| Enhanced EDSR | 2 | ✅ | `out_precision_depth_derived: true` |
| SVP / VideoProc | 1 | ✅ | `color_format_auto: 1`, `output_passthrough_dims: true` |
| TSENet | 4 | ✅ | `window_type: sliding`, `window_init_dup: true`, `nif: 3` |
| RIFE | 5 | ✅ | `window_type: sliding`, `normalize_input: true`, `nif: 2` |
| VideoSeal | 6 | ✅ | `window_type: single`, `normalize_input: false`, `normalize_output: false` |
| Enhanced BasicVSR | 0 | ⚠️ | Input side only — multi-frame `out_queue` output needs a follow-on addition |

For a worked example with SPAN, HDRTVNet++, and a multi-stage pipeline, see `MODEL_ENABLEMENT_GUIDE.md`.

### When the JSON config alone is NOT sufficient

The JSON config system covers the generic I/O contract well, but there are concrete situations where it cannot replace custom C code. Use [Path B (C code)](MODEL_ENABLEMENT_GUIDE.md) instead in any of the following cases:

| Situation | Why JSON config cannot handle it | What to do |
|:--|:--|:--|
| **Multi-frame output** (like Enhanced BasicVSR) | `unpack_output_generic_fp32` reads a single output frame. Models that write `nif` separate output frames into `task->out_queue` need the `unpack_output_basicvsr` loop, which iterates over the queue. | Write a custom `unpack_output_<name>` function and register it in `model_table[]`. |
| **Multi-input tensors** (e.g. a model with a separate conditioning input) | `ivsr_process_async` is a single-blob API. Two tensors cannot be passed via one descriptor. | Pre-fuse the second input into the OpenVINO graph at export time (e.g. bake bicubic conditioning into the graph with a `torch.nn.Module` wrapper). See `MODEL_ENABLEMENT_GUIDE.md §3` for the export pattern. |
| **Non-RGB colour spaces in output** (e.g. raw YUV planes, depth maps, HDR PQ) | `unpack_output_generic_fp32` always writes packed `rgb24`. Models whose output is Y-only, YUV planar, or a non-8-bit format need a custom unpack function to route data into the correct frame planes. | Write a custom `unpack_output_<name>` that writes directly to `task->out_frame->data[plane]`. |
| **Non-standard sliding window** (e.g. look-ahead buffering, asymmetric context) | `pack_input_window` supports `single`, `sliding` (symmetric N-frame), and `in_queue` (read from task queue). Any other access pattern — e.g. 2 past frames + 1 future frame requiring look-ahead — cannot be expressed in the window_type field. | Write a custom `pack_input_<name>` with its own queue management. |
| **Post-inference tensor fusion** (e.g. cascaded sub-networks sharing a single iVSR handle) | The JSON config describes a single model loaded once. A model that requires calling `ivsr_process_async` multiple times per frame (sequential sub-networks with state passing) cannot be expressed as a single config file. | Write a custom `pack_input` that manages the intermediate buffers, or chain multiple `dnn_processing` filters in the FFmpeg graph. |
| **Exotic precision I/O** (float16, int8 quantised output requiring dequantization) | The generic unpack path assumes `float32` output. `int8` or `fp16` outputs need explicit dequantization logic in the unpack function. | Write a custom `unpack_output_<name>` with the correct cast and scale. |
| **Spatially varying output size** (different W/H per batch item) | `output_passthrough_dims` is a boolean flag covering the W=W_in, H=H_in case. Any other dynamic output size mapping requires querying the output tensor at runtime. | Write a custom `get_output_<name>` or extend `get_output_ivsr()` for the specific case. |

**Decision rule:** start with `model_type=-1` and a JSON config. If the model runs but output is wrong (wrong colour, wrong shape, or silent hang), check this table. The most common failures in practice are multi-frame output (Enhanced BasicVSR), multi-input tensors (AGCM conditioning), and non-normalized output ranges (models trained on [0,255] but `normalize_output` was left as `true`).  Refer to `MODEL_ENABLEMENT_GUIDE.md §8` (Troubleshooting) for symptom-to-cause mapping.

