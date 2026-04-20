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
|model_type|type for models|0|0 for Enhanced BasicVSR, 1 for SVP models, 2 for Enhanced EDSR, 3 for one CUSTOM VSR, 4 for TSENet, 5 for RIFE (frame interpolation), 6 for VideoSeal (invisible watermarking)|
|normalize_factor|factor for normalization|1.0|255.0 for Enhanced EDSR, 1.0 for other models supported in current version|
|num_streams|number of execution streams for the throughput mode (now valid only for GPU devices).|1|use `benchmark_app` (a tool provided by OpenVINO Toolkit), to get the appropriate value for the best throughput|
|extension|extension lib file full path, required for loading Enhanced BasicVSR model|
|op_xml|custom op xml file full path, required for loading Enhanced BasicVSR model|
|nif|number of input frames in batch sent to the DNN backend|1|3 for Enhanced BasicVSR, 1 for other models supported in current version|
|nireq|number of request|0|use the default setting or set it to match the number of cpu cores|

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
- Command sample to run RIFE frame interpolation inference (doubles frame rate), the input pixel format supported by the model is `rgb24`. Export the model with `rife_to_openvino.py` first.
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

