"""
iVSR Demo — Streamlit UI
Demonstrates Video Super Resolution (VSR) and Smart Video Processing (SVP)
using the iVSR + FFmpeg pipeline.
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Constants & model metadata
# ─────────────────────────────────────────────────────────────────────────────

# ── BasicVSR extension defaults (relative to the iVSR repo root) ─────────────
# Change these if your build output lives elsewhere.
_REPO_ROOT = Path(__file__).parent
DEFAULT_EXTENSION_SO = str(
    _REPO_ROOT / "ivsr_ov/based_on_openvino_2022.3/openvino/bin/intel64/Release/libcustom_extension.so"
)
DEFAULT_OP_XML = str(
    _REPO_ROOT / "ivsr_ov/based_on_openvino_2022.3/openvino/flow_warp_cl_kernel/flow_warp.xml"
)

# Default RIFE model path (sibling rife_ov directory)
DEFAULT_RIFE_MODEL = str(_REPO_ROOT.parent / "rife_ov" / "rife.xml")

# Local patched FFmpeg build (has model_type=6 / VideoSeal compiled in)
DEFAULT_LOCAL_FFMPEG = str(
    _REPO_ROOT / "ivsr_ffmpeg_plugin" / "ffmpeg" / "ffmpeg"
)
DEFAULT_LOCAL_LIBAVFILTER = str(
    _REPO_ROOT / "ivsr_ffmpeg_plugin" / "ffmpeg" / "libavfilter"
)

# Mapping: display name → (model_type int, nif default, normalize_factor default,
#          needs_extension, pixel_format, description)
VSR_MODELS = {
    "Enhanced BasicVSR": {
        "model_type": 0,
        "nif": 3,
        "nif_fixed": True,          # user cannot change nif for BasicVSR
        "normalize_factor": 1.0,
        "normalize_fixed": True,
        "needs_extension": True,    # requires custom OpenCL .so + .xml
        "pixel_format": "rgb24",
        "description": (
            "Multi-frame VSR using 3 consecutive frames. "
            "Highest quality upscaling. Requires OpenVINO 2022.3 "
            "and custom flow_warp extension."
        ),
        "scale": "2×",
        "input_shape": "[1, 3, 3, H, W]",
    },
    "Enhanced EDSR": {
        "model_type": 2,
        "nif": 1,
        "nif_fixed": True,
        "normalize_factor": 255.0,
        "normalize_fixed": True,
        "needs_extension": False,
        "pixel_format": "rgb24",
        "description": (
            "Single-frame super resolution. ~79% less compute than BasicVSR. "
        ),
        "scale": "2×",
        "input_shape": "[1, 3, H, W]",
    },
    "TSENet": {
        "model_type": 4,
        "nif": 1,
        "nif_fixed": False,
        "normalize_factor": 1.0,
        "normalize_fixed": True,
        "needs_extension": False,
        "pixel_format": "rgb24",
        "description": (
            "Preview model. Multi-frame SR using temporal–spatial enhancement. "
        ),
        "scale": "2×",
        "input_shape": "[1, 9, H, W]",
    },
    "CustomVSR": {
        "model_type": 3,
        "nif": 1,
        "nif_fixed": False,
        "normalize_factor": 1.0,
        "normalize_fixed": False,
        "needs_extension": False,
        "pixel_format": "yuv420p",
        "description": (
            "Bring-your-own VSR model in OpenVINO IR format. "
            "Configure nif and normalize_factor to match your model."
        ),
        "scale": "varies",
        "input_shape": "model-defined",
    },
}

SVP_MODELS = {
    "SVP-Basic": {
        "model_type": 1,
        "nif": 1,
        "nif_fixed": True,
        "normalize_factor": 1.0,
        "normalize_fixed": True,
        "needs_extension": False,
        "pixel_format": "rgb24",
        "description": (
            "Pre-encoder AI filter. Reduces bitrate while preserving "
            "perceived visual quality. Supports RGB and YUV input."
        ),
        "scale": "1× (same resolution)",
    },
    "SVP-SE": {
        "model_type": 1,
        "nif": 1,
        "nif_fixed": True,
        "normalize_factor": 1.0,
        "normalize_fixed": True,
        "needs_extension": False,
        "pixel_format": "rgb24",
        "description": (
            "Squeeze-and-excitation variant for SVP. "
            "Up to 50% bitrate savings vs. SVP-Basic."
        ),
        "scale": "1× (same resolution)",
    },
}

DEVICE_OPTIONS = ["CPU", "GPU", "GPU.0", "GPU.1", "MULTI:GPU.0,GPU.1", "AUTO"]

FFMPEG_BIN_HINTS = [
    "/usr/local/bin/ffmpeg",
    "/usr/bin/ffmpeg",
    "ffmpeg",  # rely on PATH
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_ffmpeg() -> str:
    """Return the first usable ffmpeg binary path."""
    for path in FFMPEG_BIN_HINTS:
        result = subprocess.run(
            ["which", path] if path == "ffmpeg" else ["test", "-x", path],
            capture_output=True,
        )
        if result.returncode == 0:
            return path
    return "ffmpeg"


def build_ffmpeg_cmd(
    ffmpeg_bin: str,
    input_video: str,
    output_video: str,
    model_path: str,
    pixel_format: str,
    model_type: int,
    nif: int,
    nireq: int,
    num_streams: int,
    device: str,
    normalize_factor: float,
    extension: str = "",
    op_xml: str = "",
    batch_size: int = 1,
    extra_encode_flags: str = "",
) -> list[str]:
    """Build the FFmpeg command list for iVSR processing."""
    # Build the dnn_processing filter string
    dnn_filter = (
        f"dnn_processing=dnn_backend=ivsr"
        f":model={model_path}"
        f":input=input"
        f":output=output"
        f":nif={nif}"
        f":nireq={nireq}"
        f":num_streams={num_streams}"
        f":device={device}"
        f":model_type={model_type}"
        f":normalize_factor={normalize_factor}"
        f":batch_size={batch_size}"
    )
    if extension:
        dnn_filter += f":extension={extension}"
    if op_xml:
        dnn_filter += f":op_xml={op_xml}"

    vf = f"format={pixel_format},{dnn_filter}"

    cmd = [ffmpeg_bin, "-y", "-i", input_video, "-vf", vf]

    # Output pixel format / codec
    if pixel_format == "rgb24":
        cmd += ["-pix_fmt", "yuv420p"]
    else:
        cmd += ["-pix_fmt", "yuv420p"]

    if extra_encode_flags:
        cmd += extra_encode_flags.split()

    cmd.append(output_video)
    return cmd


def align_to_128(value: int) -> int:
    """Round up to the nearest multiple of 128 (required by RIFE model)."""
    return ((value + 127) // 128) * 128


def build_rife_ffmpeg_cmd(
    ffmpeg_bin: str,
    input_video: str,
    output_video: str,
    model_path: str,
    target_w: int,
    target_h: int,
    device: str,
    nireq: int,
    num_streams: int,
    mode: str,          # "2xfps" or "midpoints"
    src_fps: float = 25.0,
    extra_encode_flags: str = "",
    smooth_filter: str = "none",   # "none", "hqdn3d", or "deflicker"
) -> list[str]:
    """Build the FFmpeg command list for RIFE frame interpolation.

    2xfps mode:  dual-input trick — the input file is passed to FFmpeg twice.
                 [0:v] feeds the original-frames branch (fast, passthrough).
                 [1:v] feeds the DNN branch (slow, OpenVINO).
                 Two independent decoders mean interleave is never starved:
                 it pulls from the fast branch at its own pace and from the
                 DNN branch as slowly as inference requires — no split filter,
                 no fifo (removed in n6+), no queue overflow, no blank frames.
    Midpoints:   simple dnn_processing filter (outputs only interpolated frames).

    src_fps must be the actual source frame rate (e.g. 25.0, 29.97, 30.0).
    It is embedded as a literal in the setpts expression because dnn_processing
    does not propagate frame_rate to its output link (FRAME_RATE would be 0).
    """
    dnn_opts = (
        f"dnn_backend=ivsr"
        f":model={model_path}"
        f":input=input"
        f":output=output"
        f":nireq={nireq}"
        f":num_streams={num_streams}"
        f":device={device}"
        f":model_type=5"
        f":normalize_factor=1.0"
    )

    if mode == "2xfps":
        # Complex filtergraph: split → RIFE branch → shift timestamps → interleave
        # Must use -filter_complex (not -vf) for labeled pads; label final output [outv]
        #
        # PTS strategy (three-stage):
        #
        # Stage 1 — setpts=PTS-STARTPTS before split
        #   Normalises source PTS to start at 0.  Eliminates non-zero start PTS
        #   and any container offset that would corrupt the downstream math.
        #
        # Stage 2 — setpts=PTS+{half}/TB on the RIFE stream
        #   Shifts each interpolated frame forward by exactly half a frame period
        #   IN SECONDS (half = 1/(2*fps), e.g. 0.02s for 25fps).
        #   Dividing by TB converts seconds → timebase units correctly regardless
        #   of what timebase dnn_processing sets on its output link.
        #   This is safer than 1/(2*fps*TB) because TB in that expression can
        #   evaluate to 1 inside certain filter contexts, shifting by only
        #   1 clock tick (0.000078s) instead of the intended 0.02s — the
        #   exact bug that produced the (0.000000, 0.000078, 0.080000, …) pattern.
        #
        # Stage 3 — setpts=N/(out_fps*TB) + fps=out_fps AFTER interleave
        #   Re-indexes the merged stream so frame 0→0, frame 1→1/out_fps, …
        #   This ignores all previous PTS and produces a perfectly linear
        #   timeline.  fps= is the final conformer for the container header.
        out_fps_val = 2.0 * src_fps
        out_fps_str = str(int(out_fps_val)) if out_fps_val == int(out_fps_val) else f"{out_fps_val:.6g}"
        pad_w = align_to_128(target_w)
        pad_h = align_to_128(target_h)
        fc = (
            f"[0:v]scale={target_w}:{target_h},format=rgb24,"
            f"pad={pad_w}:{pad_h}:0:0,crop={target_w}:{target_h},"
            f"setpts=2*N[orig];"
            f"[1:v]scale={target_w}:{target_h},format=rgb24,"
            f"pad={pad_w}:{pad_h}:0:0,"
            f"dnn_processing={dnn_opts},crop={target_w}:{target_h},"
            f"setpts=2*N+1[dnn_out];"
            f"[orig][dnn_out]interleave=nb_inputs=2:duration=longest,"
            f"setpts=N/({out_fps_str}*TB),"
            f"fps={out_fps_str}[outv]"
        )
        cmd = [ffmpeg_bin, "-y", "-i", input_video, "-i", input_video,
               "-filter_complex", fc]
        cmd += ["-map", "[outv]", "-map", "0:a?", "-c:a", "copy"]
    else:
        # dnn_processing does not propagate frame_rate (FRAME_RATE == 0 on its
        # output link), so PTS from the filter are irregular.  Reset them to a
        # strictly uniform sequence: frame N → N/fps seconds.  Without this the
        # encoder receives uneven timestamps and the browser player jitters.
        # Optional post-smooth filter:
        #   hqdn3d=0:0:4:3  — temporal-only denoise; each pixel is averaged
        #                     across time, smoothing the spatial wobble/wave
        #                     caused by independent RIFE optical-flow errors.
        #   deflicker       — normalises per-frame luminance; only helps with
        #                     brightness flicker, not spatial ripple.
        if smooth_filter == "hqdn3d":
            _smooth = ",hqdn3d=0:0:4:3"
        elif smooth_filter == "deflicker":
            _smooth = ",deflicker"
        else:
            _smooth = ""
        pad_w = align_to_128(target_w)
        pad_h = align_to_128(target_h)
        fc = (
            f"scale={target_w}:{target_h},"
            f"format=rgb24,"
            f"pad={pad_w}:{pad_h}:0:0,"
            f"dnn_processing={dnn_opts},"
            f"crop={target_w}:{target_h},"
            f"setpts=N/({src_fps}*TB),"
            f"fps={src_fps}"
            f"{_smooth}[outv]"
        )
        cmd = [ffmpeg_bin, "-y", "-i", input_video, "-filter_complex", fc]
        cmd += ["-map", "[outv]", "-map", "0:a?", "-c:a", "copy"]

    cmd += ["-pix_fmt", "yuv420p"]
    # Disable B-frames so DTS == PTS for every frame.
    # B-frame reordering causes HTML5 players (Streamlit st.video) to decode
    # frames out of display order, producing visible "shaking" / jitter.
    cmd += ["-bf", "0"]
    if extra_encode_flags:
        cmd += extra_encode_flags.split()
    cmd.append(output_video)
    return cmd


def cmd_to_display_string(cmd: list[str]) -> str:
    """Pretty-print a command list as a shell string."""
    parts = []
    for part in cmd:
        if " " in part or ":" in part:
            parts.append(f'"{part}"')
        else:
            parts.append(part)
    return " \\\n  ".join(parts)


def _parse_ffmpeg_progress(line: str) -> tuple[int, float]:
    """Extract (frame, fps) from an FFmpeg progress line."""
    import re
    frame = 0
    fps = 0.0
    m = re.search(r"frame=\s*(\d+)", line)
    if m:
        frame = int(m.group(1))
    m = re.search(r"fps=\s*([\d.]+)", line)
    if m:
        fps = float(m.group(1))
    return frame, fps


def run_ffmpeg(cmd: list[str], status_text, total_frames: int = 0,
               extra_env: dict | None = None) -> tuple[int, str, float]:
    """Run FFmpeg, show progress, return (returncode, stderr, elapsed).

    extra_env: optional dict of additional environment variables (e.g.
               {"LD_LIBRARY_PATH": "/path/to/libavfilter"}).  Values are
               merged on top of the current process environment so all
               standard paths remain intact.
    """
    import os as _os
    start = time.time()
    stderr_lines: list[str] = []

    run_env = None
    if extra_env:
        run_env = _os.environ.copy()
        run_env.update(extra_env)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=run_env,
    )
    for line in proc.stderr:  # type: ignore[union-attr]
        line = line.rstrip()
        stderr_lines.append(line)
        # Skip uninitialised progress lines (time=N/A means no frames processed yet)
        if "time=N/A" in line:
            continue
        if "frame=" in line:
            frame, fps = _parse_ffmpeg_progress(line)
            if total_frames > 0 and frame > 0:
                pct = min(frame / total_frames, 1.0)
                label = f"Processing frame {frame} / {total_frames} ({pct * 100:.1f}%)"
                if fps > 0:
                    label += f" | {fps:.1f} fps"
                status_text.progress(pct, text=label)
            elif frame > 0:
                label = f"Processing frame {frame}"
                if fps > 0:
                    label += f" | {fps:.1f} fps"
                status_text.text(label)
        elif line and "frame=" not in line and "time=" not in line:
            status_text.text(line)
    proc.wait()
    elapsed = time.time() - start
    return proc.returncode, "\n".join(stderr_lines), elapsed


def stat(label: str, value: str, delta: str = "") -> str:
    """Return compact HTML for a small labeled stat tile."""
    delta_html = (
        f'<span style="color:#21c354;font-size:0.72rem"> {delta}</span>'
        if delta else ""
    )
    return (
        f'<div style="margin:2px 0">'
        f'<span style="font-size:0.72rem;color:#888">{label}&nbsp;</span>'
        f'<span style="font-size:0.82rem;font-weight:600">{value}</span>'
        f'{delta_html}</div>'
    )


def stats_block(*items) -> None:
    """Render multiple stat() items as a single markdown block."""
    st.markdown("".join(items), unsafe_allow_html=True)


def probe_video(path: str, ffmpeg_bin: str) -> dict:
    """Return basic video stats via ffprobe: width, height, duration, nb_frames, size."""
    ffprobe = os.path.join(os.path.dirname(ffmpeg_bin), "ffprobe")
    if not os.path.isfile(ffprobe):
        ffprobe = "ffprobe"
    try:
        # Stream-level info
        r_stream = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,nb_frames,duration,r_frame_rate",
                "-of", "default=noprint_wrappers=1",
                path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        # Format-level info (bitrate, duration fallback)
        r_format = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-show_entries", "format=bit_rate,duration,size",
                "-of", "default=noprint_wrappers=1",
                path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        info: dict = {}
        for line in r_stream.stdout.splitlines() + r_format.stdout.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                info[k.strip()] = v.strip()
        width = int(info.get("width", 0))
        height = int(info.get("height", 0))
        duration = float(info.get("duration", 0) or 0)
        nb_frames = info.get("nb_frames", "N/A")
        rfr = info.get("r_frame_rate", "0/1")
        num, den = rfr.split("/") if "/" in rfr else (rfr, "1")
        fps = round(int(num) / int(den), 2) if int(den) else 0
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        bitrate_bps = int(info.get("bit_rate", 0) or 0)
        bitrate_kbps = round(bitrate_bps / 1000) if bitrate_bps else 0
        return {"width": width, "height": height, "duration": duration,
                "nb_frames": nb_frames, "fps": fps, "size": size,
                "bitrate_kbps": bitrate_kbps}
    except Exception:
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        return {"width": 0, "height": 0, "duration": 0,
                "nb_frames": "N/A", "fps": 0, "size": size, "bitrate_kbps": 0}


def save_upload(uploaded_file, suffix: str, key: str) -> str | None:
    """Save an UploadedFile to a temp path; cache path in session_state."""
    if uploaded_file is None:
        return None
    cache_key = f"_tmp_path_{key}"
    name_key = f"_tmp_name_{key}"
    if (
        cache_key in st.session_state
        and st.session_state.get(name_key) == uploaded_file.name
    ):
        return st.session_state[cache_key]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    st.session_state[cache_key] = tmp.name
    st.session_state[name_key] = uploaded_file.name
    return tmp.name


def apply_faststart(src: str, ffmpeg_bin: str) -> str:
    """Remux src in-place with moov-at-front for smooth browser playback."""
    tmp = src + ".faststart.mp4"
    result = subprocess.run(
        [ffmpeg_bin, "-y", "-i", src, "-c", "copy", "-movflags", "+faststart", tmp],
        capture_output=True,
    )
    if result.returncode == 0:
        os.replace(tmp, src)
    elif os.path.exists(tmp):
        os.remove(tmp)
    return src


def make_preview(src_path: str, ffmpeg_bin: str) -> str:
    """Remux a video with moov-at-front (faststart) for smooth browser playback."""
    cache_key = f"_preview_{src_path}"
    if cache_key in st.session_state:
        cached = st.session_state[cache_key]
        if os.path.isfile(cached):
            return cached
    out = src_path + "_preview.mp4"
    result = subprocess.run(
        [ffmpeg_bin, "-y", "-i", src_path, "-c", "copy",
         "-movflags", "+faststart", out],
        capture_output=True,
    )
    path = out if result.returncode == 0 else src_path
    st.session_state[cache_key] = path
    return path


def save_model_uploads(xml_file, bin_file, key: str) -> str | None:
    """Save .xml and .bin to a shared temp dir; return the .xml path."""
    if xml_file is None or bin_file is None:
        return None
    cache_key = f"_tmp_model_{key}"
    names_key = f"_tmp_model_{key}_names"
    names = (xml_file.name, bin_file.name)
    if cache_key in st.session_state and st.session_state.get(names_key) == names:
        return st.session_state[cache_key]
    tmp_dir = tempfile.mkdtemp()
    stem = Path(xml_file.name).stem
    xml_path = os.path.join(tmp_dir, stem + ".xml")
    bin_path = os.path.join(tmp_dir, stem + ".bin")
    with open(xml_path, "wb") as f:
        f.write(xml_file.getbuffer())
    with open(bin_path, "wb") as f:
        f.write(bin_file.getbuffer())
    st.session_state[cache_key] = xml_path
    st.session_state[names_key] = names
    return xml_path


def build_videoseal_ffmpeg_cmd(
    ffmpeg_bin: str,
    input_video: str,
    output_video: str,
    model_path: str,
    target_w: int,
    target_h: int,
    device: str,
    nireq: int,
    num_streams: int,
    crf: int = 18,
    codec: str = "libx264",
    lossless: bool = False,
) -> list[str]:
    """Build the FFmpeg command for VideoSeal invisible watermark embedding.

    model_type=6  — VIDEOSEAL enum value in dnn_backend_ivsr.c
    normalize_factor=1.0 — the model has /255 and *255 baked in
    Input must be rgb24 at exactly the resolution the model was exported for.

    lossless=True  → FFV1 level 3, pix_fmt rgb24 (bit-exact; no chroma subsampling)
    lossless=False → lossy codec with CRF and pix_fmt yuv420p
    """
    dnn_filter = (
        f"dnn_processing=dnn_backend=ivsr"
        f":model={model_path}"
        f":input=input"
        f":output=output"
        f":model_type=6"
        f":nif=1"
        f":nireq={nireq}"
        f":num_streams={num_streams}"
        f":device={device}"
        f":normalize_factor=1.0"
    )
    vf = f"scale={target_w}:{target_h},format=rgb24,{dnn_filter}"
    if lossless:
        # FFV1 with rgb24 preserves every watermarked pixel exactly.
        # yuv420p would destroy chroma precision even with a lossless codec.
        encode_args = ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "rgb24"]
    else:
        encode_args = ["-c:v", codec, "-crf", str(crf), "-pix_fmt", "yuv420p"]
    cmd = [
        ffmpeg_bin, "-y", "-i", input_video,
        "-vf", vf,
        *encode_args,
        "-c:a", "copy",
        output_video,
    ]
    return cmd


def extract_videoseal_watermark(video_path: str) -> tuple[str, float]:
    """Extract the invisible watermark from a VideoSeal-watermarked video.

    Uses the official VideoSeal ``detect()`` API (same approach as
    videoseal/extract_videoseal.py) which bypasses training-time augmentation.

    Returns:
        (extracted_text, confidence_pct)  where confidence_pct is in [0, 100].
        confidence_pct is 0.0 for older models that don't output a presence bit.
    """
    import cv2
    import torch
    import videoseal as _vs

    model = _vs.load("videoseal")
    model.eval()

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        raise RuntimeError(f"Could not read any frame from: {video_path}")

    # Sample up to 20 frames evenly and average raw logit scores.
    # Single-frame extraction is fragile: H.264/H.265 compression artifacts in
    # high-motion content (e.g. cricket) can flip individual payload bits.
    # Averaging logits across many frames lets the watermark signal dominate.
    max_frames = 20
    step = max(1, total // max_frames)
    scores = None
    confidence_sum = 0.0
    sampled = 0
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame_bgr = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(frame_rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        with torch.no_grad():
            detected = model.detect(tensor)
        preds = detected["preds"]
        if preds.shape[-1] > 256:
            confidence_sum += float(torch.sigmoid(preds[0, 0]).item())
            payload_logits = preds[0, 1:]
        else:
            payload_logits = preds[0]
        scores = payload_logits if scores is None else scores + payload_logits
        sampled += 1
    cap.release()

    if scores is None:
        raise RuntimeError(f"No frames could be decoded from: {video_path}")

    confidence = (confidence_sum / sampled * 100.0) if sampled > 0 else 0.0
    payload = scores

    bits = (payload > 0).int().flatten().tolist()[:256]
    chars: list[str] = []
    for i in range(0, len(bits) - 7, 8):
        byte = bits[i : i + 8]
        code = int("".join(map(str, byte)), 2)
        if code == 0:
            break
        chars.append(chr(code))

    return "".join(chars), confidence


def extract_videoseal_watermark_ov(video_path: str, detector_xml: str) -> tuple[str, float]:
    """Extract the invisible watermark using an OpenVINO IR detector model.

    The detector IR is produced by
    ``videoseal/export_videoseal_detector_openvino.py``.

    Input to the model: ``[1, 3, 256, 256]`` NCHW float32 in ``[0, 1]``.
    Output: ``[1, 257]`` raw logits — index 0 = presence bit, 1–256 = payload.

    Frames are sampled evenly (up to 20) and raw logits are averaged before
    thresholding, making extraction robust to H.264/H.265 compression noise.

    Returns:
        (extracted_text, confidence_pct)  where confidence_pct is in [0, 100].
    """
    import cv2
    import numpy as np
    import openvino as ov

    core = ov.Core()
    compiled = core.compile_model(detector_xml, "CPU")
    infer_req = compiled.create_infer_request()

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        raise RuntimeError(f"Could not read any frame from: {video_path}")

    max_frames = 20
    step = max(1, total // max_frames)
    scores: "np.ndarray | None" = None
    confidence_sum = 0.0
    sampled = 0

    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame_bgr = cap.read()
        if not ret:
            continue

        # BGR → RGB, resize to 256×256 (matches model.detect() internals), NCHW [0,1]
        # Use INTER_AREA for downscaling: it correctly averages source pixels into
        # the smaller destination, closely matching PyTorch's antialias=True bilinear.
        # INTER_LINEAR without antialias causes ~16 bit-sign mismatches per frame at HD.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_256 = cv2.resize(frame_rgb, (256, 256), interpolation=cv2.INTER_AREA)
        inp = frame_256.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0

        result = infer_req.infer({0: inp})
        preds = list(result.values())[0]  # [1, 257]

        if preds.shape[-1] > 256:
            # sigmoid of index 0 = presence confidence
            confidence_sum += float(1.0 / (1.0 + np.exp(-float(preds[0, 0]))))
            payload_logits = preds[0, 1:]
        else:
            payload_logits = preds[0]

        scores = payload_logits if scores is None else scores + payload_logits
        sampled += 1

    cap.release()

    if scores is None:
        raise RuntimeError(f"No frames could be decoded from: {video_path}")

    confidence = (confidence_sum / sampled * 100.0) if sampled > 0 else 0.0

    bits = (scores > 0).astype(int).flatten().tolist()[:256]
    chars: list[str] = []
    for i in range(0, len(bits) - 7, 8):
        byte = bits[i : i + 8]
        code = int("".join(map(str, byte)), 2)
        if code == 0:
            break
        chars.append(chr(code))

    return "".join(chars), confidence


# ─────────────────────────────────────────────────────────────────────────────
# Parameter UI builders (shared between tabs)
# ─────────────────────────────────────────────────────────────────────────────

def render_common_params(model_meta: dict, key_prefix: str) -> dict:
    """Render device, nif, nireq, num_streams, normalize params. Returns dict of values."""
    params = {}

    col1, col2 = st.columns(2)
    with col1:
        params["device"] = st.selectbox(
            "Target Device",
            DEVICE_OPTIONS,
            index=0,
            key=f"{key_prefix}_device",
            help="Hardware to run inference on.",
        )
    with col2:
        params["nireq"] = st.number_input(
            "Inference Requests (nireq)",
            min_value=1,
            max_value=32,
            value=1,
            step=1,
            key=f"{key_prefix}_nireq",
            help="Number of parallel OpenVINO inference requests.",
        )

    col3, col4 = st.columns(2)
    with col3:
        if model_meta["nif_fixed"]:
            params["nif"] = model_meta["nif"]
            st.caption(f"**Input Frames (nif):** {model_meta['nif']} *(fixed by model architecture)*")
        else:
            params["nif"] = st.number_input(
                "Input Frames (nif)",
                min_value=1,
                max_value=9,
                value=model_meta["nif"],
                step=1,
                key=f"{key_prefix}_nif",
                help="Number of consecutive input frames stacked per inference.",
            )

    with col4:
        gpu_only = params["device"].startswith("GPU") or params["device"] in ("MULTI:GPU.0,GPU.1", "AUTO")
        params["num_streams"] = st.number_input(
            "GPU Streams (num_streams)",
            min_value=1,
            max_value=16,
            value=1,
            step=1,
            disabled=not gpu_only,
            key=f"{key_prefix}_num_streams",
            help="Throughput streams for GPU only.",
        )

    col5, col6 = st.columns(2)
    with col5:
        if model_meta["normalize_fixed"]:
            params["normalize_factor"] = model_meta["normalize_factor"]
            st.caption(f"**Normalize Factor:** {model_meta['normalize_factor']} *(fixed by model type)*")
        else:
            params["normalize_factor"] = st.number_input(
                "Normalize Factor",
                min_value=1.0,
                max_value=65535.0,
                value=model_meta["normalize_factor"],
                step=1.0,
                key=f"{key_prefix}_nf",
                help="Divide input pixel values by this constant before inference.",
            )

    with col6:
        if model_meta["model_type"] == 0:  # BasicVSR — batch size not supported
            params["batch_size"] = 1
            st.caption("**Batch Size:** 1 *(not supported for BasicVSR)*")
        else:
            params["batch_size"] = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=32,
                value=1,
                step=1,
                key=f"{key_prefix}_batch",
                help="Number of frames processed per inference call.",
            )

    return params


def render_extension_params(key_prefix: str) -> dict:
    """Return the default BasicVSR extension paths (configured via DEFAULT_EXTENSION_SO / DEFAULT_OP_XML)."""
    return {"extension": DEFAULT_EXTENSION_SO, "op_xml": DEFAULT_OP_XML}


# ─────────────────────────────────────────────────────────────────────────────
# Page layout
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="iVSR Demo",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Run VSR / Run SVP — primary buttons */
    .stButton button {
        background-color: #1a6edb !important;
        border: 1px solid #1a6edb !important;
        color: #ffffff !important;
    }
    .stButton button:hover {
        background-color: #155bb5 !important;
        border-color: #155bb5 !important;
        color: #ffffff !important;
    }
    /* Download buttons */
    .stDownloadButton button {
        background-color: #1a6edb !important;
        border: 1px solid #1a6edb !important;
        color: #ffffff !important;
    }
    .stDownloadButton button:hover {
        background-color: #155bb5 !important;
        border-color: #155bb5 !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("iVSR — Intel Video Processing Demo")
st.markdown(
    "Demonstrate **Video Super Resolution** (upscaling) and "
    "**Smart Video Processing** (pre-encode bitrate reduction) "
    "powered by Intel OpenVINO and hardware acceleration."
)

# ── Sidebar: global settings ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Global Settings")

    ffmpeg_bin = st.text_input(
        "FFmpeg binary path",
        value=find_ffmpeg(),
        help="Path to the iVSR-patched FFmpeg binary.",
    )

    st.markdown("---")
    st.caption(
        "**Models must be in OpenVINO IR format** (.xml + .bin). "
        "They are not included in the iVSR repo."
    )
    st.caption("Source: [github.com/OpenVisualCloud/iVSR](https://github.com/OpenVisualCloud/iVSR)")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_vsr, tab_svp, tab_rife, tab_vs = st.tabs([
    "Video Super Resolution (VSR)",
    "Smart Video Processing (SVP)",
    "Frame Interpolation (RIFE)",
    "Invisible Watermarking (VideoSeal)",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Video Super Resolution
# ═════════════════════════════════════════════════════════════════════════════
with tab_vsr:
    st.subheader("Video Super Resolution")
    st.markdown(
        "Upscale video resolution (e.g., 1080p → 4K) using deep-learning AI models "
        "running on Intel CPU or GPU hardware."
    )

    # ── Model selection ───────────────────────────────────────────────────────
    col_model, col_info = st.columns([1, 2])
    with col_model:
        vsr_model_name = st.selectbox(
            "Select VSR Model",
            list(VSR_MODELS.keys()),
            index=list(VSR_MODELS.keys()).index("Enhanced EDSR"),
            key="vsr_model_select",
        )
    vsr_meta = VSR_MODELS[vsr_model_name]
    with col_info:
        st.info(
            f"**{vsr_model_name}**  \n"
            f"{vsr_meta['description']}  \n"
            f"Scale: `{vsr_meta['scale']}` | "
            f"Input shape: `{vsr_meta.get('input_shape', 'model-defined')}` | "
            f"model_type: `{vsr_meta['model_type']}`"
        )

    st.markdown("---")

    # ── Model file upload & input video ──────────────────────────────────────
    col_xl, col_bin = st.columns(2)
    with col_xl:
        vsr_xml_file = st.file_uploader(
            "Upload Model (.xml)",
            type=["xml"],
            key="vsr_xml_upload",
            help="OpenVINO IR model XML file.",
        )
    with col_bin:
        vsr_bin_file = st.file_uploader(
            "Upload Model Weights (.bin)",
            type=["bin"],
            key="vsr_bin_upload",
            help="OpenVINO IR binary weights. Must match the .xml.",
        )
    vsr_model_path = save_model_uploads(vsr_xml_file, vsr_bin_file, "vsr")

    vsr_video_file = st.file_uploader(
        "Upload Input Video",
        type=["mp4", "mkv", "avi", "mov", "webm", "ts"],
        key="vsr_video_upload",
    )
    vsr_input_video = save_upload(vsr_video_file, ".mp4", "vsr_video")
    vsr_output_video = os.path.join(tempfile.gettempdir(), "ivsr_vsr_output.mp4")

    st.markdown("---")
    st.markdown("#### Parameters")

    # ── Common params ─────────────────────────────────────────────────────────
    vsr_params = render_common_params(vsr_meta, "vsr")

    # ── BasicVSR-specific extension fields ────────────────────────────────────
    vsr_ext_params: dict = {}
    if vsr_meta["needs_extension"]:
        st.markdown("---")
        vsr_ext_params = render_extension_params("vsr")

    # ── Generated command preview ─────────────────────────────────────────────
    if vsr_model_path and vsr_input_video:
        st.markdown("---")
        _preview_cmd = build_ffmpeg_cmd(
            ffmpeg_bin=ffmpeg_bin,
            input_video=vsr_input_video,
            output_video=vsr_output_video,
            model_path=vsr_model_path,
            pixel_format=vsr_meta["pixel_format"],
            model_type=vsr_meta["model_type"],
            nif=int(vsr_params["nif"]),
            nireq=int(vsr_params["nireq"]),
            num_streams=int(vsr_params["num_streams"]),
            device=vsr_params["device"],
            normalize_factor=float(vsr_params["normalize_factor"]),
            extension=vsr_ext_params.get("extension", ""),
            op_xml=vsr_ext_params.get("op_xml", ""),
            batch_size=int(vsr_params["batch_size"]),
        )
        with st.expander("Generated FFmpeg Command", expanded=False):
            st.code(cmd_to_display_string(_preview_cmd), language="bash")

    # ── Run button ────────────────────────────────────────────────────────────
    vsr_run = st.button("Run VSR")
    if vsr_run:
        errors = []
        if not vsr_model_path:
            errors.append("Upload both the model .xml and .bin files.")
        if not vsr_input_video:
            errors.append("Upload an input video file.")
        if vsr_meta["needs_extension"]:
            ext_path = vsr_ext_params.get("extension", "")
            opxml_path = vsr_ext_params.get("op_xml", "")
            if not ext_path:
                errors.append("Extension .so path is required for BasicVSR.")
            elif not os.path.isfile(ext_path):
                errors.append(f"Extension .so not found: {ext_path}")
            if not opxml_path:
                errors.append("Op XML path is required for BasicVSR.")
            elif not os.path.isfile(opxml_path):
                errors.append(f"Op XML not found: {opxml_path}")

        if errors:
            for e in errors:
                st.error(e)
        else:
            cmd = build_ffmpeg_cmd(
                ffmpeg_bin=ffmpeg_bin,
                input_video=vsr_input_video,
                output_video=vsr_output_video,
                model_path=vsr_model_path,
                pixel_format=vsr_meta["pixel_format"],
                model_type=vsr_meta["model_type"],
                nif=int(vsr_params["nif"]),
                nireq=int(vsr_params["nireq"]),
                num_streams=int(vsr_params["num_streams"]),
                device=vsr_params["device"],
                normalize_factor=float(vsr_params["normalize_factor"]),
                extension=vsr_ext_params.get("extension", ""),
                op_xml=vsr_ext_params.get("op_xml", ""),
                batch_size=int(vsr_params["batch_size"]),
            )
            _vsr_probe = probe_video(vsr_input_video, ffmpeg_bin)
            _vsr_total = int(_vsr_probe["nb_frames"]) if str(_vsr_probe["nb_frames"]).isdigit() else 0
            status = st.empty()
            with st.spinner("Running iVSR VSR inference..."):
                rc, stderr, elapsed = run_ffmpeg(cmd, status, total_frames=_vsr_total)
            status.empty()

            if rc == 0:
                with st.spinner("Optimising output for playback..."):
                    apply_faststart(vsr_output_video, ffmpeg_bin)
                in_info = probe_video(vsr_input_video, ffmpeg_bin)
                out_info = probe_video(vsr_output_video, ffmpeg_bin)
                throughput = (
                    round(int(out_info["nb_frames"]) / elapsed, 1)
                    if out_info["nb_frames"] != "N/A" else None
                )
                st.session_state["vsr_result"] = {
                    "output": vsr_output_video,
                    "input": vsr_input_video,
                    "elapsed": elapsed,
                    "in_info": in_info,
                    "out_info": out_info,
                    "throughput": throughput,
                }
            else:
                st.session_state.pop("vsr_result", None)
                st.error(f"FFmpeg failed (exit code {rc})")
                with st.expander("FFmpeg stderr"):
                    st.code(stderr, language="text")

    # ── Results (persists across reruns via session_state) ────────────────────
    if "vsr_result" in st.session_state:
        res = st.session_state["vsr_result"]
        ii = res.get("in_info", {})
        oi = res.get("out_info", {})
        st.success(f"Done in **{res['elapsed']:.1f}s** | Throughput: {res['throughput']} fps" if res.get('throughput') else f"Done in **{res['elapsed']:.1f}s**")

        # ── Before / After comparison ─────────────────────────────────────────
        res_str = lambda w, h: f"{w}×{h}" if w else "N/A"
        size_str = lambda s: f"{s / 1024 / 1024:.1f} MB" if s else "N/A"
        br_str = lambda k: f"{k:,} kbps" if k else "N/A"
        dur_str = lambda d: f"{int(d // 60)}m {d % 60:.1f}s" if d else "N/A"
        frames_str = lambda n: str(n) if n != "N/A" else "N/A"

        col_in, col_out = st.columns(2)
        with col_in:
            st.markdown("**Input Video**")
            st.video(make_preview(res["input"], ffmpeg_bin))
            stats_block(
                stat("Resolution", res_str(ii.get('width'), ii.get('height'))),
                stat("Bitrate",    br_str(ii.get('bitrate_kbps', 0))),
                stat("File Size",  size_str(ii.get('size', 0))),
                stat("Duration",   dur_str(ii.get('duration', 0))),
                stat("Frames",     frames_str(ii.get('nb_frames', 'N/A'))),
            )
        with col_out:
            st.markdown("**Output Video (Super Resolved)**")
            if os.path.isfile(res["output"]):
                st.video(res["output"])
                stats_block(
                    stat("Resolution", res_str(oi.get('width'), oi.get('height'))),
                    stat("Bitrate",    br_str(oi.get('bitrate_kbps', 0))),
                    stat("File Size",  size_str(oi.get('size', 0))),
                    stat("Duration",   dur_str(oi.get('duration', 0))),
                    stat("Frames",     frames_str(oi.get('nb_frames', 'N/A'))),
                )
                st.download_button(
                    "Download Output",
                    data=open(res["output"], "rb").read(),
                    file_name="vsr_output.mp4",
                    mime="video/mp4",
                    key="vsr_download",
                )
            else:
                st.warning("Output file not found.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Smart Video Processing
# ═════════════════════════════════════════════════════════════════════════════
with tab_svp:
    st.subheader("Smart Video Processing (SVP)")
    st.markdown(
        "A pre-encoder AI filter that reduces bitrate while preserving perceived visual quality. "
        "Output resolution is the **same** as the input — the AI enhances encoding efficiency."
    )

    # ── Model selection ───────────────────────────────────────────────────────
    col_model, col_info = st.columns([1, 2])
    with col_model:
        svp_model_name = st.selectbox(
            "Select SVP Model",
            list(SVP_MODELS.keys()),
            key="svp_model_select",
        )
    svp_meta = SVP_MODELS[svp_model_name]
    with col_info:
        st.info(
            f"**{svp_model_name}**  \n"
            f"{svp_meta['description']}  \n"
            f"Scale: `{svp_meta['scale']}` | "
            f"model_type: `{svp_meta['model_type']}`"
        )

    st.markdown("---")

    # ── Pixel format selection (SVP supports both RGB and YUV) ───────────────
    svp_pixel_format = st.selectbox(
        "Input Pixel Format",
        ["rgb24", "yuv420p"],
        key="svp_pf",
        help=(
            "rgb24: model processes RGB. "
            "yuv420p: model processes luma+chroma (requires Y-input SVP model)."
        ),
    )

    col_xl, col_bin = st.columns(2)
    with col_xl:
        svp_xml_file = st.file_uploader(
            "Upload Model (.xml)",
            type=["xml"],
            key="svp_xml_upload",
            help="OpenVINO IR model XML file.",
        )
    with col_bin:
        svp_bin_file = st.file_uploader(
            "Upload Model Weights (.bin)",
            type=["bin"],
            key="svp_bin_upload",
            help="OpenVINO IR binary weights. Must match the .xml.",
        )
    svp_model_path = save_model_uploads(svp_xml_file, svp_bin_file, "svp")

    svp_video_file = st.file_uploader(
        "Upload Input Video",
        type=["mp4", "mkv", "avi", "mov", "webm", "ts"],
        key="svp_video_upload",
    )
    svp_input_video = save_upload(svp_video_file, ".mp4", "svp_video")
    svp_output_video = os.path.join(tempfile.gettempdir(), "ivsr_svp_output.mp4")

    # ── Encoding options ──────────────────────────────────────────────────────
    with st.expander("Encoding Options"):
        col_enc1, col_enc2 = st.columns(2)
        with col_enc1:
            svp_codec = st.selectbox(
                "Output Codec",
                ["libx265", "libx264", "copy"],
                key="svp_codec",
                help="Encoder to use after SVP filtering.",
            )
        with col_enc2:
            svp_crf = st.slider(
                "CRF (quality, lower=better)",
                min_value=0,
                max_value=51,
                value=28,
                key="svp_crf",
                help="Constant Rate Factor for lossy encoders.",
            )
        svp_extra_flags = f"-c:v {svp_codec} -crf {svp_crf}" if svp_codec != "copy" else "-c:v copy"

    st.markdown("---")
    st.markdown("#### Parameters")

    svp_params = render_common_params(svp_meta, "svp")

    # ── Generated command preview ─────────────────────────────────────────────
    if svp_model_path and svp_input_video:
        st.markdown("---")
        cmd = build_ffmpeg_cmd(
            ffmpeg_bin=ffmpeg_bin,
            input_video=svp_input_video,
            output_video=svp_output_video,
            model_path=svp_model_path,
            pixel_format=svp_pixel_format,
            model_type=svp_meta["model_type"],
            nif=int(svp_params["nif"]),
            nireq=int(svp_params["nireq"]),
            num_streams=int(svp_params["num_streams"]),
            device=svp_params["device"],
            normalize_factor=float(svp_params["normalize_factor"]),
            batch_size=int(svp_params["batch_size"]),
            extra_encode_flags=svp_extra_flags,
        )
        with st.expander("Generated FFmpeg Command", expanded=False):
            st.code(cmd_to_display_string(cmd), language="bash")

    # ── Run button ────────────────────────────────────────────────────────────
    st.markdown("---")
    svp_run = st.button("Run SVP")
    if svp_run:
        errors = []
        if not svp_model_path:
            errors.append("Upload both the model .xml and .bin files.")
        if not svp_input_video:
            errors.append("Upload an input video file.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            cmd = build_ffmpeg_cmd(
                ffmpeg_bin=ffmpeg_bin,
                input_video=svp_input_video,
                output_video=svp_output_video,
                model_path=svp_model_path,
                pixel_format=svp_pixel_format,
                model_type=svp_meta["model_type"],
                nif=int(svp_params["nif"]),
                nireq=int(svp_params["nireq"]),
                num_streams=int(svp_params["num_streams"]),
                device=svp_params["device"],
                normalize_factor=float(svp_params["normalize_factor"]),
                batch_size=int(svp_params["batch_size"]),
                extra_encode_flags=svp_extra_flags,
            )
            _svp_probe = probe_video(svp_input_video, ffmpeg_bin)
            _svp_total = int(_svp_probe["nb_frames"]) if str(_svp_probe["nb_frames"]).isdigit() else 0
            status = st.empty()
            with st.spinner("Running iVSR SVP inference ..."):
                rc, stderr, elapsed = run_ffmpeg(cmd, status, total_frames=_svp_total)
            status.empty()

            if rc == 0:
                with st.spinner("Optimising output for playback..."):
                    apply_faststart(svp_output_video, ffmpeg_bin)
                in_info = probe_video(svp_input_video, ffmpeg_bin)
                out_info = probe_video(svp_output_video, ffmpeg_bin)
                throughput = (
                    round(int(out_info["nb_frames"]) / elapsed, 1)
                    if out_info["nb_frames"] != "N/A" else None
                )
                st.session_state["svp_result"] = {
                    "output": svp_output_video,
                    "input": svp_input_video,
                    "elapsed": elapsed,
                    "input_size": in_info.get("size", 0),
                    "output_size": out_info.get("size", 0),
                    "throughput": throughput,
                    "in_info": in_info,
                    "out_info": out_info,
                }
            else:
                st.session_state.pop("svp_result", None)
                st.error(f"FFmpeg failed (exit code {rc})")
                with st.expander("FFmpeg stderr"):
                    st.code(stderr, language="text")

    # ── Results (persists across reruns via session_state) ────────────────────
    if "svp_result" in st.session_state:
        res = st.session_state["svp_result"]
        ii = res.get("in_info", {})
        oi = res.get("out_info", {})
        st.success(f"Done in **{res['elapsed']:.1f}s** | Throughput: {res['throughput']} fps" if res.get('throughput') else f"Done in **{res['elapsed']:.1f}s**")

        # ── Before / After comparison ─────────────────────────────────────────
        size_str = lambda s: f"{s / 1024 / 1024:.1f} MB" if s else "N/A"
        br_str = lambda k: f"{k:,} kbps" if k else "N/A"
        dur_str = lambda d: f"{int(d // 60)}m {d % 60:.1f}s" if d else "N/A"

        in_size = ii.get('size', res.get('input_size', 0))
        out_size = oi.get('size', res.get('output_size', 0))
        in_br = ii.get('bitrate_kbps', 0)
        out_br = oi.get('bitrate_kbps', 0)
        size_reduction = round((1 - out_size / in_size) * 100, 1) if in_size > 0 else 0
        br_reduction = round((1 - out_br / in_br) * 100, 1) if in_br > 0 else 0

        col_in2, col_out2 = st.columns(2)
        with col_in2:
            st.markdown("**Input Video (original)**")
            st.video(make_preview(res["input"], ffmpeg_bin))
            stats_block(
                stat("File Size",  size_str(in_size)),
                stat("Bitrate",    br_str(in_br)),
                stat("Duration",   dur_str(ii.get('duration', 0))),
                stat("Resolution", f"{ii.get('width','?')}×{ii.get('height','?')}" if ii.get('width') else "N/A"),
            )
        with col_out2:
            st.markdown("**Output Video (SVP-filtered)**")
            if os.path.isfile(res["output"]):
                st.video(res["output"])
                stats_block(
                    stat("File Size",  size_str(out_size), f"-{size_reduction}%" if size_reduction > 0 else ""),
                    stat("Bitrate",    br_str(out_br),     f"-{br_reduction}%" if br_reduction > 0 else ""),
                    stat("Duration",   dur_str(oi.get('duration', 0))),
                    stat("Resolution", f"{oi.get('width','?')}×{oi.get('height','?')}" if oi.get('width') else "N/A"),
                )
                st.download_button(
                    "Download Output",
                    data=open(res["output"], "rb").read(),
                    file_name="svp_output.mp4",
                    mime="video/mp4",
                    key="svp_download",
                )
            else:
                st.warning("Output file not found.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — RIFE Frame Interpolation
# ═════════════════════════════════════════════════════════════════════════════
with tab_rife:
    st.subheader("RIFE Frame Interpolation")
    st.markdown(
        "Double the frame rate of a video (e.g. 30 fps → 60 fps) using the "
        "**RIFE** (Real-time Intermediate Flow Estimation) model. "
        "Generates a temporally interpolated frame between each pair of originals."
    )

    st.info(
        "**RIFE model constraints**  \n"
        "- Input tensor: `[1, 6, H, W]` — two RGB frames packed channel-wise (NCHW float32)  \n"
        "- Width and Height are **automatically padded** to the nearest multiple of 128 and cropped back — no manual alignment needed  \n"
        "- Input pixels are normalised **0–255 → [0.0, 1.0]** automatically in the C backend  \n"
        "- model_type `5` · nif `2` (fixed) · normalize_factor `1.0` (fixed)  \n"
        "- **2x FPS mode**: original frames are preserved; one interpolated frame is "
        "inserted between each pair → 2× frame count  \n"
        "- **Midpoints-only mode**: only the interpolated frames are emitted (same frame count, "
        "lower sharpness — useful for inspection)"
    )

    st.markdown("---")

    # ── Model upload ──────────────────────────────────────────────────────────
    st.markdown("#### Model")

    rife_default_exists = os.path.isfile(DEFAULT_RIFE_MODEL)
    col_xl, col_bin = st.columns(2)
    with col_xl:
        rife_xml_file = st.file_uploader(
            "Upload RIFE Model (.xml)",
            type=["xml"],
            key="rife_xml_upload",
            help="OpenVINO IR model XML file for RIFE.",
        )
    with col_bin:
        rife_bin_file = st.file_uploader(
            "Upload RIFE Weights (.bin)",
            type=["bin"],
            key="rife_bin_upload",
            help="OpenVINO IR binary weights. Must match the .xml.",
        )
    rife_model_path = save_model_uploads(rife_xml_file, rife_bin_file, "rife")

    if not rife_model_path:
        if rife_default_exists:
            use_default_rife = st.checkbox(
                f"Use default RIFE model  (`{DEFAULT_RIFE_MODEL}`)",
                value=True,
                key="rife_use_default",
            )
            if use_default_rife:
                rife_model_path = DEFAULT_RIFE_MODEL
        else:
            pass

    st.markdown("---")

    # ── Input video ───────────────────────────────────────────────────────────
    st.markdown("#### Input Video")
    rife_video_file = st.file_uploader(
        "Upload Input Video",
        type=["mp4", "mkv", "avi", "mov", "webm", "ts"],
        key="rife_video_upload",
    )
    rife_input_video = save_upload(rife_video_file, ".mp4", "rife_video")
    rife_output_video = os.path.join(tempfile.gettempdir(), "ivsr_rife_output.mp4")

    _rife_src_info: dict = {}
    if rife_input_video:
        _rife_src_info = probe_video(rife_input_video, ffmpeg_bin)
        st.caption(
            f"Source: **{_rife_src_info['width']}x{_rife_src_info['height']}** · "
            f"**{_rife_src_info['fps']} fps** · **{_rife_src_info['nb_frames']} frames** · "
            f"{int(_rife_src_info['duration'] // 60)}m {_rife_src_info['duration'] % 60:.1f}s"
        )

    st.markdown("---")

    # ── Scale (128-aligned) ───────────────────────────────────────────────────
    st.markdown("#### Scale (must be multiples of 128)")

    _def_w = _rife_src_info["width"] if _rife_src_info.get("width") else 640
    _def_h = _rife_src_info["height"] if _rife_src_info.get("height") else 512

    col_w, col_h = st.columns(2)
    with col_w:
        rife_w = st.number_input(
            "Target Width",
            min_value=2,
            max_value=7680,
            value=_def_w,
            step=2,
            key="rife_width",
            help="Desired output width. Automatically padded to the next multiple of 128 for model input and cropped back.",
        )
    with col_h:
        rife_h = st.number_input(
            "Target Height",
            min_value=2,
            max_value=4320,
            value=_def_h,
            step=2,
            key="rife_height",
            help="Desired output height. Automatically padded to the next multiple of 128 for model input and cropped back.",
        )

    aligned_w = align_to_128(int(rife_w))
    aligned_h = align_to_128(int(rife_h))
    if aligned_w != int(rife_w) or aligned_h != int(rife_h):
        st.caption(
            f"Output: **{int(rife_w)}x{int(rife_h)}** — padded to **{aligned_w}x{aligned_h}** "
            f"for model input, cropped back after inference."
        )
    else:
        st.caption(f"Output: **{int(rife_w)}x{int(rife_h)}** (already 128-aligned — no padding needed)")

    st.markdown("---")

    # ── Output mode ───────────────────────────────────────────────────────────
    st.markdown("#### Output Mode")
    rife_mode_label = st.radio(
        "Mode",
        ["2x FPS - originals + interpolated (recommended)"],
        # ["2x FPS - originals + interpolated (recommended)", "Midpoints only"],
        index=0,
        key="rife_mode",
        help=(
            "2x FPS: keeps all original frames and inserts one interpolated frame "
            "between each pair, resulting in 2x frame count with full sharpness on originals.  "
            # "Midpoints only: emits only the RIFE-generated frames. Used for "
            # "debugging or special effects."
        ),
    )
    rife_mode = "2xfps" if rife_mode_label.startswith("2x FPS") else "midpoints"

    rife_smooth_filter = "none"
    # if rife_mode == "midpoints":
    #     rife_smooth_filter = st.selectbox(
    #         "Post-processing smooth filter",
    #         ["none", "hqdn3d (temporal denoise — fixes spatial wave/ripple)", "deflicker (fixes luminance flicker)"],
    #         index=1,
    #         key="rife_smooth_filter",
    #         help=(
    #             "**hqdn3d**: averages each pixel across neighbouring frames in time — "
    #             "best for the spatial wobble/wave caused by independent RIFE optical-flow errors.  \n"
    #             "**deflicker**: normalises per-frame brightness — only helps with luminance flicker.  \n"
    #             "**none**: no post-processing."
    #         ),
    #     )
    #     # Extract the key before the first space/parenthesis
    #     rife_smooth_filter = rife_smooth_filter.split()[0]

    st.markdown("---")

    # ── Inference parameters ──────────────────────────────────────────────────
    st.markdown("#### Inference Parameters")

    col_d, col_nr, col_ns = st.columns(3)
    with col_d:
        rife_device = st.selectbox(
            "Target Device",
            DEVICE_OPTIONS,
            index=0,
            key="rife_device",
            help="Hardware to run RIFE inference on.",
        )
    with col_nr:
        rife_nireq = st.number_input(
            "Inference Requests (nireq)",
            min_value=1,
            max_value=32,
            value=1,
            step=1,
            key="rife_nireq",
            help="Number of parallel OpenVINO inference requests.",
        )
    with col_ns:
        _rife_gpu = rife_device.startswith("GPU") or rife_device in ("MULTI:GPU.0,GPU.1", "AUTO")
        rife_num_streams = st.number_input(
            "GPU Streams (num_streams)",
            min_value=1,
            max_value=16,
            value=1,
            step=1,
            disabled=not _rife_gpu,
            key="rife_num_streams",
            help="Throughput streams for GPU only.",
        )

    st.caption(
        "**nif (filter):** 1 *(RIFE backend manages its own 2-frame sliding window internally)* | "
        "**normalize_factor:** 1.0 *(fixed)* | **model_type:** 5"
    )

    # ── Encoding options ──────────────────────────────────────────────────────
    with st.expander("RIFE Encoding Options"):
        col_enc1, col_enc2 = st.columns(2)
        with col_enc1:
            rife_codec = st.selectbox(
                "Output Codec",
                ["libx264", "libx265", "copy"],
                index=0,
                key="rife_codec",
                help="Encoder for the output video.",
            )
        with col_enc2:
            rife_crf = st.slider(
                "CRF (quality, lower=better)",
                min_value=0,
                max_value=51,
                value=23,
                key="rife_crf",
                help="Constant Rate Factor.",
            )
        rife_encode_flags = (
            f"-c:v {rife_codec} -crf {rife_crf}" if rife_codec != "copy" else "-c:v copy"
        )

    # ── Generated command preview ─────────────────────────────────────────────
    if rife_model_path and rife_input_video:
        st.markdown("---")
        _rife_preview_cmd = build_rife_ffmpeg_cmd(
            ffmpeg_bin=ffmpeg_bin,
            input_video=rife_input_video,
            output_video=rife_output_video,
            model_path=rife_model_path,
            target_w=int(rife_w),
            target_h=int(rife_h),
            device=rife_device,
            nireq=int(rife_nireq),
            num_streams=int(rife_num_streams),
            mode=rife_mode,
            src_fps=float(_rife_src_info.get("fps") or 25.0),
            extra_encode_flags=rife_encode_flags,
            smooth_filter=rife_smooth_filter,
        )
        with st.expander("RIFE Generated FFmpeg Command", expanded=False):
            st.code(cmd_to_display_string(_rife_preview_cmd), language="bash")

    # ── Run button ────────────────────────────────────────────────────────────
    rife_run = st.button("Run RIFE Interpolation", key="rife_run")
    if rife_run:
        errors = []
        if not rife_model_path:
            errors.append("Upload a RIFE model (.xml + .bin) or enable the default model above.")
        if not rife_input_video:
            errors.append("Upload an input video file.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            _rife_run_cmd = build_rife_ffmpeg_cmd(
                ffmpeg_bin=ffmpeg_bin,
                input_video=rife_input_video,
                output_video=rife_output_video,
                model_path=rife_model_path,
                target_w=int(rife_w),
                target_h=int(rife_h),
                device=rife_device,
                nireq=int(rife_nireq),
                num_streams=int(rife_num_streams),
                mode=rife_mode,
                src_fps=float(_rife_src_info.get("fps") or 25.0),
                extra_encode_flags=rife_encode_flags,
                smooth_filter=rife_smooth_filter,
            )
            _rife_probe = probe_video(rife_input_video, ffmpeg_bin)
            _rife_src_frames = (
                int(_rife_probe["nb_frames"])
                if str(_rife_probe["nb_frames"]).isdigit() else 0
            )
            _rife_expected = _rife_src_frames * 2 if rife_mode == "2xfps" else _rife_src_frames
            rife_status = st.empty()
            with st.spinner("Running RIFE frame interpolation..."):
                rife_rc, rife_stderr, rife_elapsed = run_ffmpeg(
                    _rife_run_cmd, rife_status, total_frames=_rife_expected
                )
            rife_status.empty()

            if rife_rc == 0:
                with st.spinner("Optimising output for playback..."):
                    apply_faststart(rife_output_video, ffmpeg_bin)
                in_info = probe_video(rife_input_video, ffmpeg_bin)
                out_info = probe_video(rife_output_video, ffmpeg_bin)
                throughput = (
                    round(int(out_info["nb_frames"]) / rife_elapsed, 1)
                    if str(out_info.get("nb_frames", "N/A")).isdigit() else None
                )
                st.session_state["rife_result"] = {
                    "output": rife_output_video,
                    "input": rife_input_video,
                    "elapsed": rife_elapsed,
                    "in_info": in_info,
                    "out_info": out_info,
                    "throughput": throughput,
                    "mode": rife_mode,
                }
            else:
                st.session_state.pop("rife_result", None)
                st.error(f"RIFE FFmpeg failed (exit code {rife_rc})")
                with st.expander("FFmpeg stderr"):
                    st.code(rife_stderr, language="text")

    # ── Results ───────────────────────────────────────────────────────────────
    if "rife_result" in st.session_state:
        res = st.session_state["rife_result"]
        ii = res.get("in_info", {})
        oi = res.get("out_info", {})
        mode_label = (
            "2x FPS (original + interpolated)"
            if res.get("mode") == "2xfps" else "midpoints only"
        )
        tp_str = f" | Throughput: {res['throughput']} fps" if res.get("throughput") else ""
        st.success(f"RIFE done in **{res['elapsed']:.1f}s**{tp_str}")

        res_str  = lambda w, h: f"{w}x{h}" if w else "N/A"
        size_str = lambda s: f"{s / 1024 / 1024:.1f} MB" if s else "N/A"
        dur_str  = lambda d: f"{int(d // 60)}m {d % 60:.1f}s" if d else "N/A"
        fps_str  = lambda f: f"{f} fps" if f else "N/A"

        in_frames  = ii.get("nb_frames", "N/A")
        out_frames = oi.get("nb_frames", "N/A")
        in_fps     = ii.get("fps", 0)
        out_fps    = oi.get("fps", 0)

        col_in, col_out = st.columns(2)
        with col_in:
            st.markdown("**Input Video**")
            st.video(make_preview(res["input"], ffmpeg_bin))
            stats_block(
                stat("Resolution", res_str(ii.get("width"), ii.get("height"))),
                stat("Frame Rate", fps_str(in_fps)),
                stat("Frames",     str(in_frames)),
                stat("File Size",  size_str(ii.get("size", 0))),
                stat("Duration",   dur_str(ii.get("duration", 0))),
            )
        with col_out:
            st.markdown(f"**Output Video ({mode_label})**")
            if os.path.isfile(res["output"]):
                st.video(res["output"])
                fps_delta = ""
                if in_fps and out_fps and out_fps > in_fps:
                    fps_delta = f"+{round(out_fps - in_fps, 2)} fps"
                frames_delta = ""
                if str(in_frames).isdigit() and str(out_frames).isdigit():
                    frames_delta = f"+{int(out_frames) - int(in_frames)}"
                stats_block(
                    stat("Resolution", res_str(oi.get("width"), oi.get("height"))),
                    stat("Frame Rate", fps_str(out_fps), fps_delta),
                    stat("Frames",     str(out_frames), frames_delta),
                    stat("File Size",  size_str(oi.get("size", 0))),
                    stat("Duration",   dur_str(oi.get("duration", 0))),
                )
                st.download_button(
                    "Download Output",
                    data=open(res["output"], "rb").read(),
                    file_name="rife_output.mp4",
                    mime="video/mp4",
                    key="rife_download",
                )
            else:
                st.warning("Output file not found.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — VideoSeal Invisible Watermarking
# ═════════════════════════════════════════════════════════════════════════════
with tab_vs:
    st.subheader("Invisible Watermarking — VideoSeal (model_type=6)")
    st.markdown(
        "Embed an imperceptible per-frame watermark into a video using the "
        "[Meta VideoSeal](https://ai.meta.com/research/publications/videoseal/) model "
        "running through the iVSR FFmpeg plugin, then extract and verify it."
    )
    st.info(
        "**How it works**  \n"
        "- The watermark bit-string is baked into the model at export time "
        "(`videoseal/export_videoseal_openvino.py`).  \n"
        "- The FFmpeg filter applies the model frame-by-frame: `model_type=6`, "
        "`nif=1`, `normalize_factor=1.0` (normalisation is inside the model).  \n"
        "- Extraction uses Meta's `model.detect()` API on the first frame to "
        "recover the payload and a watermark-presence confidence score."
    )

    st.markdown("---")

    # ── Two sub-sections: Embed and Extract ────────────────────────────────
    vs_embed_tab, vs_extract_tab = st.tabs(["Embed Watermark", "Extract Watermark"])

    # ─────────────────────────────────────────────────────────────────────────
    # SUB-TAB A — Embed
    # ─────────────────────────────────────────────────────────────────────────
    with vs_embed_tab:
        st.markdown("#### Model")
        st.caption(
            "The VideoSeal OpenVINO IR model must be exported **with the watermark "
            "payload baked in** using `videoseal/export_videoseal_openvino.py`. "
            "The model is **resolution-specific** — set the target size to match "
            "the resolution used during export (default: 1280×720)."
        )

        col_vx, col_vb = st.columns(2)
        with col_vx:
            vs_xml_file = st.file_uploader(
                "Upload VideoSeal Model (.xml)",
                type=["xml"],
                key="vs_xml_upload",
                help="OpenVINO IR XML produced by export_videoseal_openvino.py.",
            )
        with col_vb:
            vs_bin_file = st.file_uploader(
                "Upload VideoSeal Weights (.bin)",
                type=["bin"],
                key="vs_bin_upload",
                help="OpenVINO IR binary weights matching the .xml.",
            )
        vs_model_path = save_model_uploads(vs_xml_file, vs_bin_file, "vs")

        st.markdown("---")
        st.markdown("#### Input Video")
        vs_video_file = st.file_uploader(
            "Upload Input Video",
            type=["mp4", "mkv", "avi", "mov", "webm", "ts"],
            key="vs_video_upload",
        )
        vs_input_video = save_upload(vs_video_file, ".mp4", "vs_video")
        vs_output_video = os.path.join(tempfile.gettempdir(), "ivsr_vs_output.mp4")

        _vs_src_info: dict = {}
        if vs_input_video:
            _vs_src_info = probe_video(vs_input_video, ffmpeg_bin)
            st.caption(
                f"Source: **{_vs_src_info['width']}×{_vs_src_info['height']}** · "
                f"**{_vs_src_info['fps']} fps** · "
                f"**{_vs_src_info['nb_frames']} frames** · "
                f"{int(_vs_src_info['duration'] // 60)}m "
                f"{_vs_src_info['duration'] % 60:.1f}s"
            )

        st.markdown("---")
        st.markdown("#### Target Resolution")
        st.caption(
            "Must match the resolution the model was exported for. "
            "Input frames are scaled to this size before watermarking."
        )
        col_vw, col_vh = st.columns(2)
        with col_vw:
            vs_w = st.number_input(
                "Target Width",
                min_value=2, max_value=7680,
                value=_vs_src_info.get("width") or 1280,
                step=2,
                key="vs_width",
            )
        with col_vh:
            vs_h = st.number_input(
                "Target Height",
                min_value=2, max_value=4320,
                value=_vs_src_info.get("height") or 720,
                step=2,
                key="vs_height",
            )

        st.markdown("---")
        st.markdown("#### Inference Parameters")
        col_vd, col_vnr, col_vns = st.columns(3)
        with col_vd:
            vs_device = st.selectbox(
                "Target Device", DEVICE_OPTIONS, index=0, key="vs_device",
            )
        with col_vnr:
            vs_nireq = st.number_input(
                "Inference Requests (nireq)",
                min_value=1, max_value=32, value=4, step=1,
                key="vs_nireq",
                help="Parallel OpenVINO inference requests.",
            )
        with col_vns:
            _vs_gpu = vs_device.startswith("GPU") or vs_device in ("MULTI:GPU.0,GPU.1", "AUTO")
            vs_num_streams = st.number_input(
                "GPU Streams (num_streams)",
                min_value=1, max_value=16, value=1, step=1,
                disabled=not _vs_gpu,
                key="vs_num_streams",
            )
        st.caption("**model_type:** 6 · **nif:** 1 · **normalize_factor:** 1.0 *(all fixed)*")

        with st.expander("Encoding Options"):
            col_ec1, col_ec2 = st.columns(2)
            with col_ec1:
                vs_codec_choice = st.selectbox(
                    "Output Codec",
                    ["libx264", "libx265", "FFV1 (lossless, rgb24)"],
                    index=0,
                    key="vs_codec",
                    help=(
                        "**FFV1 (lossless, rgb24)**: bit-exact output — preserves every "
                        "watermarked pixel. Recommended for reliable single-frame extraction.  \n"
                        "**libx264/libx265**: lossy — multi-frame averaging compensates."
                    ),
                )
            vs_lossless = vs_codec_choice == "FFV1 (lossless, rgb24)"
            vs_codec = vs_codec_choice if not vs_lossless else "libx264"
            with col_ec2:
                vs_crf = st.slider(
                    "CRF (quality, lower=better)",
                    min_value=0, max_value=51, value=18,
                    key="vs_crf",
                    disabled=vs_lossless,
                    help="Ignored when FFV1 lossless is selected.",
                )

        st.markdown("---")
        st.markdown("#### FFmpeg Binary & Library Path")
        st.caption(
            "VideoSeal requires the **locally built** iVSR-patched FFmpeg "
            "(`model_type=6` is not in the system build). The `LD_LIBRARY_PATH` "
            "must point to the freshly built `libavfilter` so the patched "
            "`.so` is loaded instead of the system one."
        )
        col_fb, col_lp = st.columns(2)
        with col_fb:
            _local_ffmpeg_exists = os.path.isfile(DEFAULT_LOCAL_FFMPEG)
            vs_ffmpeg_bin = st.text_input(
                "FFmpeg binary (VideoSeal)",
                value=DEFAULT_LOCAL_FFMPEG if _local_ffmpeg_exists else ffmpeg_bin,
                key="vs_ffmpeg_bin",
                help="Path to the locally built FFmpeg binary that has model_type=6 compiled in.",
            )
        with col_lp:
            vs_lib_path = st.text_input(
                "LD_LIBRARY_PATH (libavfilter dir)",
                value=DEFAULT_LOCAL_LIBAVFILTER,
                key="vs_lib_path",
                help=(
                    "Directory containing the patched libavfilter.so.11. "
                    "Leave blank if the system library is already up to date."
                ),
            )

        # ── Command preview ───────────────────────────────────────────────────
        if vs_model_path and vs_input_video:
            _vs_preview_cmd = build_videoseal_ffmpeg_cmd(
                ffmpeg_bin=vs_ffmpeg_bin,
                input_video=vs_input_video,
                output_video=vs_output_video,
                model_path=vs_model_path,
                target_w=int(vs_w),
                target_h=int(vs_h),
                device=vs_device,
                nireq=int(vs_nireq),
                num_streams=int(vs_num_streams),
                crf=int(vs_crf),
                codec=vs_codec,
                lossless=vs_lossless,
            )
            _vs_env_prefix = (
                f"LD_LIBRARY_PATH={vs_lib_path} \\\n" if vs_lib_path.strip() else ""
            )
            with st.expander("Generated FFmpeg Command", expanded=False):
                st.code(
                    _vs_env_prefix + cmd_to_display_string(_vs_preview_cmd),
                    language="bash",
                )

        # ── Run button ────────────────────────────────────────────────────────
        st.markdown("---")
        vs_run = st.button("Embed Watermark", key="vs_run")
        if vs_run:
            errors = []
            if not vs_model_path:
                errors.append("Provide a VideoSeal model (.xml + .bin) or enable the default model above.")
            if not vs_input_video:
                errors.append("Upload an input video file.")
            if not os.path.isfile(vs_ffmpeg_bin):
                errors.append(
                    f"FFmpeg binary not found: `{vs_ffmpeg_bin}`.  \n"
                    "Build the iVSR-patched FFmpeg first: `cd sandesh/iVSR && bash build.sh`"
                )
            for e in errors:
                st.error(e)

            if not errors:
                _vs_cmd = build_videoseal_ffmpeg_cmd(
                    ffmpeg_bin=vs_ffmpeg_bin,
                    input_video=vs_input_video,
                    output_video=vs_output_video,
                    model_path=vs_model_path,
                    target_w=int(vs_w),
                    target_h=int(vs_h),
                    device=vs_device,
                    nireq=int(vs_nireq),
                    num_streams=int(vs_num_streams),
                    crf=int(vs_crf),
                    codec=vs_codec,
                    lossless=vs_lossless,
                )
                _vs_extra_env = (
                    {"LD_LIBRARY_PATH": vs_lib_path.strip()}
                    if vs_lib_path.strip() else None
                )
                _vs_probe = probe_video(vs_input_video, ffmpeg_bin)
                _vs_total = (
                    int(_vs_probe["nb_frames"])
                    if str(_vs_probe["nb_frames"]).isdigit() else 0
                )
                vs_status = st.empty()
                with st.spinner("Embedding watermark via VideoSeal..."):
                    vs_rc, vs_stderr, vs_elapsed = run_ffmpeg(
                        _vs_cmd, vs_status,
                        total_frames=_vs_total,
                        extra_env=_vs_extra_env,
                    )
                vs_status.empty()

                if vs_rc == 0:
                    with st.spinner("Optimising output for playback..."):
                        apply_faststart(vs_output_video, vs_ffmpeg_bin)
                    in_info = probe_video(vs_input_video, ffmpeg_bin)
                    out_info = probe_video(vs_output_video, ffmpeg_bin)
                    throughput = (
                        round(int(out_info["nb_frames"]) / vs_elapsed, 1)
                        if str(out_info.get("nb_frames", "N/A")).isdigit() else None
                    )
                    st.session_state["vs_result"] = {
                        "output": vs_output_video,
                        "input": vs_input_video,
                        "elapsed": vs_elapsed,
                        "in_info": in_info,
                        "out_info": out_info,
                        "throughput": throughput,
                    }
                else:
                    st.session_state.pop("vs_result", None)
                    st.error(f"FFmpeg failed (exit code {vs_rc})")
                    with st.expander("FFmpeg stderr"):
                        st.code(vs_stderr, language="text")

        # ── Results ───────────────────────────────────────────────────────────
        if "vs_result" in st.session_state:
            res = st.session_state["vs_result"]
            ii = res.get("in_info", {})
            oi = res.get("out_info", {})
            tp_str = f" | Throughput: {res['throughput']} fps" if res.get("throughput") else ""
            st.success(f"Done in **{res['elapsed']:.1f}s**{tp_str}")

            size_str  = lambda s: f"{s / 1024 / 1024:.1f} MB" if s else "N/A"
            br_str    = lambda k: f"{k:,} kbps" if k else "N/A"
            dur_str   = lambda d: f"{int(d // 60)}m {d % 60:.1f}s" if d else "N/A"
            res_str   = lambda w, h: f"{w}×{h}" if w else "N/A"

            col_vi, col_vo = st.columns(2)
            with col_vi:
                st.markdown("**Input Video (original)**")
                st.video(make_preview(res["input"], ffmpeg_bin))
                stats_block(
                    stat("Resolution", res_str(ii.get("width"), ii.get("height"))),
                    stat("Bitrate",    br_str(ii.get("bitrate_kbps", 0))),
                    stat("File Size",  size_str(ii.get("size", 0))),
                    stat("Duration",   dur_str(ii.get("duration", 0))),
                )
            with col_vo:
                st.markdown("**Output Video (watermarked — visually identical)**")
                if os.path.isfile(res["output"]):
                    st.video(res["output"])
                    stats_block(
                        stat("Resolution", res_str(oi.get("width"), oi.get("height"))),
                        stat("Bitrate",    br_str(oi.get("bitrate_kbps", 0))),
                        stat("File Size",  size_str(oi.get("size", 0))),
                        stat("Duration",   dur_str(oi.get("duration", 0))),
                    )
                    st.download_button(
                        "Download Watermarked Video",
                        data=open(res["output"], "rb").read(),
                        file_name="videoseal_watermarked.mp4",
                        mime="video/mp4",
                        key="vs_download",
                    )
                else:
                    st.warning("Output file not found.")

    # ─────────────────────────────────────────────────────────────────────────
    # SUB-TAB B — Extract
    # ─────────────────────────────────────────────────────────────────────────
    with vs_extract_tab:
        st.markdown("#### Extract Watermark from Video")
        st.markdown(
            "Recover the embedded payload from a VideoSeal-watermarked video. "
            "Up to 20 frames are sampled evenly and logit scores are averaged "
            "before thresholding — this makes extraction robust to H.264/H.265 "
            "compression artifacts in high-motion content."
        )

        st.markdown("---")

        # ── Detector Model (OpenVINO IR) ──────────────────────────────────────
        st.markdown("#### Detector Model (OpenVINO IR)")
        st.caption(
            "Export once with: "
            "`python3 videoseal/export_videoseal_detector_openvino.py`  \n"
            "Produces `videoseal_detector.xml` + `videoseal_detector.bin`"
        )
        col_dxml, col_dbin = st.columns(2)
        with col_dxml:
            vs_det_xml = st.file_uploader(
                "Detector Model (.xml)",
                type=["xml"],
                key="vs_det_xml_upload",
                help="OpenVINO IR XML for the VideoSeal detector.",
            )
        with col_dbin:
            vs_det_bin = st.file_uploader(
                "Detector Weights (.bin)",
                type=["bin"],
                key="vs_det_bin_upload",
                help="OpenVINO IR binary weights. Must match the .xml.",
            )
        vs_detector_path = save_model_uploads(vs_det_xml, vs_det_bin, "vs_detector")

        st.markdown("---")

        # ── Video source ──────────────────────────────────────────────────────
        st.markdown("#### Video Source")
        _vs_embed_result = st.session_state.get("vs_result")
        if _vs_embed_result and os.path.isfile(_vs_embed_result.get("output", "")):
            use_embed_output = st.checkbox(
                "Use watermarked video produced by the Embed tab",
                value=True,
                key="vs_extract_use_embed",
            )
            if use_embed_output:
                vs_extract_src = _vs_embed_result["output"]
                st.caption(f"Using: `{vs_extract_src}`")
            else:
                vs_extract_src = None
        else:
            use_embed_output = False
            vs_extract_src = None

        if not use_embed_output:
            vs_ext_video_file = st.file_uploader(
                "Upload Watermarked Video",
                type=["mp4", "mkv", "avi", "mov", "webm", "ts"],
                key="vs_extract_upload",
                help="Any video watermarked with VideoSeal.",
            )
            vs_extract_src = save_upload(vs_ext_video_file, ".mp4", "vs_extract_video")

        st.markdown("---")

        # ── Run button ────────────────────────────────────────────────────────
        vs_extract_run = st.button("Extract Watermark", key="vs_extract_run")
        if vs_extract_run:
            errors = []
            if not vs_extract_src:
                errors.append("Provide a watermarked video to extract from.")
            if not vs_detector_path:
                errors.append("Upload both the detector .xml and .bin files.")
            if errors:
                for e in errors:
                    st.error(e)
            else:
                with st.spinner("Loading OpenVINO detector and extracting payload..."):
                    try:
                        extracted_text, confidence_pct = extract_videoseal_watermark_ov(
                            vs_extract_src, vs_detector_path
                        )
                        st.session_state["vs_extract_result"] = {
                            "text": extracted_text,
                            "confidence": confidence_pct,
                            "src": vs_extract_src,
                        }
                    except Exception as ex:
                        st.session_state.pop("vs_extract_result", None)
                        st.error(f"Extraction failed: {ex}")

        # ── Results ───────────────────────────────────────────────────────────
        if "vs_extract_result" in st.session_state:
            er = st.session_state["vs_extract_result"]
            st.success("Watermark extracted")

            if os.path.isfile(er["src"]):
                st.markdown("**Analysed Video**")
                st.video(make_preview(er["src"], ffmpeg_bin))

            st.markdown("---")
            col_et, col_ec = st.columns([3, 1])
            with col_et:
                st.markdown("**Extracted Watermark Text**")
                if er["text"]:
                    st.code(er["text"], language="text")
                else:
                    st.warning(
                        "No readable text was found. The video may not be "
                        "watermarked, or was watermarked with a different payload."
                    )
            with col_ec:
                st.markdown("**Detection Confidence**")
                if er["confidence"] > 0:
                    conf = er["confidence"]
                    colour = "#21c354" if conf >= 80 else "#f0a500" if conf >= 50 else "#e03c31"
                    st.markdown(
                        f'<p style="font-size:2rem;font-weight:700;color:{colour}">'
                        f'{conf:.1f}%</p>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("Confidence score not available (older model format).")

