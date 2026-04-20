#!/usr/bin/env python3
"""Extract a VideoSeal watermark using the OpenVINO IR detector model.

The detector IR is produced by export_videoseal_detector_openvino.py.
This script has no PyTorch dependency at runtime — only OpenVINO and OpenCV.

Pipeline:
  1. Sample up to --max-frames frames evenly across the video.
  2. For each frame: BGR→RGB, resize to 256×256 (INTER_AREA), NCHW float32 [0,1].
  3. Run the detector IR: input [1,3,256,256] → output [1,257] logits.
  4. Accumulate logits across frames (averaging noise out, reinforcing signal).
  5. Threshold the average → binary bits → ASCII text.

Usage:
    python3 videoseal/extract_videoseal_ov.py \\
        --model videoseal_detector.xml \\
        --video /path/to/watermarked.mp4

Options:
    --model       Path to the detector IR .xml file  (required)
    --video       Path to the watermarked video file  (required)
    --device      OpenVINO device  (default: CPU)
    --max-frames  Maximum number of frames to sample  (default: 20)
"""

import argparse
import sys
import cv2
import numpy as np
import openvino as ov


def decode_payload(scores: np.ndarray, num_bits: int = 256) -> str:
    bits = (scores > 0).astype(int).flatten().tolist()[:num_bits]
    chars: list[str] = []
    for i in range(0, len(bits) - 7, 8):
        code = int("".join(map(str, bits[i : i + 8])), 2)
        if code == 0:
            break
        chars.append(chr(code))
    return "".join(chars)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a VideoSeal watermark using an OpenVINO IR detector."
    )
    parser.add_argument("--model",      required=True, help="Path to videoseal_detector.xml")
    parser.add_argument("--video",      required=True, help="Path to watermarked video")
    parser.add_argument("--device",     default="CPU",  help="OpenVINO device (default: CPU)")
    parser.add_argument("--max-frames", type=int, default=20,
                        help="Max frames to sample for averaging (default: 20)")
    args = parser.parse_args()

    # ── Load OpenVINO model ───────────────────────────────────────────────────
    print(f"[*] Loading OpenVINO detector: {args.model}")
    core = ov.Core()
    compiled = core.compile_model(args.model, args.device)
    infer_req = compiled.create_infer_request()

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total == 0:
        print(f"[!] Could not read video: {args.video}")
        sys.exit(1)
    print(f"[*] Video: {w}x{h}  {fps:.2f} fps  {total} frames — {args.video}")

    # ── Sample frames and accumulate logits ───────────────────────────────────
    step = max(1, total // args.max_frames)
    scores: "np.ndarray | None" = None
    confidence_sum = 0.0
    sampled = 0

    print(f"[*] Sampling up to {args.max_frames} frames (step={step})...")
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, bgr = cap.read()
        if not ret:
            continue

        # BGR→RGB, downscale to 256×256 using INTER_AREA (correct for downscaling;
        # closely matches PyTorch antialias=True bilinear used inside model.detect())
        rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb256 = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
        inp   = rgb256.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0

        result = infer_req.infer({0: inp})
        preds  = list(result.values())[0]  # [1, 257]

        if preds.shape[-1] > 256:
            confidence_sum += float(1.0 / (1.0 + np.exp(-float(preds[0, 0]))))
            payload_logits = preds[0, 1:]
        else:
            payload_logits = preds[0]

        scores = payload_logits if scores is None else scores + payload_logits
        sampled += 1

    cap.release()

    if scores is None:
        print("[!] No frames could be read.")
        sys.exit(1)

    confidence = (confidence_sum / sampled * 100.0) if sampled > 0 else 0.0
    print(f"[*] Averaged over {sampled} frames")
    print(f"[*] Watermark Detection Confidence: {confidence:.1f}%")

    text = decode_payload(scores)

    print()
    print("=" * 40)
    print(f"   EXTRACTED WATERMARK: {text}")
    print("=" * 40)


if __name__ == "__main__":
    main()
