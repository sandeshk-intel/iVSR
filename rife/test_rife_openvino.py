#!/usr/bin/env python3
"""
Test a RIFE OpenVINO IR model using OpenCV to read the video.
Dynamically pads and crops frames to satisfy RIFE_HDv3's multiple-of-128 
geometry requirement, and uses RGB color space to match the PyTorch export.

Usage:
    python3 test_rife_openvino.py --video input.mp4 --model ./rife_ov/rife.xml
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import openvino as ov


def parse_args():
    p = argparse.ArgumentParser(description="Test RIFE OpenVINO model on a video using OpenCV")
    p.add_argument("--video", required=True, help="Path to input video")
    p.add_argument("--model", required=True, help="Path to RIFE .xml model")
    p.add_argument("--num_frames", type=int, default=10, help="Number of interpolated frames to save")
    p.add_argument("--output_dir", default="./rife_test_output", help="Directory for output PNGs")
    p.add_argument("--device", default="CPU", help="OpenVINO device (CPU, GPU, etc.)")
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load OpenVINO model
    core = ov.Core()
    model = core.read_model(args.model)

    input_shape = list(model.inputs[0].partial_shape)
    model_h = int(str(input_shape[2]))
    model_w = int(str(input_shape[3]))
    print(f"Model expected input shape: [1, 6, {model_h}, {model_w}]")

    compiled = core.compile_model(model, args.device)
    infer = compiled.create_infer_request()
    input_name = compiled.input(0).any_name

    # 2. Open video with OpenCV
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Error: Could not open video {args.video}")

    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source video resolution: {video_w}x{video_h}")

    # Calculate padding needed to reach model requirements
    pad_h = model_h - video_h
    pad_w = model_w - video_w
    if pad_h < 0 or pad_w < 0:
        sys.exit("Error: Video is larger than the exported model's dimensions.")
        
    print(f"Padding frames by (H:{pad_h}, W:{pad_w}) to avoid stretching geometry...")

    # Read the very first frame
    ret, prev_frame = cap.read()
    if not ret:
        sys.exit("Error: Could not read the first frame.")

    # Pad the first frame
    prev_padded = cv2.copyMakeBorder(
        prev_frame, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # 3. Main inference loop
    saved = 0
    while saved < args.num_frames:
        ret, curr_frame = cap.read()
        if not ret:
            print("End of video reached.")
            break

        # Pad the current frame
        curr_padded = cv2.copyMakeBorder(
            curr_frame, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        # OpenCV frames are natively BGR. Convert to RGB and normalize to [0, 1]
        prev_rgb = cv2.cvtColor(prev_padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        curr_rgb = cv2.cvtColor(curr_padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Transpose [H, W, C] -> [C, H, W]
        f0 = prev_rgb.transpose(2, 0, 1)
        f1 = curr_rgb.transpose(2, 0, 1)

        # Concatenate channels to [1, 6, H, W]
        tensor = np.concatenate([f0, f1], axis=0)[np.newaxis].astype(np.float32)

        # Run inference
        infer.infer({input_name: tensor})
        output = infer.get_output_tensor().data.copy()

        # Output is [1, 3, H, W] in RGB. Transpose back to [H, W, C] and scale to uint8
        out_rgb = (np.clip(output[0], 0, 1).transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)

        # CROP the padding off to restore original video resolution!
        final_out_rgb = out_rgb[0:video_h, 0:video_w]

        # Convert back to BGR so OpenCV saves the colors correctly
        final_out_bgr = cv2.cvtColor(final_out_rgb, cv2.COLOR_RGB2BGR)

        # Save via OpenCV
        out_path = output_dir / f"rife_{saved + 1:03d}.png"
        cv2.imwrite(str(out_path), final_out_bgr)

        saved += 1
        print(f"  [{saved}/{args.num_frames}] Saved {out_path.name}")

        # Shift the padded frame for the next loop
        prev_padded = curr_padded

    cap.release()
    print(f"\nDone. {saved} interpolated frames saved to {output_dir}/")


if __name__ == "__main__":
    main()