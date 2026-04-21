#!/usr/bin/env python3
"""
Test a SPAN OpenVINO IR super-resolution model using OpenCV.
Extracts frames from a video, upscales them, and saves the output as PNGs.

Usage:
    python3 test_span_openvino.py --video input.mp4 --model ./span_ir/spanx4_ch48.xml
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

def parse_args():
    p = argparse.ArgumentParser(description="Test SPAN OpenVINO model on a video")
    p.add_argument("--video", required=True, help="Path to input video")
    p.add_argument("--model", required=True, help="Path to SPAN .xml model")
    p.add_argument("--num_frames", type=int, default=10, help="Number of upscaled frames to save")
    p.add_argument("--output_dir", default="./span_test_output", help="Directory for output PNGs")
    p.add_argument("--device", default="CPU", help="OpenVINO device (CPU, GPU, etc.)")
    return p.parse_args()

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Open video to get source resolution
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[!] Error: Could not open video {args.video}")

    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[*] Source video resolution: {video_w}x{video_h}")

    # 2. Load OpenVINO model
    core = ov.Core()
    model = core.read_model(args.model)

    # PERFORMANCE HACK: The model was exported with dynamic shapes [-1, -1]. 
    # By locking the shape to the exact video dimensions before compiling, 
    # OpenVINO can aggressively optimize memory and layer execution for Xeon hardware.
    print("[*] Locking model shape to match video resolution...")
    model.reshape([1, 3, video_h, video_w])

    compiled = core.compile_model(model, args.device)
    infer = compiled.create_infer_request()
    input_name = compiled.input(0).any_name

    # 3. Main inference loop
    saved = 0
    print(f"[*] Starting inference for {args.num_frames} frames...")
    
    while saved < args.num_frames:
        ret, frame = cap.read()
        if not ret:
            print("[*] End of video reached.")
            break

        # --- PRE-PROCESSING ---
        # OpenCV frames are BGR [0-255]. SPAN expects RGB [0.0 - 1.0].
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Transpose [H, W, C] -> [C, H, W]
        rgb_transposed = rgb.transpose(2, 0, 1)
        
        # Add Batch dimension [C, H, W] -> [1, C, H, W]
        tensor = np.expand_dims(rgb_transposed, axis=0)

        # --- INFERENCE ---
        infer.infer({input_name: tensor})
        output = infer.get_output_tensor().data[0] # Drop the batch dimension: [3, H*scale, W*scale]

        # --- POST-PROCESSING ---
        # Output is [3, H_scaled, W_scaled] in RGB format (float32).
        # 1. Clip values to valid 0.0 - 1.0 range
        # 2. Transpose back to [H, W, C]
        # 3. Scale back to 0-255 and cast to uint8
        out_rgb = (np.clip(output, 0.0, 1.0).transpose(1, 2, 0) * 255.0).astype(np.uint8)

        # Convert back to BGR for OpenCV saving
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        # --- SAVE ---
        out_path = output_dir / f"span_upscaled_{saved + 1:03d}.png"
        cv2.imwrite(str(out_path), out_bgr)

        saved += 1
        # Print the output resolution dynamically on the first frame
        if saved == 1:
            out_h, out_w = out_bgr.shape[:2]
            print(f"[*] Upscaled output resolution confirmed: {out_w}x{out_h}")
            
        print(f"  [{saved}/{args.num_frames}] Saved {out_path.name}")

    cap.release()
    print(f"\n[*] Done. {saved} upscaled frames saved to {output_dir}/")

if __name__ == "__main__":
    main()