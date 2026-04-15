#!/usr/bin/env python3
"""
Export the RIFE_HDv3 PyTorch model to OpenVINO IR (.xml / .bin) 
by routing through ONNX Opset 16 to preserve the grid_sample layers.

Usage:
    python3 export_rife_openvino.py --model_dir ./train_log --height 768 --width 1280
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import openvino as ov


def parse_args():
    p = argparse.ArgumentParser(description="Export RIFE PyTorch model to OpenVINO via ONNX")
    p.add_argument("--model_dir", default="./train_log",
                   help="Directory containing flownet.pkl")
    p.add_argument("--output_dir", default="./rife_ov", help="Directory to save the OpenVINO model")
    p.add_argument("--height", type=int, default=768, help="Baked model height (multiple of 128)")
    p.add_argument("--width", type=int, default=1280, help="Baked model width (multiple of 128)")
    p.add_argument("--timestep", type=float, default=0.5, help="Hardcoded interpolation timestep")
    p.add_argument("--scale", type=float, default=1.0, help="Hardcoded flow scale factor")
    return p.parse_args()


class RIFEExportWrapper(nn.Module):
    def __init__(self, rife_model, timestep=0.5, scale=1.0):
        super().__init__()
        self.model = rife_model
        self.timestep = timestep
        self.scale = scale

    def forward(self, x):
        img0 = x[:, :3, :, :]
        img1 = x[:, 3:6, :, :]
        return self.model.inference(img0, img1, timestep=self.timestep, scale=self.scale)


def load_pytorch_model(model_dir: str):
    script_dir = Path(__file__).parent.resolve()
    practical_rife_dir = script_dir / "Practical-RIFE"
    for p in (script_dir, practical_rife_dir):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from train_log.RIFE_HDv3 import Model

    model = Model()
    if not hasattr(model, "version"):
        model.version = 0
    model.load_model(model_dir, rank=-1)
    
    model.eval()
    model.flownet.to(torch.device("cpu")) 
    return model


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_path = output_dir / "rife_temp.onnx"
    xml_path = output_dir / "rife.xml"

    # 1. Load PyTorch Model
    print(f"Loading PyTorch model from {args.model_dir}...")
    pytorch_model = load_pytorch_model(args.model_dir)

    wrapper = RIFEExportWrapper(pytorch_model, timestep=args.timestep, scale=args.scale)
    wrapper.eval()

    dummy_input = torch.randn(1, 6, args.height, args.width, dtype=torch.float32)

    # 2. Export to ONNX Opset 16 (CRITICAL FIX FOR GRID_SAMPLE)
    print("Exporting to ONNX (Opset 16) to preserve grid_sample layers...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=16,          # This opset fully supports grid_sample
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )

    # 3. Convert ONNX to OpenVINO
    print("Converting ONNX to OpenVINO IR...")
    core = ov.Core()
    ov_model = core.read_model(str(onnx_path))
    
    print(f"Saving OpenVINO model to {xml_path}...")
    ov.save_model(ov_model, str(xml_path))

    # 4. Clean up the temp ONNX file
    if onnx_path.exists():
        os.remove(onnx_path)

    print("\nExport Complete! ✅")
    print("Run this new model using the updated OpenCV inference script.")


if __name__ == "__main__":
    main()