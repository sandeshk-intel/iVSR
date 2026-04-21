#!/usr/bin/env python3
"""
SPAN to OpenVINO IR Converter (Direct PyTorch Export)

Execution Instructions:
-----------------------
1. Clone the official SPAN repository in the same directory as this script:
   $ git clone https://github.com/hongyuanyu/span.git

2. MANUAL DOWNLOAD REQUIRED (Offline Setup):
   - Visit the official repo: https://github.com/hongyuanyu/SPAN
   - Download your desired weights (e.g., spanx4_ch48.pth) offline.
   - Place the .pth file in the same directory as this script.

3. Run the conversion script:
   $ python span_to_openvino.py --weights spanx2_ch48.pth --scale 2 --channels 48
"""

import sys
import argparse
from pathlib import Path
import torch
import types
from unittest.mock import MagicMock

# ==============================================================================
# MONKEY PATCHES FOR SPAN REPOSITORY (BasicSR bypasses)
import torchvision.transforms.functional as modern_tv_f
sys.modules['torchvision.transforms.functional_tensor'] = modern_tv_f
sys.modules['basicsr.archs.vgg_arch'] = MagicMock()
mock_version = types.ModuleType('basicsr.version')
mock_version.__version__ = '1.0.0'
mock_version.__gitsha__ = 'unknown'
sys.modules['basicsr.version'] = mock_version
# ==============================================================================

try:
    import openvino as ov
except ImportError:
    print("OpenVINO missing. Please install via: pip install openvino")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert SPAN directly to OpenVINO IR.")
    parser.add_argument("--weights", type=str, default="spanx4_ch48.pth", help="Path to SPAN weights.")
    parser.add_argument("--scale", type=int, default=4, help="Upscaling factor (e.g., 2 or 4).")
    parser.add_argument("--channels", type=int, default=48, help="Number of feature channels (48 or 52).")
    return parser.parse_args()

def main():
    args = parse_args()
    weights_path = Path(args.weights)
    
    # 1. Automatic Directory Creation
    export_dir = weights_path.parent / f"{weights_path.stem}_ir"
    export_dir.mkdir(parents=True, exist_ok=True)
    xml_path = export_dir / f"{weights_path.stem}.xml"

    # 2. Offline Weight Verification
    if not weights_path.exists():
        print(f"[!] Error: Model weights not found at '{weights_path}'.")
        print("[!] Please manually download the .pth file and place it next to this script.")
        sys.exit(1)

    # 3. Dynamic Path Injection
    script_dir = Path(__file__).parent.resolve()
    repo_root = script_dir / "span"
    
    if not repo_root.exists() or not repo_root.is_dir():
        print(f"[!] Error: Could not find the 'span' repository at {repo_root}")
        print("[!] Please run: git clone https://github.com/hongyuanyu/span.git")
        sys.exit(1)
        
    sys.path.insert(0, str(repo_root))

    try:
        from basicsr.archs.span_arch import SPAN 
    except ImportError as e:
        print(f"[!] Error: Could not import SPAN from basicsr.archs.span_arch. ({e})")
        sys.exit(1)

    print(f"[*] Initializing SPAN model (Scale: {args.scale}x, Channels: {args.channels})...")
    model = SPAN(num_in_ch=3, num_out_ch=3, feature_channels=args.channels, upscale=args.scale)
    
    print(f"[*] Loading weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location="cpu")
    
    if "params" in state_dict:
        state_dict = state_dict["params"]
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 4. Direct PyTorch -> OpenVINO Export (Bypass ONNX entirely)
    print(f"[*] Converting PyTorch model directly to OpenVINO IR...")
    dummy_input = torch.randn(1, 3, 360, 640) 
    
    # Using OpenVINO's native PyTorch frontend
    ov_model = ov.convert_model(model, example_input=dummy_input)
    
    # Make the dimensions dynamic so FFmpeg can pass any resolution
    # -1 represents a dynamic axis: [Batch, Channels, Height, Width]
    ov_model.reshape([1, 3, -1, -1])

    # 5. Save Model
    print(f"[*] Saving OpenVINO IR in {export_dir}...")
    ov.save_model(ov_model, str(xml_path))
    
    print(f"[*] Success! IR files exported to: {export_dir}/")
    print(f"[*] FFmpeg model path: {xml_path}")

if __name__ == "__main__":
    main()