#!/usr/bin/env python3
"""
VideoSeal to OpenVINO IR Converter

Execution Instructions:
-----------------------
1. Environment Setup:
   Ensure PyTorch, OpenVINO, and Meta's official VideoSeal library are installed.
   $ pip install torch openvino
   $ pip install git+https://github.com/facebookresearch/VideoSeal.git

2. Run the Export:
   You can customize the baked-in watermark text and the target video resolution.
   Because OpenVINO static shapes are highly optimized, the exported resolution 
   MUST match your FFmpeg input stream exactly.

   Example (Default 720p with "IVSR" watermark):
   $ python3 export_videoseal_ov.py

   Example (Custom 1080p with custom watermark):
   $ python3 export_videoseal_ov.py --text "CONFIDENTIAL" --height 1080 --width 1920

3. FFmpeg Integration:
   Mount the resulting .xml file directly into your FFmpeg dnn_processing filter.
   The model expects NCHW format natively.
"""
import argparse
import torch
import torch.nn as nn
import openvino as ov
import videoseal

def text_to_payload(text: str, num_bits: int = 256) -> torch.Tensor:
    binary_string = ''.join(format(ord(char), '08b') for char in text)
    binary_string = binary_string.ljust(num_bits, '0')[:num_bits]
    return torch.tensor([[float(bit) for bit in binary_string]], dtype=torch.float32)

class VideoSealFinalWrapper(nn.Module):
    def __init__(self, full_model, text: str, num_bits: int = 256):
        super().__init__()
        self.videoseal_model = full_model
        self.register_buffer('baked_payload', text_to_payload(text, num_bits))

    def forward(self, video_frame):
        # 1. FFmpeg gives us Float32 (0.0 to 255.0). Scale it to 0.0-1.0.
        x = video_frame / 255.0
        
        # 2. Embed the watermark
        out = self.videoseal_model.embed(x, self.baked_payload)
        
        # 3. Strip dictionaries
        if isinstance(out, dict):
            out = out.get('imgs', list(out.values())[0])
        elif isinstance(out, (tuple, list)):
            out = out[0]

        # 4. Safety Clamp to prevent pixel explosions
        out = torch.clamp(out, 0.0, 1.0)
        
        # 5. Scale back to 0-255.0. FFmpeg handles the final float->int conversion.
        return out * 255.0

def main():
    parser = argparse.ArgumentParser(description="Export VideoSeal with targeted NHWC Layout Translation.")
    parser.add_argument("--text", type=str, default="IVSR")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()
    
    print(f"[*] Loading PyTorch Model...")
    model = videoseal.load("videoseal")
    model.eval()
    
    wrapper = VideoSealFinalWrapper(model, args.text)
    wrapper.eval()

    pytorch_shape = [1, 3, args.height, args.width]
    dummy_input = torch.zeros(pytorch_shape, dtype=torch.float32)

    # ==========================================
    # STEP 1: TORCHSCRIPT TRACING
    # ==========================================
    print("[*] Tracing model to TorchScript...")
    with torch.no_grad():
        traced_wrapper = torch.jit.trace(wrapper, dummy_input, strict=False)

    # ==========================================
    # STEP 2: DIRECT OPENVINO CONVERSION
    # ==========================================
    print(f"[*] Converting TorchScript directly to OpenVINO IR...")
    ov_model = ov.convert_model(traced_wrapper, example_input=dummy_input)

    # ==========================================
    # STEP 3: RESHAPE TO NCHW (no layout translation)
    # ==========================================
    # The C backend (pack_input_videoseal) packs rgb24 → NCHW float32 [1,3,H,W]
    # directly, so the model must accept NCHW input.  Do NOT insert an
    # NHWC↔NCHW PrePostProcessor transpose here — it would cause a format
    # mismatch that scrambles individual watermark bits (e.g. 'S' → '[').
    print(f"[*] Reshaping model to NCHW [1, 3, {args.height}, {args.width}]...")
    ov_model.reshape({0: [1, 3, args.height, args.width]})
    
    ov_model.inputs[0].get_tensor().set_names({"video_frame"})
    ov_model.outputs[0].get_tensor().set_names({"clean_tensor"})

    output_filename = f"videoseal_baked_{args.height}p.xml"
    ov.save_model(ov_model, output_filename)
        
    print(f"\n[+] SUCCESS! Model saved with precise layout translation to: {output_filename}")

if __name__ == "__main__":
    main()