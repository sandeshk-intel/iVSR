#!/usr/bin/env python3
"""Export the VideoSeal detector (ConvnextExtractor) to OpenVINO IR.

The embedder IR (export_videoseal_openvino.py) bakes the watermark payload into
a model that runs inside FFmpeg via the iVSR plugin (model_type=6).

This script exports the *extractor* side: ConvnextExtractor, which recovers
the payload logits from a watermarked frame.  The exported model is used by
extract_videoseal_ov.py and by the Streamlit Extract tab as an alternative to
the PyTorch runtime.

Pipeline inside model.detect():
  Input [B, 3, H, W] → bilinear resize to [B, 3, 256, 256] → ConvnextExtractor
    → output [B, 257] raw logits (index 0 = presence bit, 1–256 = payload bits)

The exported model takes a **fixed [1, 3, 256, 256] NCHW float32** input in
[0.0, 1.0].  The caller is responsible for:
  1. Converting the raw frame from BGR to RGB
  2. Resizing to 256×256 (bilinear with antialiasing matches PyTorch detect())
  3. Normalising to [0, 1]
  4. Transposing to NCHW

Usage:
    python3 videoseal/export_videoseal_detector_openvino.py
    # → produces videoseal_detector.xml + videoseal_detector.bin

Optional:
    --output  path/to/filename.xml   (default: videoseal_detector.xml)
"""

import argparse
import torch
import torch.nn as nn
import openvino as ov
import videoseal


class VideoSealDetectorWrapper(nn.Module):
    """Thin wrapper around ConvnextExtractor for clean TorchScript tracing.

    The raw detector.forward() accepts a pre-resized [B, 3, 256, 256] tensor
    and returns [B, 257] logits directly — no dynamic control flow needed.
    """

    def __init__(self, detector: nn.Module):
        super().__init__()
        self.detector = detector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, 3, 256, 256] float32 in [0, 1]
        return self.detector(x)  # [1, 257] raw logits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the VideoSeal detector to OpenVINO IR."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="videoseal_detector.xml",
        help="Output .xml path (default: videoseal_detector.xml).",
    )
    args = parser.parse_args()

    # ── 1. Load full VideoSeal model and extract the detector sub-network ─────
    print("[*] Loading VideoSeal model weights...")
    model = videoseal.load("videoseal")
    model.eval()

    detector = model.detector  # ConvnextExtractor
    detector.eval()

    wrapper = VideoSealDetectorWrapper(detector)
    wrapper.eval()

    # Fixed input shape: [1, 3, 256, 256] — detector always runs at 256×256.
    dummy_input = torch.zeros(1, 3, 256, 256, dtype=torch.float32)

    # ── 2. Trace to TorchScript ───────────────────────────────────────────────
    print("[*] Tracing detector to TorchScript...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy_input, strict=False)

    # ── 3. Convert to OpenVINO IR ─────────────────────────────────────────────
    print("[*] Converting TorchScript to OpenVINO IR...")
    ov_model = ov.convert_model(traced, example_input=dummy_input)

    # Fix input shape to [1, 3, 256, 256] (static) so OpenVINO can fully
    # optimise the ConvNeXt layers.
    ov_model.reshape({0: [1, 3, 256, 256]})

    ov_model.inputs[0].get_tensor().set_names({"frame"})
    ov_model.outputs[0].get_tensor().set_names({"logits"})

    # ── 4. Save ───────────────────────────────────────────────────────────────
    print(f"[*] Saving model to: {args.output}")
    ov.save_model(ov_model, args.output)

    # ── 5. Quick sanity check ─────────────────────────────────────────────────
    print("[*] Running sanity check (PyTorch vs OpenVINO outputs)...")
    test_input = torch.rand(1, 3, 256, 256, dtype=torch.float32)

    with torch.no_grad():
        pt_out = wrapper(test_input).numpy()

    core = ov.Core()
    compiled = core.compile_model(args.output, "CPU")
    ov_out = list(compiled({0: test_input.numpy()}).values())[0]

    max_diff = abs(pt_out - ov_out).max()
    print(f"    Max abs difference PyTorch vs OpenVINO: {max_diff:.6f}")
    if max_diff < 1e-3:
        print("[+] Sanity check PASSED")
    else:
        print("[!] Sanity check WARNING: difference is larger than expected")

    xml_path = args.output
    bin_path = xml_path.replace(".xml", ".bin")
    import os
    xml_mb = os.path.getsize(xml_path) / 1024 / 1024
    bin_mb = os.path.getsize(bin_path) / 1024 / 1024
    print(f"\n[+] SUCCESS")
    print(f"    {xml_path}  ({xml_mb:.2f} MB)")
    print(f"    {bin_path}  ({bin_mb:.2f} MB)")
    print()
    print("Use with extract_videoseal_ov.py:")
    print(f"    python3 videoseal/extract_videoseal_ov.py \\")
    print(f"        --model {xml_path} \\")
    print(f"        --video <watermarked.mp4>")


if __name__ == "__main__":
    main()
