#!/usr/bin/env python3
"""SMP U-Net wrapper for Phase A (binary baseline).

Contract: docs/PHASE_A_SPEC.md section 5. Architecture is LOCKED (D022):
ResNet-34 encoder pretrained on ImageNet, SMP default U-Net decoder, 2-channel
raw-logit output (D007: 2-channel softmax head, not 1-channel sigmoid).

The wrapper is a thin nn.Module around `smp.Unet` so training/eval code depends
on a stable local interface rather than the library signature directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# LOCKED architecture defaults (PHASE_A_SPEC 5.1 / D022).
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"
IN_CHANNELS = 3
NUM_CLASSES = 2                 # 2-channel softmax head (D007)


class UNetBinary(nn.Module):
    """SMP U-Net, ResNet-34/ImageNet encoder, 2-channel raw-logit output.

    Forward: [B, 3, H, W] -> [B, 2, H, W] raw logits (spatial dims preserved).
    """

    def __init__(
        self,
        encoder_name: str = ENCODER_NAME,
        encoder_weights: str | None = ENCODER_WEIGHTS,
        in_channels: int = IN_CHANNELS,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.num_classes = num_classes
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,   # None -> random init (offline/smoke use)
            in_channels=in_channels,
            classes=num_classes,
            activation=None,                   # raw logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] -> [B, num_classes, H, W] raw logits
        return self.net(x)

    def param_counts(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def build_model(config: dict | None = None) -> UNetBinary:
    """Construct the model from a config dict's `model` block, falling back to
    the LOCKED defaults. Keeps train.py decoupled from the constructor."""
    m = (config or {}).get("model", {}) if config else {}
    return UNetBinary(
        encoder_name=m.get("encoder", ENCODER_NAME),
        encoder_weights=m.get("encoder_weights", ENCODER_WEIGHTS),
        in_channels=m.get("in_channels", IN_CHANNELS),
        num_classes=m.get("num_classes", NUM_CLASSES),
    )


if __name__ == "__main__":
    # Smoke test (PHASE_A_SPEC 5.3): random (2,3,640,640) -> assert (2,2,640,640),
    # print parameter counts. Uses the LOCKED ImageNet weights by default.
    import argparse

    ap = argparse.ArgumentParser(description="UNetBinary smoke test.")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="Use random init (skip ImageNet download).")
    args = ap.parse_args()

    weights = None if args.no_pretrained else ENCODER_WEIGHTS
    model = UNetBinary(encoder_weights=weights).eval()

    x = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 2, 640, 640), f"unexpected output shape: {tuple(y.shape)}"
    total, trainable = model.param_counts()
    print(f"encoder={model.encoder_name} weights={weights}")
    print(f"input  {tuple(x.shape)}  ->  output {tuple(y.shape)}  (raw logits)")
    print(f"total params:     {total:,} ({total / 1e6:.2f}M)")
    print(f"trainable params: {trainable:,} ({trainable / 1e6:.2f}M)")
    print("Smoke test passed.")
