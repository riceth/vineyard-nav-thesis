#!/usr/bin/env python3
"""Combined CE + Dice loss for Phase A (binary U-Net).

Contract: docs/PHASE_A_SPEC.md section 6, decision D009.

    loss = ce_weight * CE + dice_weight * Dice     (defaults 0.5 / 0.5)

  * CE   : nn.CrossEntropyLoss over the 2-channel logits, no class weighting.
  * Dice : multiclass soft Dice on softmax probabilities, averaged over classes
           with EQUAL class weighting.

On "generalised": PHASE_A_SPEC 6 says "generalised soft Dice across both
classes." This is implemented as the multiclass generalisation of soft Dice with
equal per-class weighting (mean of per-class Dice), NOT the inverse-volume-
weighted Generalised Dice Loss of Sudre et al. (2017). Rationale: the config
exposes only ce_weight/dice_weight (no Dice class-weighting term) and D009 sets
"equal weighting" as the design principle; the foreground Dice term already
addresses class imbalance without volume-based reweighting. `eps` smooths both
numerator and denominator so empty classes contribute a perfect (1.0) score
rather than a 0/0 NaN.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        num_classes: int = 2,
        eps: float = 1.0e-6,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.num_classes = num_classes
        self.eps = eps

    def ce_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Mathematically identical to nn.CrossEntropyLoss(reduction="mean") with no
        # class weighting (D009 / PHASE_A_SPEC 6), but computed via one-hot + log_softmax
        # (elementwise mul + reductions) instead of nll_loss2d. The CUDA nll_loss2d
        # kernel has no deterministic implementation in PyTorch 2.11 and was the sole
        # source of same-seed run-to-run divergence; this formulation is fully
        # deterministic (rule 7 / D016 / PHASE_A_SPEC 8.3).
        log_probs = F.log_softmax(logits, dim=1)                 # [B, C, H, W]
        target_1h = F.one_hot(target, self.num_classes)          # [B, H, W, C]
        target_1h = target_1h.permute(0, 3, 1, 2).to(log_probs.dtype)
        return -(target_1h * log_probs).sum(dim=1).mean()

    def dice_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B, C, H, W] raw; target: [B, H, W] int64 in [0, C)
        probs = F.softmax(logits, dim=1)
        target_1h = F.one_hot(target, self.num_classes)          # [B, H, W, C]
        target_1h = target_1h.permute(0, 3, 1, 2).to(probs.dtype)  # [B, C, H, W]

        dims = (0, 2, 3)                                          # sum over batch + pixels
        intersection = (probs * target_1h).sum(dims)             # [C]
        cardinality = probs.sum(dims) + target_1h.sum(dims)      # [C]
        dice_per_class = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice_per_class.mean()                       # equal class weighting

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(logits, target)
        dice = self.dice_loss(logits, target)
        return self.ce_weight * ce + self.dice_weight * dice


# --- Unit tests (PHASE_A_SPEC section 6) --------------------------------------
def _test_perfect_prediction_near_zero():
    torch.manual_seed(0)
    target = torch.randint(0, 2, (2, 32, 32))
    # Confident correct logits: large positive on the true class channel.
    logits = F.one_hot(target, 2).permute(0, 3, 1, 2).float() * 30.0
    loss = CombinedLoss()(logits, target)
    assert loss.item() < 1e-3, f"perfect prediction loss not ~0: {loss.item()}"
    return loss.item()


def _test_uniform_prediction_positive():
    target = torch.randint(0, 2, (2, 32, 32))
    logits = torch.zeros(2, 2, 32, 32)          # uniform -> softmax 0.5 / 0.5
    loss = CombinedLoss()(logits, target)
    # CE of uniform 2-class = ln(2) ~= 0.693; total must be clearly positive.
    assert loss.item() > 0.3, f"uniform prediction loss not > 0: {loss.item()}"
    return loss.item()


def _test_empty_foreground_is_stable():
    # All-background target with confident background prediction -> ~0 loss,
    # and the empty foreground class must not produce a NaN.
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    logits = torch.zeros(1, 2, 16, 16)
    logits[:, 0] = 30.0
    loss = CombinedLoss()(logits, target)
    assert torch.isfinite(loss), "loss is NaN/Inf on empty-foreground input"
    assert loss.item() < 1e-3, f"empty-fg loss not ~0: {loss.item()}"
    return loss.item()


def _test_ce_matches_nn_crossentropy():
    # The deterministic one-hot CE must match nn.CrossEntropyLoss(reduction="mean").
    torch.manual_seed(3)
    logits = torch.randn(4, 2, 16, 16)
    target = torch.randint(0, 2, (4, 16, 16))
    ours = CombinedLoss().ce_loss(logits, target)
    ref = nn.CrossEntropyLoss()(logits, target)
    assert torch.allclose(ours, ref, atol=1e-6), (ours.item(), ref.item())
    return (ours.item(), ref.item())


if __name__ == "__main__":
    ce_ours, ce_ref = _test_ce_matches_nn_crossentropy()
    print(f"CE one-hot={ce_ours:.6f} vs nn.CrossEntropyLoss={ce_ref:.6f}  (match)")
    perfect = _test_perfect_prediction_near_zero()
    uniform = _test_uniform_prediction_positive()
    empty = _test_empty_foreground_is_stable()
    print(f"perfect prediction loss: {perfect:.3e}  (expected ~0)")
    print(f"uniform prediction loss: {uniform:.4f}  (expected > 0, ~0.35 + dice)")
    print(f"empty-foreground  loss: {empty:.3e}  (finite, ~0)")
    print("losses.py unit tests passed.")
