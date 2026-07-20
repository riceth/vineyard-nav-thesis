#!/usr/bin/env python3
"""Segmentation metrics for Phase A via confusion-matrix accumulation.

Contract: docs/PHASE_A_SPEC.md section 7.

Reports per-class IoU (background, foreground), mean IoU, and per-class
precision / recall / F1.

Accumulation semantics (IMPORTANT): a single K x K confusion matrix is
accumulated across every batch of a split; all metrics are derived from those
TOTAL counts at epoch end. Metrics are NOT computed per batch and averaged.
This matters because mean-of-per-batch-IoU != IoU-of-pooled-counts (IoU is a
ratio of sums, not a mean of ratios); the pooled-count value is the correct
split-level metric. The unit tests below feed size-1 batches and confirm the
accumulated result equals the analytic (single-shot) answer.

Confusion matrix convention: cm[gt, pred] (rows = ground truth, cols = prediction).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int = 2, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        if class_names is None:
            class_names = ["background", "foreground"][:num_classes]
            if len(class_names) != num_classes:
                class_names = [f"class_{i}" for i in range(num_classes)]
        if len(class_names) != num_classes:
            raise ValueError("class_names length must equal num_classes")
        self.class_names = class_names
        self.reset()

    def reset(self) -> None:
        # int64 CPU accumulator; counts can exceed int32 over a full split.
        self.confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate one batch.

        Args:
            preds:  logits [B, C, H, W] (argmax taken) or label map [B, H, W].
            target: label map [B, H, W] int64 in [0, num_classes).
        """
        if preds.dim() == target.dim() + 1:      # logits -> hard labels
            preds = preds.argmax(dim=1)
        if preds.shape != target.shape:
            raise ValueError(f"preds {tuple(preds.shape)} vs target {tuple(target.shape)}")

        p = preds.reshape(-1).to(torch.long)
        t = target.reshape(-1).to(torch.long)
        n = self.num_classes
        # cm[gt, pred] via linear index gt*n + pred.
        idx = t * n + p
        binc = torch.bincount(idx, minlength=n * n).reshape(n, n)
        self.confusion += binc.cpu()

    def compute(self) -> Dict:
        cm = self.confusion.double()
        tp = cm.diag()
        fp = cm.sum(dim=0) - tp        # predicted c but gt != c  (column sum - diag)
        fn = cm.sum(dim=1) - tp        # gt c but predicted != c  (row sum - diag)

        iou = tp / (tp + fp + fn)                 # NaN where a class is wholly absent
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2.0 * tp) / (2.0 * tp + fp + fn)
        miou = torch.nanmean(iou)

        names = self.class_names
        return {
            "miou": miou.item(),
            "iou": {names[i]: iou[i].item() for i in range(self.num_classes)},
            "precision": {names[i]: precision[i].item() for i in range(self.num_classes)},
            "recall": {names[i]: recall[i].item() for i in range(self.num_classes)},
            "f1": {names[i]: f1[i].item() for i in range(self.num_classes)},
            "confusion_matrix": self.confusion.tolist(),
        }


# --- Unit tests (PHASE_A_SPEC section 7) --------------------------------------
def _test_known_iou_accumulated_over_size1_batches():
    """4 pixels with known confusion: [gt,pred] = (1,1),(1,0),(0,1),(0,0).

    Foreground: TP=1, FP=1, FN=1 -> IoU=1/3, precision=recall=F1=1/2.
    Background: TP=1, FP=1, FN=1 -> IoU=1/3. mIoU = 1/3.
    Fed as four separate size-1 batches; the accumulated metric must equal the
    analytic answer (a per-batch average would be degenerate here).
    """
    pairs = [(1, 1), (1, 0), (0, 1), (0, 0)]
    m = SegmentationMetrics(num_classes=2)
    for gt, pred in pairs:
        m.update(torch.tensor([[[pred]]]), torch.tensor([[[gt]]]))  # [1,1,1] each
    r = m.compute()

    assert abs(r["miou"] - 1 / 3) < 1e-9, r["miou"]
    assert abs(r["iou"]["foreground"] - 1 / 3) < 1e-9, r["iou"]
    assert abs(r["iou"]["background"] - 1 / 3) < 1e-9, r["iou"]
    assert abs(r["precision"]["foreground"] - 0.5) < 1e-9, r["precision"]
    assert abs(r["recall"]["foreground"] - 0.5) < 1e-9, r["recall"]
    assert abs(r["f1"]["foreground"] - 0.5) < 1e-9, r["f1"]
    return r["miou"]


def _test_accumulation_equals_single_shot():
    """Accumulating in chunks must equal one single-shot update (proves the
    metric is a function of pooled counts, order- and batching-invariant)."""
    torch.manual_seed(1)
    preds = torch.randint(0, 2, (10, 8, 8))
    target = torch.randint(0, 2, (10, 8, 8))

    chunked = SegmentationMetrics(num_classes=2)
    for i in range(10):                                    # size-1 batches
        chunked.update(preds[i:i + 1], target[i:i + 1])

    single = SegmentationMetrics(num_classes=2)
    single.update(preds, target)                           # one big batch

    rc, rs = chunked.compute(), single.compute()
    assert rc["confusion_matrix"] == rs["confusion_matrix"], "cm differs by batching"
    assert abs(rc["miou"] - rs["miou"]) < 1e-12, (rc["miou"], rs["miou"])
    return rc["miou"]


def _test_logits_input_argmax():
    """update() must accept raw logits [B,C,H,W] and argmax them."""
    target = torch.tensor([[[1, 0]]])                      # [1,1,2]
    logits = torch.zeros(1, 2, 1, 2)
    logits[0, 1, 0, 0] = 5.0                               # pixel0 -> class 1 (correct)
    logits[0, 0, 0, 1] = 5.0                               # pixel1 -> class 0 (correct)
    m = SegmentationMetrics(num_classes=2)
    m.update(logits, target)
    r = m.compute()
    assert abs(r["miou"] - 1.0) < 1e-12, r["miou"]         # perfect
    return r["miou"]


if __name__ == "__main__":
    a = _test_known_iou_accumulated_over_size1_batches()
    b = _test_accumulation_equals_single_shot()
    c = _test_logits_input_argmax()
    print(f"known-IoU (size-1 batches) mIoU: {a:.6f}  (expected 0.333333)")
    print(f"chunked == single-shot mIoU:     {b:.6f}  (batching-invariant)")
    print(f"logits-input perfect mIoU:       {c:.6f}  (expected 1.0)")
    print("metrics.py unit tests passed.")
