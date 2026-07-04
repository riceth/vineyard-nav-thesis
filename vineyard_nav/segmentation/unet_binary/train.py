#!/usr/bin/env python3
"""Phase A U-Net binary training loop.

Contract: docs/PHASE_A_SPEC.md section 8. Decisions: D004 (AMP, persistent
GradScaler), D009 (loss), D011/D022 (optim, pretrained encoder), D015 (CSV+TB),
D016 (reproducibility). Test-set is never touched here (working rule 5).

Run directory layout (PHASE_A_SPEC section 2):
  results/runs/<experiment>_<timestamp>/
    checkpoints/{best.pt,final.pt}  tensorboard/  metrics.csv
    config_snapshot.yaml  git_commit.txt  predictions/

Smoke run (PHASE_A_SPEC 8.3): `python -m segmentation.unet_binary.train --smoke`
runs 2 epochs on 50 train + 10 val samples and prints AMP diagnostics
(backward-pass timing, per-iteration finite-loss checks, GradScaler scale).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import subprocess
import time
from datetime import datetime
from typing import Dict, Optional

# Must be set BEFORE the first cuBLAS call (i.e. before any CUDA context init) for
# torch.use_deterministic_algorithms to make cuBLAS GEMMs deterministic. Setting it
# later (e.g. inside set_seed, after torch.cuda.*) is too late. See set_seed / D016.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

# The devcontainer's /dev/shm is only 64 MB (Docker default), too small for the
# default 'file_descriptor' strategy to pass 640x640 tensors between DataLoader
# workers -> "unable to allocate shared memory". 'file_system' avoids /dev/shm.
torch.multiprocessing.set_sharing_strategy("file_system")
from torch.utils.tensorboard import SummaryWriter

from .dataset import (
    SemanticBLTBinaryDataset,
    train_transform,
    eval_transform,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from .losses import CombinedLoss
from .metrics import SegmentationMetrics
from .model import build_model

CLASS_NAMES = ["background", "foreground"]


# --- Reproducibility (D016) ---------------------------------------------------
def set_seed(seed: int, deterministic: bool, benchmark: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    if deterministic:
        # cudnn.deterministic alone does not cover non-cudnn CUDA reductions
        # (e.g. cuBLAS GEMM workspace, atomicAdd backward). Force deterministic
        # kernels for every op that has one (rule 7 / D016 / PHASE_A_SPEC 8.3).
        # CUBLAS_WORKSPACE_CONFIG is set at module import (before CUDA init).
        # warn_only=True: the CE loss is computed deterministically in losses.py,
        # so the remaining warn-only op (nll_loss2d, unused by our loss) does not
        # affect results; keeping warn_only avoids crashing on any incidental op.
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # No git repo yet: reproducibility rule 7 / D016 not fully satisfiable.
        return "NO_GIT_REPO"


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


# --- Data ---------------------------------------------------------------------
def build_loaders(config: dict, smoke: bool):
    d = config["data"]
    train_ds = SemanticBLTBinaryDataset(d["root"], "train", d["split_manifest"],
                                        transform=train_transform(seed=config["seed"]))
    val_ds = SemanticBLTBinaryDataset(d["root"], "valid", d["split_manifest"],
                                      transform=eval_transform())

    num_workers = d["num_workers"]
    batch_size = config["train"]["batch_size"]
    if smoke:
        # Deterministic 50-train / 10-val subset (PHASE_A_SPEC 8.3).
        train_ds = Subset(train_ds, list(range(min(50, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(10, len(val_ds)))))
        num_workers = 0   # 64 MB /dev/shm can't back worker IPC in this container

    g = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=False, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


# --- Checkpoint ---------------------------------------------------------------
def save_checkpoint(path: str, model, optimizer, scheduler, scaler, epoch,
                    val_miou, config, git_commit) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "val_miou": val_miou,
        "config": config,
        "git_commit": git_commit,
        "rng_state": rng_state(),           # for full resumability (PHASE_A_SPEC 8.3)
    }, path)


def denormalize(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=img.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=img.device).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)


@torch.no_grad()
def log_predictions(writer, model, batch, epoch, device, k: int) -> None:
    model.eval()
    images = batch["image"][:k].to(device)
    masks = batch["mask"][:k]
    logits = model(images)
    preds = logits.argmax(1).cpu()
    for i in range(images.shape[0]):
        rgb = denormalize(images[i]).cpu()
        gt = masks[i].unsqueeze(0).float()
        pr = preds[i].unsqueeze(0).float()
        writer.add_image(f"pred/{i}/image", rgb, epoch)
        writer.add_image(f"pred/{i}/gt", gt, epoch)
        writer.add_image(f"pred/{i}/pred", pr, epoch)


# --- Train / validate ---------------------------------------------------------
def train_one_epoch(model, loader, loss_fn, optimizer, scaler, scheduler, device,
                    config) -> Dict[str, float]:
    model.train()
    accum = max(1, config["train"].get("grad_accumulation_steps", 1))
    amp = config["train"]["amp"] and device.type == "cuda"

    running, n_batches = 0.0, 0
    nonfinite = 0
    optimizer.zero_grad(set_to_none=True)

    for it, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(images)
            loss = loss_fn(logits, masks)
        loss_value = loss.detach()
        if not torch.isfinite(loss_value):
            nonfinite += 1

        # Grad accumulation (D004 / PHASE_A_SPEC 8.2): scale by 1/accum, step every N.
        scaler.scale(loss / accum).backward()
        if (it + 1) % accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running += loss_value.item()
        n_batches += 1

    scheduler.step()
    if nonfinite:
        print(f"  [WARN] {nonfinite} non-finite training losses this epoch")
    return {"train_loss": running / max(1, n_batches), "nonfinite": nonfinite}


@torch.no_grad()
def validate(model, loader, loss_fn, device) -> Dict:
    model.eval()
    metrics = SegmentationMetrics(num_classes=2, class_names=CLASS_NAMES)
    amp = loader is not None and device.type == "cuda"
    running, n = 0.0, 0
    last_batch = None
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(images)
            loss = loss_fn(logits, masks)
        running += loss.item()
        n += 1
        metrics.update(logits.float(), masks)
        last_batch = batch
    m = metrics.compute()
    return {
        "val_loss": running / max(1, n),
        "val_miou": m["miou"],
        "val_iou_foreground": m["iou"]["foreground"],
        "val_iou_background": m["iou"]["background"],
        "val_precision": m["precision"]["foreground"],
        "val_recall": m["recall"]["foreground"],
        "val_f1": m["f1"]["foreground"],
        "_last_batch": last_batch,
    }


CSV_COLUMNS = ["epoch", "train_loss", "val_loss", "val_miou", "val_iou_foreground",
               "val_iou_background", "val_precision", "val_recall", "val_f1", "lr"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase A U-Net binary training.")
    ap.add_argument("--config",
                    default="/workspaces/dissertation/vineyard_nav/configs/phase_a_unet_binary.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="2-epoch run on 50 train / 10 val subset with AMP diagnostics.")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"], config["reproducibility"]["deterministic"],
             config["reproducibility"]["benchmark"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 2 if args.smoke else config["train"]["epochs"]
    patience = config["train"]["early_stopping"]["patience"]

    # Run directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = config["experiment_name"] + ("_smoke" if args.smoke else "")
    run_dir = os.path.join("/workspaces/dissertation/vineyard_nav/results/runs",
                           f"{name}_{ts}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "predictions"), exist_ok=True)
    shutil.copy(args.config, os.path.join(run_dir, "config_snapshot.yaml"))
    git_commit = get_git_commit()
    with open(os.path.join(run_dir, "git_commit.txt"), "w") as f:
        f.write(git_commit + "\n")
    if git_commit == "NO_GIT_REPO":
        print("[WARN] No git repository — checkpoints record 'NO_GIT_REPO'. "
              "Run `git init` before the full 60-epoch run (D016 / rule 7).")

    writer = SummaryWriter(os.path.join(run_dir, "tensorboard")) \
        if config["logging"]["tensorboard"] else None
    csv_path = os.path.join(run_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(CSV_COLUMNS)

    train_loader, val_loader = build_loaders(config, args.smoke)
    print(f"Device: {device} | train batches: {len(train_loader)} "
          f"| val batches: {len(val_loader)} | epochs: {epochs}")

    model = build_model(config).to(device)
    loss_fn = CombinedLoss(ce_weight=config["loss"]["ce_weight"],
                           dice_weight=config["loss"]["dice_weight"],
                           num_classes=config["model"]["num_classes"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["optim"]["lr"],
                                 weight_decay=config["optim"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["schedule"]["t_max"], eta_min=config["schedule"]["eta_min"])
    # GradScaler instantiated ONCE, persisted across all epochs (D004).
    scaler = torch.amp.GradScaler("cuda", enabled=config["train"]["amp"] and device.type == "cuda")

    if args.smoke:
        run_smoke_diagnostics(model, train_loader, loss_fn, optimizer, scaler, device, config)

    best_miou, best_epoch, epochs_no_improve = -1.0, -1, 0
    for epoch in range(epochs):
        t_epoch = time.perf_counter()
        tr = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler,
                             scheduler, device, config)
        val = validate(model, val_loader, loss_fn, device)
        lr = optimizer.param_groups[0]["lr"]
        dt = time.perf_counter() - t_epoch

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, tr["train_loss"], val["val_loss"],
                val["val_miou"], val["val_iou_foreground"], val["val_iou_background"],
                val["val_precision"], val["val_recall"], val["val_f1"], lr])
        if writer:
            writer.add_scalar("loss/train", tr["train_loss"], epoch)
            writer.add_scalar("loss/val", val["val_loss"], epoch)
            writer.add_scalar("metric/val_miou", val["val_miou"], epoch)
            writer.add_scalar("metric/val_iou_foreground", val["val_iou_foreground"], epoch)
            writer.add_scalar("lr", lr, epoch)
            n_every = config["logging"]["log_predictions_every_n_epochs"]
            if epoch % n_every == 0 or epoch == epochs - 1:
                log_predictions(writer, model, val["_last_batch"], epoch, device,
                                config["logging"]["prediction_subset_size"])

        print(f"epoch {epoch:3d} | train {tr['train_loss']:.4f} | val {val['val_loss']:.4f} "
              f"| mIoU {val['val_miou']:.4f} (fg {val['val_iou_foreground']:.4f}) "
              f"| lr {lr:.2e} | {dt:.1f}s")

        save_checkpoint(os.path.join(ckpt_dir, "final.pt"), model, optimizer,
                        scheduler, scaler, epoch, val["val_miou"], config, git_commit)
        if val["val_miou"] > best_miou:
            best_miou, best_epoch, epochs_no_improve = val["val_miou"], epoch, 0
            save_checkpoint(os.path.join(ckpt_dir, "best.pt"), model, optimizer,
                            scheduler, scaler, epoch, val["val_miou"], config, git_commit)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no val_miou improvement "
                      f"for {patience} epochs)")
                break

    if writer:
        writer.close()
    print(f"\nDone. Best val mIoU {best_miou:.4f} @ epoch {best_epoch}. Run dir: {run_dir}")
    return run_dir


def run_smoke_diagnostics(model, loader, loss_fn, optimizer, scaler, device, config) -> None:
    """Explicit AMP diagnostics on sm_120 before committing to a long run:
    per-iteration forward/backward timing, finite-loss check, autocast dtype,
    and GradScaler scale (a dropping scale => inf/nan grads => silent skips)."""
    print("\n=== AMP smoke diagnostics (first 5 iters) ===")
    amp = config["train"]["amp"] and device.type == "cuda"
    model.train()
    it = 0
    for batch in loader:
        if it >= 5:
            break
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(images)
            loss = loss_fn(logits, masks)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_fwd = time.perf_counter() - t0

        t1 = time.perf_counter()
        scaler.scale(loss).backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_bwd = time.perf_counter() - t1

        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()

        finite = torch.isfinite(loss).item()
        skipped = scale_after < scale_before  # scaler reduced scale => grads were inf/nan
        print(f"  iter {it}: loss={loss.item():.4f} finite={finite} "
              f"autocast_dtype={logits.dtype} fwd={t_fwd*1000:.1f}ms "
              f"bwd={t_bwd*1000:.1f}ms scale {scale_before:.0f}->{scale_after:.0f}"
              f"{' [STEP SKIPPED: inf/nan grads]' if skipped else ''}")
        it += 1
    # Reset so real training starts from a clean, seeded state.
    optimizer.zero_grad(set_to_none=True)
    print("=== end diagnostics ===\n")


if __name__ == "__main__":
    main()
