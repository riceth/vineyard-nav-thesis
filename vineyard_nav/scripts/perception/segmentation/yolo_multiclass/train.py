#!/usr/bin/env python3
"""Phase C YOLOv11-seg MULTICLASS training (PHASE_C_SPEC section 6).

Faithful copy of segmentation/yolo_binary/train.py — IDENTICAL training logic,
differing only in the default config path (Phase C multiclass). Keeping the code
identical (rather than importing across Phase B's committed module) guarantees the
B <-> C comparison isolates class structure, not training procedure (D021, D025),
while respecting "do not touch Phase B files". Decisions: D023 (YOLOv11-seg),
D013 (augmentation parity), D016 (reproducibility). No test-set access (rule 5).

Smoke run: `python -m scripts.perception.segmentation.yolo_multiclass.train --smoke` runs 2 epochs on
a 10% subsample to check no-OOM + healthy loss before the full 100-epoch run.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

# 64 MB /dev/shm in this container cannot back worker IPC; file_system avoids it
# (workers is also set to 0 in the config, but this is a harmless safety net).
torch.multiprocessing.set_sharing_strategy("file_system")

from ultralytics import YOLO

REPO = Path("/workspaces/dissertation/vineyard_nav")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "NO_GIT_REPO"


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase C YOLOv11-seg multiclass training.")
    ap.add_argument("--config", default=str(REPO / "configs/phase_c_yolo_multiclass_train.yaml"))
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs on a 10% subsample (no-OOM / loss sanity check).")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    train_args = {
        **cfg["train"],
        **cfg["augmentation"],
        "seed": cfg["seed"],
        "deterministic": cfg["reproducibility"]["deterministic"],
    }
    # ultralytics resolves 'data' relative to CWD; make it absolute for safety.
    data_path = (REPO / cfg["data"]) if not os.path.isabs(cfg["data"]) else Path(cfg["data"])
    train_args["data"] = str(data_path)
    # Make 'project' absolute too: a relative project gets nested under ultralytics'
    # default runs/segment/ root, not the repo's results/runs/ (PHASE_C_SPEC §2).
    if not os.path.isabs(train_args["project"]):
        train_args["project"] = str(REPO / train_args["project"])

    if args.smoke:
        train_args["epochs"] = 2
        train_args["fraction"] = 0.1            # ~72 train images
        train_args["name"] = cfg["train"]["name"] + "_smoke"
        train_args["exist_ok"] = True           # allow smoke reruns
        train_args["save_period"] = -1

    model = YOLO(cfg["model"])
    results = model.train(**train_args)

    # Provenance: git commit + config snapshot into the run directory.
    run_dir = Path(getattr(results, "save_dir", REPO / cfg["train"]["project"] / train_args["name"]))
    try:
        (run_dir / "git_commit.txt").write_text(git_commit() + "\n")
        shutil.copy(args.config, run_dir / "config_snapshot.yaml")
    except OSError as e:
        print(f"[WARN] could not write provenance files: {e}")

    print(f"\nRun dir: {run_dir}")
    print(f"results.csv: {run_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
