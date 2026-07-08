#!/usr/bin/env python3
"""Phase B YOLOv11-seg binary training (PHASE_B_SPEC section 6).

Fine-tunes yolo11n-seg from COCO-pretrained weights on the binary YOLO labels
produced by scripts/coco_to_yolo.py (O005). Decisions: D023 (YOLOv11-seg via
ultralytics), D013 (augmentation parity), D016 (reproducibility). No test-set
access here (rule 5).

Smoke run (PHASE_B_SPEC 6.4): `python -m segmentation.yolo_binary.train --smoke`
runs 2 epochs on a 10% subsample (fraction=0.1) to check no-OOM + healthy loss
before committing to the full 100-epoch run.
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
    ap = argparse.ArgumentParser(description="Phase B YOLOv11-seg binary training.")
    ap.add_argument("--config", default=str(REPO / "configs/phase_b_yolo_binary_train.yaml"))
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
    # default runs/segment/ root, not the repo's results/runs/ (PHASE_B_SPEC §2).
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
