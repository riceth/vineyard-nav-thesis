"""Legacy detection-cache location for the val/test-era geometric scripts.

`CACHE_DIR` is used ONLY by the superseded val/test scripts in this directory
(`extract_detections_{val,test}.py` write detections_{val,test}.csv;
`config_sweep_val.py` / `config_ablation_{val,test}.py` read them). The whole-bag
pipeline resolves its cache via `bag_config.py` (per-bag `cache/detections.csv`)
instead, so this module lives here with the audit-trail scripts it serves — those
scripts find it via the script's own directory on sys.path when run.
"""
from pathlib import Path

PKG = Path(__file__).resolve().parents[3]                      # vineyard_nav/ (superseded/ is one deeper)
CACHE_DIR = PKG / "results" / "geometric" / "march" / "cache"  # gitignored Phase-C detection cache
