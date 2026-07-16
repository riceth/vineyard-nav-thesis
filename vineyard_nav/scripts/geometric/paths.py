"""Legacy detection-cache location for the val/test-era geometric scripts.

`CACHE_DIR` is now used ONLY by the superseded val/test scripts
(`superseded/extract_detections_{val,test}.py` write detections_{val,test}.csv;
`superseded/config_sweep_val.py` / `config_ablation_{val,test}.py` read them).
The whole-bag pipeline resolves its cache via `bag_config.py` (per-bag
`cache/detections.csv`) instead, so this module is retained only for the audit-
trail scripts.
"""
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]                       # vineyard_nav/
CACHE_DIR = PKG / "results" / "geometric" / "march" / "cache"   # gitignored Phase-C detection cache
