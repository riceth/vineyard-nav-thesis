"""Shared filesystem locations for the geometric-strand scripts.

Single source of truth for the detection cache, which is a producer/consumer
contract: extract_detections_{val,test}.py write it, config_sweep_val.py /
config_ablation_{val,test}.py read it. Keeping the path here prevents the two
sides drifting.
"""
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]                       # vineyard_nav/
CACHE_DIR = PKG / "results" / "geometric" / "march" / "cache"   # gitignored Phase-C detection cache
