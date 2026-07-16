"""Phase C — YOLOv11-seg multiclass baseline (trunk=0, pole=1).

See docs/PHASE_C_SPEC.md for the implementation contract. Data prep reuses
scripts/perception/pipeline/coco_to_yolo.py --mode multiclass (O005); training here is a faithful
copy of the Phase B trainer with only the default config path changed, so the
B <-> C comparison isolates class structure (PHASE_C_SPEC 6).
"""
