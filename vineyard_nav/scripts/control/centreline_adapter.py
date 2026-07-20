"""P-1a centreline input adapter (PID_PIPELINE_SPEC.md §2, CP-P2).

Reads the geometric pipeline's per-frame centreline CSV (line_fit_per_frame.csv) and produces,
per (arm, seed), an INDEPENDENT time-ordered command-input stream — P-1a: the 3 seeds of an arm
run independently, no cross-seed averaging at this stage (cross-seed mean ± SD is a CP-P4 reporting
choice, not a controller input). Per frame:

  cls == two_row  -> emit offset (m) + heading (deg) at the 2 m look-ahead (D038)
  cls != two_row  -> F024 abstention: no offset/heading emitted; the command generator holds last (D043)

Frames carry their pass_id (from the CP-1 manifest) so the command generator can segment the stream
into individual in-row passes (per-pass controller state reset). Bag-parametrised. Consumed by
command_generator.py (CP-P2). No state, no controller logic here — this is purely the input shaping.
"""
import sys
import json
from pathlib import Path
from collections import defaultdict

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / "scripts" / "geometric"))
from bag_config import resolve

SEEDS = (42, 43, 44)
ARMS = ("A", "B", "C")


def load_streams(bag="march"):
    """Return (streams, manifest). streams[(arm, seed)] = list of per-frame dicts, sorted by frame
    index i: {i, pass_id, cls, offset, heading, abstained}. offset/heading are None on abstention
    (cls != two_row); abstained is True there. P-1a: one independent stream per (arm, seed)."""
    B = resolve(bag, "eligible")
    man = json.load(open(B["manifest"]))
    pass_of = {f["i"]: f["pass_id"] for f in man["frames"]}
    streams = defaultdict(list)
    for ln in Path(B["per_frame_csv"]).read_text().splitlines()[1:]:
        a, s, i, cls, off, hdg, *_ = ln.split(",")
        s, i = int(s), int(i)
        two = (cls == "two_row" and off and hdg)
        streams[(a, s)].append({
            "i": i, "pass_id": pass_of[i], "cls": cls,
            "offset": float(off) if two else None,
            "heading": float(hdg) if two else None,
            "abstained": not two})
    for k in streams:
        streams[k].sort(key=lambda r: r["i"])
    return dict(streams), man


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--bag", default="march")
    st, _ = load_streams(ap.parse_args().bag)
    print(f"loaded {len(st)} streams (arm x seed)")
    for k in sorted(st):
        s = st[k]
        ab = sum(1 for r in s if r["abstained"])
        print(f"  {k}: {len(s)} frames, {ab} abstentions ({100*ab/len(s):.1f}%), "
              f"passes {sorted(set(r['pass_id'] for r in s))}")
