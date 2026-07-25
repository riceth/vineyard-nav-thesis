"""O019 follow-up — validate the fine-verify stage of the proposed threshold rule.

The coarse thumbnail bank samples every COARSE-th frame, so a TRUE member's exact source frame is
usually skipped — the best coarse candidate is a near-neighbour (a few frames away), giving a
moderate inlier count (the April weak positive scored only 68 this way). Fine-verify decodes the
FULL-RESOLUTION ±HALF-frame neighbourhood (non-strided) around the best coarse candidate and takes
the max inlier count. Hypotheses to confirm before locking the rule:
  (a) the April weak positive RECOVERS: coarse ~68 -> fine >= T_high (200)  [true frame found]
  (b) the cross-session tail STAYS DOWN: march_118 coarse 127 -> fine < T_high  [no identical frame exists]

  python3 scripts/geometric/one_time/scene_attribution_fineverify.py
    -> prints coarse vs fine for the April weak positive + the top tail negatives; writes fineverify.json
"""
import sys, json, time
from pathlib import Path
import numpy as np
import cv2

PKG = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PKG / "scripts" / "geometric" / "one_time"))
sys.path.insert(0, str(PKG / "scripts" / "geometric"))
import scene_attribution_orb as O

BAG = "april"
T_HIGH = 200
HALF = 30               # ±frames decoded at full res around the best coarse candidate


def coarse_best(scene_orb, sdesc, bank, coarse_idx, ids, cur):
    """Best coarse candidate for one scene: (max_inliers, best_frame_msg_index)."""
    sl = np.argsort(bank @ sdesc)[::-1][:O.SHORTLIST_K]
    best = (0, -1)
    for p in sl:
        fo = O.orb_of(O.decode_gray(cur, ids[coarse_idx[int(p)]]))
        n = O.inliers(*scene_orb, *fo)
        if n > best[0]:
            best = (n, coarse_idx[int(p)])
    return best


def fine_verify(scene_orb, center_idx, ids, cur):
    """Max inliers over the full-res ±HALF neighbourhood (non-strided) of center_idx."""
    lo, hi = max(0, center_idx - HALF), min(len(ids) - 1, center_idx + HALF)
    best = 0
    for j in range(lo, hi + 1):
        fo = O.orb_of(O.decode_gray(cur, ids[j]))
        best = max(best, O.inliers(*scene_orb, *fo))
    return best


def main():
    scenes = O.scene_table()
    ids, cur = O.bag_stream(BAG)
    N = len(ids); coarse_idx = list(range(0, N, O.COARSE))
    t0 = time.time()
    bank = np.stack([O._thumb(cv2.resize(O.decode_gray(cur, ids[i]), (128, 128))) for i in coarse_idx])
    print(f"[{BAG}] bank {len(coarse_idx)} over {N} ({time.time()-t0:.0f}s)", flush=True)

    def load(path):
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return (O._thumb(cv2.resize(im, (128, 128))), O.orb_of(cv2.resize(im, (O.MATCH_RES, O.MATCH_RES))))

    # 1. April positives (10 april-prefix scenes) -> coarse, find the weak one
    print("\n=== April positives (coarse) ===", flush=True)
    pos = {b: p for b, (g, p) in scenes.items() if g == "april"}
    pos_rows = []
    for b, path in pos.items():
        sd, so = load(path)
        n, fi = coarse_best(so, sd, bank, coarse_idx, ids, cur)
        pos_rows.append({"scene": b, "coarse": n, "best_idx": fi})
        print(f"  {n:>4}  {b}", flush=True)
    pos_rows.sort(key=lambda r: r["coarse"])
    weak = pos_rows[0]

    # 2. Fine-verify: the weak positive + the top tail negatives
    tail = json.load(open(PKG / "results/geometric" / BAG / "diagnostics/attribution_tail/tail.json"))
    top_negs = tail["all_sorted"][:3]     # march_118 (127), march_107 (114), march_147 (103)

    checks = [("weak_positive", weak["scene"], scenes[weak["scene"]][1], weak["coarse"], weak["best_idx"], "recover>=%d" % T_HIGH)]
    for r in top_negs:
        checks.append(("tail_negative", r["scene"], scenes[r["scene"]][1], r["inliers"], r["bag_frame_msg_index"], "stay<%d" % T_HIGH))

    print("\n=== Fine-verify (full-res +/-%d neighbourhood) ===" % HALF, flush=True)
    out = []
    for kind, b, path, coarse_n, center, expect in checks:
        _, so = load(path)
        fine_n = fine_verify(so, center, ids, cur)
        verdict = "PRESENT" if fine_n >= T_HIGH else "absent"
        ok = (kind == "weak_positive" and fine_n >= T_HIGH) or (kind == "tail_negative" and fine_n < T_HIGH)
        out.append({"kind": kind, "scene": b, "coarse": coarse_n, "fine": fine_n,
                    "verdict": verdict, "expected": expect, "as_expected": ok})
        print(f"  {kind:<14} {b:<26} coarse {coarse_n:>4} -> fine {fine_n:>4}  [{verdict}]  expect {expect}: {'OK' if ok else 'FAIL'}", flush=True)
    cur.connection.close()

    OUT = PKG / "results/geometric" / BAG / "diagnostics/attribution_tail/fineverify.json"
    OUT.write_text(json.dumps({"bag": BAG, "T_high": T_HIGH, "half": HALF,
                               "april_positives_coarse": pos_rows, "checks": out}, indent=2))
    allok = all(r["as_expected"] for r in out)
    print(f"\n  all checks as expected: {allok}")
    print(f"  wrote {OUT.relative_to(PKG)}")


if __name__ == "__main__":
    main()
