"""O019 follow-up — investigate the April control-overlap tail.

scene_attribution_orb.py found that on the APRIL bag the known-negative (march/may prefix) inlier
distribution has an elevated tail (p90 72, max 127) that overlaps the weakest april positive (68),
breaking the clean pos/neg separation march showed. Hypothesis (D046-style, to confirm not assume):
those high-scoring negatives are GENUINE same-site cross-session structural matches — the vineyard's
fixed infrastructure (buildings, pergola arch, pole clusters) seen from similar angles in different
sessions — not mis-prefixed/mislabelled scenes.

This probe re-runs the march+may scenes against the april bag PER SCENE, records the matched april
frame, prints the tail (>= TAIL_MIN inliers), and saves scene|matched-frame side-by-side panels so
the overlap can be eyeballed.

  python3 scripts/geometric/one_time/scene_attribution_tail_probe.py
    -> results/geometric/april/diagnostics/attribution_tail/  (panels + tail.json)
"""
import sys, json, time
from pathlib import Path
import numpy as np
import cv2

PKG = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PKG / "scripts" / "geometric" / "one_time"))
sys.path.insert(0, str(PKG / "scripts" / "geometric"))
# reuse the validated primitives rather than re-implement them
import scene_attribution_orb as O
from bag_config import resolve

BAG = "april"
TAIL_MIN = 60           # report every negative at/above this (covers the 68-127 overlap zone)
N_PANELS = 8            # save side-by-side panels for the top-N tail scenes


def main():
    scenes = O.scene_table()
    negs = {b: (g, p) for b, (g, p) in scenes.items() if g in ("march", "may")}   # april-bag negatives
    print(f"april-bag negatives (march+may prefix): {len(negs)}", flush=True)

    sdesc, sorb = {}, {}
    for b, (g, path) in negs.items():
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        sdesc[b] = O._thumb(cv2.resize(im, (128, 128)))
        sorb[b] = O.orb_of(cv2.resize(im, (O.MATCH_RES, O.MATCH_RES)))

    ids, cur = O.bag_stream(BAG)
    N = len(ids); coarse = list(range(0, N, O.COARSE))
    t0 = time.time()
    bank = np.stack([O._thumb(cv2.resize(O.decode_gray(cur, ids[i]), (128, 128))) for i in coarse])
    print(f"  bank {len(coarse)} over {N} ({time.time()-t0:.0f}s)", flush=True)

    shortlist = {b: np.argsort(bank @ sdesc[b])[::-1][:O.SHORTLIST_K] for b in sdesc}
    need = sorted({int(p) for sl in shortlist.values() for p in sl})
    fgray = {p: O.decode_gray(cur, ids[coarse[p]]) for p in need}
    forb = {p: O.orb_of(fgray[p]) for p in need}
    print(f"  candidate ORB {len(need)} ({time.time()-t0:.0f}s)", flush=True)

    rows = []
    for b in sdesc:
        kpS, desS = sorb[b]
        per = [(O.inliers(kpS, desS, *forb[int(p)]), int(p)) for p in shortlist[b]]
        inl, bestp = max(per, default=(0, -1))
        rows.append({"scene": b, "group": negs[b][0], "inliers": inl,
                     "frame_pos": bestp, "bag_frame_msg_index": coarse[bestp] if bestp >= 0 else -1})
    rows.sort(key=lambda r: -r["inliers"])

    tail = [r for r in rows if r["inliers"] >= TAIL_MIN]
    print(f"\n  tail (>= {TAIL_MIN} inliers): {len(tail)} of {len(rows)}")
    for r in tail:
        print(f"    {r['inliers']:>4}  {r['group']:<5}  {r['scene']}")

    OUT = PKG / "results" / "geometric" / BAG / "diagnostics" / "attribution_tail"
    OUT.mkdir(parents=True, exist_ok=True)
    for r in rows[:N_PANELS]:
        b = r["scene"]; scene_im = cv2.imread(negs[b][1])
        scene_im = cv2.resize(scene_im, (O.MATCH_RES, O.MATCH_RES))
        frame_im = cv2.cvtColor(fgray[r["frame_pos"]], cv2.COLOR_GRAY2BGR)
        panel = np.hstack([scene_im, np.full((O.MATCH_RES, 8, 3), 255, np.uint8), frame_im])
        cv2.putText(panel, f"{r['group']} {b[:22]}  |  april frame  |  inliers={r['inliers']}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imwrite(str(OUT / f"tail_{r['inliers']:04d}_{r['group']}_{b[:30]}.jpg"), panel)
    cur.connection.close()

    (OUT / "tail.json").write_text(json.dumps(
        {"bag": BAG, "tail_min_inliers": TAIL_MIN, "n_negatives": len(rows),
         "n_in_tail": len(tail), "tail": tail, "all_sorted": rows}, indent=2))
    print(f"\n  wrote {(OUT / 'tail.json').relative_to(PKG)} + {min(N_PANELS, len(rows))} panels")


if __name__ == "__main__":
    main()
