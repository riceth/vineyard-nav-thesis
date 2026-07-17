"""March-strand REPORT FIGURES (O012, Commit 9). Bag-parametrised, self-contained plotting module —
the locked multi-bag template (April+ reuse without edit). Illustrates the complete strand narrative:
in-row headline (F013/F017/F018) + in-row abstention (F024) -> non-in-row deployment gap (F020/F021)
-> mitigation (F022/F023).

  python3 figures.py --bag march            # regenerate the whole locked set (15 figures)
  python3 figures.py --bag march --only 4b  # one figure by id

Design (approved gate decisions D1-D5):
- Self-contained inference front-end MIRRORS line_fit_infer.py exactly (that verified headline script
  is NOT imported/refactored -- Commit 2b provenance protected). Base-point pixel coords + Phase-C
  class are needed for drawing and are not stored in the CSV, so inference is re-run for the selected
  frames only.
- CSV-CONSISTENCY ASSERTION (load-bearing): every per-frame figure recomputes (cls, offset, heading)
  via the mirrored estimate() and asserts equality with the committed per-frame CSV row before it
  plots. A figure can never disagree with the metric it illustrates.
- Row/centreline overlays on the raw image use projection_calibration.project_ground (D1 inverse).
- Frame reuse across strands (Figs 5/6/7 recur in 8/9/12) is intentional narrative continuity (the
  same frame carrying characterisation -> mitigation) -- see FIGURE_SPEC.md.
"""
import sys, json, argparse, bisect, collections
from pathlib import Path
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG / "scripts" / "geometric"))
import torch
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation.unet_binary.model import UNetBinary
from segmentation.unet_binary.dataset import IMAGENET_MEAN, IMAGENET_STD
import projection_calibration as C
from single_arm_dryrun import CONF, BLOB_FRAC, FRAME_PX
from bag_config import resolve, frames_for_scope
exec(open(Path(__file__).resolve().parent / "row_model.py").read())   # NEAR, FARMAX, fit_side_far, centre_linefit

# ---------------- locked style ----------------
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "savefig.dpi": 300, "figure.dpi": 150, "savefig.bbox": "tight"})
COL = {"trunk": "#2b6cff", "pole": "#ffd21e", "binary": "#00d0d0", "row": "#d1341c",
       "centre": "#1a9e4b", "driven": "#d1341c", "accept": "#1a9e4b", "reject": "#d1341c",
       "A": "#4477aa", "B": "#ee6677", "C": "#228833"}
def cls_col(cls): return COL["trunk"] if cls == 0 else (COL["pole"] if cls == 1 else COL["binary"])
LOOK = 2.0

MODELS = {
    ("A", 42): ("unet", "phase_a_unet_binary_20260704_004105/checkpoints/best.pt"),
    ("A", 43): ("unet", "phase_a_unet_binary_seed43_20260710_154347/checkpoints/best.pt"),
    ("A", 44): ("unet", "phase_a_unet_binary_seed44_20260710_181339/checkpoints/best.pt"),
    ("B", 42): ("yolo", "phase_b_yolo_binary/weights/best.pt"),
    ("B", 43): ("yolo", "phase_b_yolo_binary_seed43/weights/best.pt"),
    ("B", 44): ("yolo", "phase_b_yolo_binary_seed44/weights/best.pt"),
    ("C", 42): ("yolo", "phase_c_yolo_multiclass/weights/best.pt"),
    ("C", 43): ("yolo", "phase_c_yolo_multiclass_seed43/weights/best.pt"),
    ("C", 44): ("yolo", "phase_c_yolo_multiclass_seed44/weights/best.pt"),
}
_TF = A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])
_DEV = torch.device("cuda")
_CACHE = {}
UNET_MIN_AREA = 40


def _model(arm, seed):
    if (arm, seed) in _CACHE: return _CACHE[(arm, seed)]
    typ, ck = MODELS[(arm, seed)]
    if typ == "yolo":
        m = ("yolo", YOLO(str(PKG / "results/runs" / ck)))
    else:
        net = UNetBinary(encoder_weights=None).to(_DEV).eval()
        net.load_state_dict(torch.load(PKG / "results/runs" / ck, map_location=_DEV, weights_only=False)["model_state_dict"])
        m = ("unet", net)
    _CACHE[(arm, seed)] = m
    return m


def frontend(arm, seed, img):
    """-> list of (uc, v, cls); cls in {0 trunk, 1 pole} for Phase C, else -1 (binary). Base-point
    (uc, v) generation is IDENTICAL to line_fit_infer.py; class is extra draw-only metadata."""
    typ, m = _model(arm, seed)
    if typ == "yolo":
        r = m.predict(source=img, conf=CONF, half=True, device=0, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0: return []
        xy = r.boxes.xyxy.cpu().numpy(); cl = r.boxes.cls.cpu().numpy().astype(int)
        keep = (xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1]) <= BLOB_FRAC * FRAME_PX * FRAME_PX
        return [((x1 + x2) / 2, y2, int(c) if arm == "C" else -1)
                for (x1, y1, x2, y2), c in zip(xy[keep], cl[keep])]
    x = _TF(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))["image"].unsqueeze(0).to(_DEV)
    with torch.no_grad(): fg = (m(x).argmax(1)[0].cpu().numpy() == 1).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    return [(st[k][0] + st[k][2] / 2., st[k][1] + st[k][3] - 1, -1) for k in range(1, n) if st[k][4] >= UNET_MIN_AREA]


def fit_frame(base_cls):
    """Mirror of line_fit_infer.estimate() (byte-identical logic) + geometry retained for drawing.
    Returns dict: cls, offset, heading, n_base, plus L/R (Nx2), Lc/Rc (class), fL/fR (side fits), cl."""
    L, R, Lc, Rc = [], [], [], []
    for (uc, v, cls) in base_cls:
        g = C.project_px(uc, v, near_m=FARMAX)
        if g is not None:
            (L if uc < 320 else R).append(g); (Lc if uc < 320 else Rc).append(cls)
    L = np.array(L) if L else np.empty((0, 2)); R = np.array(R) if R else np.empty((0, 2))
    fL, fR = fit_side_far(L), fit_side_far(R)
    o = {"n_base": len(base_cls), "L": L, "R": R, "Lc": np.array(Lc), "Rc": np.array(Rc),
         "fL": fL, "fR": fR, "cl": None, "offset": None, "heading": None}
    if fL["ok"] and fR["ok"]:
        cl = centre_linefit(L[fL["inl"]], R[fR["inl"]])
        if cl is None: o["cls"] = "fitfail"; return o
        o.update(cls="two_row", offset=cl["offset"], heading=cl["heading"], cl=cl)
    elif fL["ok"] or fR["ok"]:
        o["cls"] = "single_row"
    else:
        o["cls"] = "none"
    return o


# ---------------- committed-CSV consistency assertion (load-bearing) ----------------
_CSV = {}
def _csv(scope, B):
    if scope in _CSV: return _CSV[scope]
    d = {}
    for ln in Path(B["per_frame_csv"]).read_text().splitlines()[1:]:
        a, s, i, cls, o, h, *_ = ln.split(",")
        d[(a, int(s), int(i))] = (cls, o, h)
    _CSV[scope] = d
    return d


def assert_csv(scope, B, arm, seed, frame, o):
    """A figure can never disagree with its metric: recomputed (cls, offset, heading) MUST equal the
    committed per-frame CSV row. Raises AssertionError otherwise (aborts the figure)."""
    cls_c, off_c, hdg_c = _csv(scope, B)[(arm, seed, frame)]
    assert o["cls"] == cls_c, f"{scope} {arm}s{seed} f{frame}: cls {o['cls']} != CSV {cls_c}"
    if cls_c == "two_row":
        assert abs(o["offset"] - float(off_c)) < 1e-6, f"{arm}s{seed} f{frame}: offset {o['offset']} != {off_c}"
        assert abs(o["heading"] - float(hdg_c)) < 1e-6, f"{arm}s{seed} f{frame}: heading {o['heading']} != {hdg_c}"
    return True


# ---------------- drawing primitives ----------------
def _img_line(ax, m, c, x0, x1, style, lw=2.0, n=40, color=None, label=None):
    """Draw ground line Y=mX+c (X in [x0,x1]) back onto the raw image via project_ground."""
    us, vs = [], []
    for X in np.linspace(x0, x1, n):
        px = C.project_ground(X, m * X + c)
        if px and -20 <= px[0] <= 660 and -20 <= px[1] <= 660:
            us.append(px[0]); vs.append(px[1])
    if len(us) >= 2: ax.plot(us, vs, style, lw=lw, color=color, label=label, zorder=4)


def _side_line(P, fit):
    """Display (m, c) for a fitted side + its inlier X-span (polyfit of the inliers, matching
    centre_linefit's per-side model)."""
    inl = fit["inl"]; Pi = P[inl]
    m, c = np.polyfit(Pi[:, 0], Pi[:, 1], 1)
    return float(m), float(c), float(Pi[:, 0].min()), float(Pi[:, 0].max())


def draw_combined(ax_img, ax_bev, img, base_cls, o, *, rows=True, centre=True,
                  driven_ref=False, look_marker=False, driven_path=None, near_seed=False, title_img="", title_bev=""):
    """The shared 2-panel combined view. Left = raw image + overlays; right = bird's-eye."""
    ax_img.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax_img.axis("off"); ax_img.set_title(title_img)
    for (uc, v, cls) in base_cls:
        ax_img.scatter([uc], [v], c=cls_col(cls), s=34, edgecolors="k", linewidths=0.4, zorder=3)
    L, R, fL, fR, cl = o["L"], o["R"], o["fL"], o["fR"], o["cl"]
    lines = {}
    if rows:
        if o["cls"] == "two_row":
            mL, cL, mR, cR = cl["lines"]
            xL = L[fL["inl"], 0]; xR = R[fR["inl"], 0]
            lines["L"] = (mL, cL, xL.min(), xL.max()); lines["R"] = (mR, cR, xR.min(), xR.max())
        else:
            if fL.get("ok"): m, c, x0, x1 = _side_line(L, fL); lines["L"] = (m, c, x0, x1)
            if fR.get("ok"): m, c, x0, x1 = _side_line(R, fR); lines["R"] = (m, c, x0, x1)
        for s, (m, c, x0, x1) in lines.items():
            _img_line(ax_img, m, c, x0, x1, "-", lw=2.2, color=COL["row"])
    if centre and o["cls"] == "two_row":
        mL, cL, mR, cR = cl["lines"]; mc, cc = (mL + mR) / 2, (cL + cR) / 2
        x0 = max(lines["L"][2], lines["R"][2]); x1 = min(lines["L"][3], lines["R"][3])
        _img_line(ax_img, mc, cc, x0, max(x1, LOOK + 0.5), "-", lw=2.4, color=COL["centre"])
    if driven_ref:
        _img_line(ax_img, 0.0, 0.0, 0.3, LOOK + 0.5, ":", lw=1.8, color=COL["driven"])
    if look_marker and o["cls"] == "two_row":
        px = C.project_ground(LOOK, o["offset"])
        if px: ax_img.scatter([px[0]], [px[1]], marker="*", s=150, c=COL["centre"], edgecolors="k", zorder=6)

    # bird's-eye
    def scat(P, cls_arr, fit):
        if not len(P): return
        inl = fit["inl"] if fit.get("ok") else np.zeros(len(P), bool)
        for mask, fc in ((inl, None), (~inl, "none")):
            if mask.sum() == 0: continue
            cols = [cls_col(cc) for cc in (cls_arr[mask] if len(cls_arr) else [-1] * mask.sum())]
            ax_bev.scatter(-P[mask, 1], P[mask, 0], s=30,
                           facecolors=(cols if fc is None else "none"),
                           edgecolors=("k" if fc is None else cols), linewidths=0.5, zorder=3)
    scat(L, o["Lc"], fL); scat(R, o["Rc"], fR)
    xs = np.array([0.0, 9.0])
    for s, (m, c, x0, x1) in lines.items():
        ax_bev.plot(-(m * xs + c), xs, "-", color=COL["row"], lw=2, zorder=4)
    if o["cls"] == "two_row" and centre:
        mL, cL, mR, cR = cl["lines"]; mc, cc = (mL + mR) / 2, (cL + cR) / 2
        ax_bev.plot(-(mc * xs + cc), xs, "-", color=COL["centre"], lw=2.4, zorder=5)
        ax_bev.scatter([-o["offset"]], [LOOK], marker="*", s=150, c=COL["centre"], edgecolors="k", zorder=6)
    if driven_ref:
        ax_bev.plot([0, 0], [0, LOOK + 0.5], ":", color=COL["driven"], lw=1.8, zorder=2)
    if driven_path is not None and len(driven_path):
        dp = np.array(driven_path); ax_bev.plot(-dp[:, 1], dp[:, 0], ":", color=COL["driven"], lw=2.2, zorder=5)
    if near_seed:
        ax_bev.axhline(NEAR, color="0.35", ls="--", lw=1.0, zorder=2)
        ax_bev.text(-3.9, NEAR + 0.12, "5 m near-seed window (D037)", ha="left", va="bottom", fontsize=7, color="0.3")
    ax_bev.axvline(0, color="0.6", lw=0.5); ax_bev.set_xlim(-4, 4); ax_bev.set_ylim(0, 10)
    ax_bev.set_xlabel("−Y  (right +, m)"); ax_bev.set_ylabel("X forward (m)")
    ax_bev.grid(alpha=0.3); ax_bev.set_title(title_bev)


# ---------------- context (kinematics + categories), lazily built per bag ----------------
_CTX = {}
def ctx(bag):
    if bag in _CTX: return _CTX[bag]
    B = resolve(bag, "eligible"); man = json.load(open(B["manifest"]))
    fr = sorted(man["frames"], key=lambda f: f["i"])
    x = np.array([f["x"] for f in fr]); y = np.array([f["y"] for f in fr]); t = np.array([f["t_offset_s"] for f in fr])
    speed = np.array([f["speed"] for f in fr]); elig = np.array([f["eligible"] for f in fr])
    vy = np.convolve(np.gradient(y, t), np.ones(15) / 15, mode="same")
    vx = np.convolve(np.gradient(x, t), np.ones(15) / 15, mode="same")
    cr = vx[:-1] * vy[1:] - vy[:-1] * vx[1:]; dt_ = np.maximum(np.diff(t), 1e-6)
    dang = np.abs(np.arctan2(cr, vx[:-1] * vx[1:] + vy[:-1] * vy[1:]))
    hr_raw = np.concatenate([[0.0], dang / dt_]); hr_raw[np.hypot(vx, vy) < 0.05] = 0.0
    hr = np.convolve(hr_raw, np.ones(15) / 15, mode="same")
    HR = float(np.percentile(hr[elig], 99))
    idx = {fr[k]["i"]: k for k in range(len(fr))}
    kin = {fr[k]["i"]: {"speed": float(speed[k]), "vy": float(vy[k]), "hr_deg": float(np.degrees(hr[k])),
                        "theta": float(np.arctan2(vy[k], vx[k]))} for k in range(len(fr))}
    frames = {f["i"]: f for f in man["frames"]}
    elig_idx = sorted(i for i, f in frames.items() if f["eligible"]); ec = {i: frames[i]["corridor"] for i in elig_idx}
    def category(i):
        if frames[i]["stationary"]: return "stationary"
        p = bisect.bisect_left(elig_idx, i); bc = ec[elig_idx[p - 1]] if p > 0 else None; ac = ec[elig_idx[p]] if p < len(elig_idx) else None
        return ("turn" if bc == ac else "transition") if (bc is not None and ac is not None) else "transition"
    _CTX[bag] = {"man": man, "fr": fr, "x": x, "y": y, "idx": idx, "kin": kin,
                 "HR_deg": float(np.degrees(HR)), "category": category, "frames": frames}
    return _CTX[bag]


def _out(bag, sub, name):
    p = resolve(bag, "eligible")["out_dir"].parent / "figures" / sub
    p.mkdir(parents=True, exist_ok=True)
    return p / name


def _load(bag, scope, frame):
    B = resolve(bag, scope)
    img = cv2.imread(str(B["frames_dir"] / f"{frame:05d}.jpg"))
    if img is None: raise FileNotFoundError(f"frame {frame} ({scope})")
    return B, img


# ================= public per-frame plot functions =================
def plot_in_row_frame(bag, frame, arm="A", seed=42, *, anatomy=False, near_seed=False, tag="in_row", fname=None):
    B, img = _load(bag, "eligible", frame)
    base = frontend(arm, seed, img); o = fit_frame(base); assert_csv("eligible", B, arm, seed, frame, o)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
    draw_combined(ax[0], ax[1], img, base, o, driven_ref=anatomy, look_marker=anatomy, near_seed=near_seed,
                  title_img="raw frame + detections", title_bev="bird's-eye (base_link)")
    if o["cls"] == "two_row":
        cap = f"frame {frame} · arm {arm} · two_row · offset={o['offset']:+.3f} m, heading={o['heading']:+.2f}°  (centreline_error_rms convention)"
    else:
        cap = f"frame {frame} · arm {arm} · {o['cls']} — no centreline emitted (in-row abstention, F024)"
    if anatomy: cap += "\nbase_link forward = red dotted · 2 m look-ahead = green star"
    fig.suptitle(cap, y=1.02, fontsize=9)
    fig.tight_layout(); p = _out(bag, tag, fname or f"fig_{frame}_{arm}.png"); fig.savefig(p); plt.close(fig)
    return p


def plot_non_in_row_frame(bag, frame, category, arm="A", seed=42, *, driven=False, fname=None):
    B, img = _load(bag, "non_in_row", frame); K = ctx(bag)["kin"].get(frame, {})
    base = frontend(arm, seed, img); o = fit_frame(base); assert_csv("non_in_row", B, arm, seed, frame, o)
    dp = _driven_path(bag, frame) if driven else None
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
    draw_combined(ax[0], ax[1], img, base, o, driven_path=dp,
                  title_img="raw frame + detections", title_bev="bird's-eye (base_link)")
    if o["cls"] == "two_row":
        cap = f"frame {frame} · {category} · arm {arm} · HALLUCINATED offset={o['offset']:+.2f} m, heading={o['heading']:+.1f}°  (driven_path_error)"
    else:
        cap = f"frame {frame} · {category} · arm {arm} · {o['cls']}"
    if K: cap += f"  ·  v={K.get('speed',0):.2f} m/s, |v_y|={abs(K.get('vy',0)):.2f}"
    if driven: cap += "\ngreen = hallucinated centreline · red dotted = actual driven path (odometry)"
    fig.suptitle(cap, y=1.02, fontsize=9)
    fig.tight_layout(); p = _out(bag, "non_in_row", fname or f"fig_{frame}_{category}.png"); fig.savefig(p); plt.close(fig)
    return p


def _driven_path(bag, frame, horizon=90):
    c = ctx(bag); k = c["idx"].get(frame)
    if k is None: return None
    th = c["kin"][frame]["theta"]; x0, y0 = c["x"][k], c["y"][k]; ct, st = np.cos(th), np.sin(th)
    path = []
    for j in range(k, min(k + horizon, len(c["x"]))):
        dx, dy = c["x"][j] - x0, c["y"][j] - y0
        X = ct * dx + st * dy; Y = -st * dx + ct * dy
        if -0.5 < X < 10: path.append((X, Y))
    return path


def plot_mitigation_frame(bag, frame, category, layer, arm="A", seed=42, *, fname=None):
    """layer in {'f022','f023','turn_blind'} -- single-frame mitigation panels with accept/reject banner."""
    B, img = _load(bag, "non_in_row", frame); c = ctx(bag); K = c["kin"][frame]; HR = c["HR_deg"]
    base = frontend(arm, seed, img); o = fit_frame(base); assert_csv("non_in_row", B, arm, seed, frame, o)
    # verdicts
    f022_rej = (K["speed"] <= 0.10) or (abs(K["vy"]) <= 0.30) or (K["hr_deg"] >= HR)
    why22 = "speed<0.10" if K["speed"] <= 0.10 else ("|v_y|<0.30" if abs(K["vy"]) <= 0.30 else ("heading-rate>=%.0f deg/s" % HR))
    G = (0.71, 6.7, 0.22, 12); fires = {}
    if o["cls"] == "two_row":
        off, hdg = abs(o["offset"]), abs(o["heading"]); dm = abs(o["cl"]["m_L"] - o["cl"]["m_R"])
        fires = {"|offset|>0.71": off > G[0], "|heading|>6.7": hdg > G[1], "|Δm|>0.22": dm > G[2], "n_base<12": o["n_base"] < G[3]}
    f023_rej = any(fires.values()); firedk = [k for k, v in fires.items() if v]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.9))
    draw_combined(ax[0], ax[1], img, base, o, title_img=f"frame {frame} · {category} · arm {arm}",
                  title_bev=f"bird's-eye · {o['cls']}")
    if layer == "f022":
        banner = f"F022 state gate: {'REJECT' if f022_rej else 'ACCEPT'}" + (f'  ({why22})' if f022_rej else '')
        col = COL["reject"] if f022_rej else COL["accept"]
    elif layer == "f023":
        banner = f"F023 geometry filter: {'REJECT' if f023_rej else 'ACCEPT'}" + (f"  ({', '.join(firedk)})" if f023_rej else "")
        col = COL["reject"] if f023_rej else COL["accept"]
    else:  # turn_blind
        banner = (f"F022 REJECT ({why22})   |   F023 ACCEPT (geometry within in-row p99: "
                  f"|off|={abs(o['offset']):.2f} |hdg|={abs(o['heading']):.1f})")
        col = "#8844aa"
    fig.suptitle(banner, color=col, fontsize=9, y=1.02)
    fig.tight_layout(); p = _out(bag, "mitigation", fname or f"fig_{layer}_{frame}.png"); fig.savefig(p); plt.close(fig)
    return p


# ================= 3-up wrappers =================
def plot_arm_invariance(bag, frame, seed=42):
    B = resolve(bag, "eligible"); img = cv2.imread(str(B["frames_dir"] / f"{frame:05d}.jpg"))
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
    for j, arm in enumerate("ABC"):
        base = frontend(arm, seed, img); o = fit_frame(base); assert_csv("eligible", B, arm, seed, frame, o)
        # draw only image panel per arm (compact 3-up): reuse draw on a 1x2 would be heavy; show image+overlay
        ax[j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[j].axis("off")
        for (uc, v, cls) in base: ax[j].scatter([uc], [v], c=cls_col(cls), s=20, edgecolors="k", linewidths=0.3, zorder=3)
        if o["cls"] == "two_row":
            mL, cL, mR, cR = o["cl"]["lines"]; mc, cc = (mL + mR) / 2, (cL + cR) / 2
            xL = o["L"][o["fL"]["inl"], 0]; xR = o["R"][o["fR"]["inl"], 0]
            _img_line(ax[j], mL, cL, xL.min(), xL.max(), "-", lw=2, color=COL["row"])
            _img_line(ax[j], mR, cR, xR.min(), xR.max(), "-", lw=2, color=COL["row"])
            _img_line(ax[j], mc, cc, max(xL.min(), xR.min()), LOOK + 0.5, "-", lw=2.2, color=COL["centre"])
        ax[j].set_title(f"arm {arm}  ·  offset={o['offset']:+.3f} m  hdg={o['heading']:+.2f}°", color=COL[arm])
    fig.suptitle(f"F013 arm-invariance · frame {frame} (near-identical centrelines; GT-1 indistinguishable)", y=1.02)
    fig.tight_layout(); p = _out(bag, "in_row", f"fig2_arm_invariance_{frame}.png"); fig.savefig(p); plt.close(fig)
    return p


def plot_mitigation_3up(bag, triples, layer, title, fname):
    """triples: [(frame, category, arm), ...] -> a 1x3 image-panel row with verdict subtitles."""
    c = ctx(bag); HR = c["HR_deg"]; B = resolve(bag, "non_in_row")
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.4))
    for j, (frame, category, arm) in enumerate(triples):
        img = cv2.imread(str(B["frames_dir"] / f"{frame:05d}.jpg")); K = c["kin"][frame]
        base = frontend(arm, 42, img); o = fit_frame(base); assert_csv("non_in_row", B, arm, 42, frame, o)
        ax[j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[j].axis("off")
        for (uc, v, cls) in base: ax[j].scatter([uc], [v], c=cls_col(cls), s=18, edgecolors="k", linewidths=0.3, zorder=3)
        if o["cls"] == "two_row":
            mL, cL, mR, cR = o["cl"]["lines"]; mc, cc = (mL + mR) / 2, (cL + cR) / 2
            _img_line(ax[j], mc, cc, 0.5, LOOK + 0.5, "-", lw=2.2, color=COL["centre"])
        if layer == "f022":
            rej = (K["speed"] <= 0.10) or (abs(K["vy"]) <= 0.30) or (K["hr_deg"] >= HR)
            why = "speed<0.10" if K["speed"] <= 0.10 else ("|v_y|<0.30" if abs(K["vy"]) <= 0.30 else "hr>=%.0f deg/s" % HR)
            sub = f"{category}: REJECT ({why})" if rej else f"{category}: ACCEPT"; col = COL["reject"] if rej else COL["accept"]
        else:  # f023
            G = (0.71, 6.7, 0.22, 12); fired = []
            if o["cls"] == "two_row":
                if abs(o["offset"]) > G[0]: fired.append("|offset|")
                if abs(o["heading"]) > G[1]: fired.append("|heading|")
                if abs(o["cl"]["m_L"] - o["cl"]["m_R"]) > G[2]: fired.append("|Δm|")
                if o["n_base"] < G[3]: fired.append("n_base")
            rej = bool(fired); sub = f"{category}: REJECT ({','.join(fired)})" if rej else f"{category}: ACCEPT"
            col = COL["reject"] if rej else COL["accept"]
        ax[j].set_title(f"frame {frame} · {sub}", color=col, fontsize=8)
    fig.suptitle(title, y=1.02); fig.tight_layout()
    p = _out(bag, "mitigation", fname); fig.savefig(p); plt.close(fig)
    return p


# ================= summary figures (from committed JSON; no inference) =================
def fig_forest(bag):
    d = json.load(open(resolve(bag, "eligible")["out_dir"] / "paired_crossarm.json"))["across_seed"]
    pairs = ["A-B", "A-C", "B-C"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for a, metric, lab in ((ax[0], "GT1", "GT-1 lateral offset (m)"), (ax[1], "GT2", "GT-2 heading (°)")):
        for k, pr in enumerate(pairs):
            md = d[pr][metric]["mean_diff"]; lo, hi = d[pr][metric]["ci95"]
            exc = d[pr][metric]["ci_excludes_zero"]
            a.errorbar(md, k, xerr=[[md - lo], [hi - md]], fmt="o", color=("#ee6677" if exc else "#4477aa"),
                       capsize=3, ms=5)
        a.axvline(0, color="k", lw=0.8, ls="--"); a.set_yticks(range(len(pairs))); a.set_yticklabels(pairs)
        a.set_xlabel(f"cross-arm Δ {lab}"); a.invert_yaxis(); a.grid(alpha=0.3, axis="x")
        a.set_title(("F013: GT-1 — all CIs include 0" if metric == "GT1" else "GT-2 — sub-noise-floor Δ"))
    fig.suptitle("F013 paired cross-arm bootstrap (moving-block, whole-bag) · blue = CI includes 0", y=1.03)
    fig.tight_layout(); p = _out(bag, "in_row", "fig2b_forest_paired.png"); fig.savefig(p); plt.close(fig)
    return p


def fig_tilt_sensor(bag):
    """F017 sensor-common tilt (C3): camera vs LiDAR centreline heading across the 10 pooled anchors.
    Summary-integrity assertion (the summary-figure analog of the per-frame CSV assertion): the plotted
    per-anchor means must equal the committed JSON's mean fields."""
    d = json.load(open(resolve(bag, "eligible")["out_dir"] / "lidar_crosscheck.json"))
    anc = d["anchors"]; n = len(anc)
    cam = np.array([a["cam_hdg"] for a in anc]); lid = np.array([a["lidar_hdg"] for a in anc])
    cor = [a["corridor"] for a in anc]
    assert abs(cam.mean() - d["mean_cam_hdg"]) < 0.02, "camera mean drift vs JSON"
    assert abs(lid.mean() - d["mean_lidar_hdg"]) < 0.02, "lidar mean drift vs JSON"
    assert abs((d["mean_cam_hdg"] - d["mean_lidar_hdg"]) - d["camera_minus_lidar"]) < 0.02, "diff drift vs JSON"
    camc, lidc = "#4477aa", "#e08214"; x = np.arange(n)
    fig, ax = plt.subplots(figsize=(9, 4))
    for xi, c, l in zip(x, cam, lid):
        ax.plot([xi, xi], [c, l], color="0.75", lw=1.0, zorder=1)
    ax.scatter(x, cam, s=48, color=camc, label="camera (line-fit centreline, mean of 9 models)", zorder=3)
    ax.scatter(x, lid, s=48, color=lidc, marker="s", label="LiDAR (independent row-plane fit)", zorder=3)
    ax.axhline(d["mean_cam_hdg"], color=camc, ls="--", lw=1.2); ax.axhline(d["mean_lidar_hdg"], color=lidc, ls="--", lw=1.2)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.text(n - 0.4, d["mean_cam_hdg"], f" cam mean {d['mean_cam_hdg']:+.2f}°", va="center", color=camc, fontsize=7.5)
    ax.text(n - 0.4, d["mean_lidar_hdg"], f" LiDAR mean {d['mean_lidar_hdg']:+.2f}°", va="center", color=lidc, fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([f"c{c}" for c in cor])
    ax.set_xlabel("anchor (2 per corridor · all 5 corridors)"); ax.set_ylabel("centreline heading (°)")
    ax.set_xlim(-0.5, n + 1.4); ax.grid(alpha=0.3, axis="y"); ax.legend(loc="lower right", fontsize=7.5)
    ax.set_title(f"F017 sensor-common tilt — camera vs LiDAR heading, 10 anchors × 5 corridors "
                 f"(cam {d['mean_cam_hdg']:+.2f}°, LiDAR {d['mean_lidar_hdg']:+.2f}°, diff {d['camera_minus_lidar']:+.2f}°)")
    fig.tight_layout(); p = _out(bag, "in_row", "fig3_tilt_sensor_common.png"); fig.savefig(p); plt.close(fig)
    return p


def fig_dist_bars(bag):
    d = json.load(open(resolve(bag, "non_in_row")["out_dir"] / "non_in_row_analysis.json"))["F020_output_distribution"]["per_category"]
    cats = ["stationary", "turn", "transition"]
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.25
    for j, arm in enumerate("ABC"):
        vals = [d[c][arm]["two_row"] for c in cats]
        ax.bar(np.arange(len(cats)) + (j - 1) * w, vals, w, color=COL[arm], label=f"arm {arm}")
        for x, val in zip(np.arange(len(cats)) + (j - 1) * w, vals):
            ax.text(x, val + 1, f"{val:.0f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels([f"{c}\n(n={d[c]['A']['n']//3})" for c in cats])
    ax.set_ylabel("spurious two_row  (% of frames)"); ax.set_ylim(0, 100); ax.legend()
    ax.set_title("F020 non-in-row output distribution — spurious two_row rate by category")
    fig.tight_layout(); p = _out(bag, "non_in_row", "fig5b_output_distribution.png"); fig.savefig(p); plt.close(fig)
    return p


def fig_complementarity(bag):
    d = json.load(open(resolve(bag, "eligible")["out_dir"].parent / "mitigation_evaluation" / "mitigation_analysis.json"))
    pc = d["F022_F023_causal"]["non_in_row"]["per_category"]; cats = ["stationary", "turn", "transition"]
    fig, ax = plt.subplots(figsize=(7, 4)); bottom = np.zeros(len(cats))
    segs = []
    for c in cats:
        f22 = np.mean([pc[c][a]["f022_%"] for a in "ABC"]); f23 = np.mean([pc[c][a]["f023_%"] for a in "ABC"])
        ei = np.mean([pc[c][a]["either_%"] for a in "ABC"])
        both = f22 + f23 - ei; segs.append((f22 - both, both, f23 - both, 100 - ei))  # f022-only, both, f023-only, neither
    segs = np.array(segs)
    for k, (lab, col) in enumerate([("F022 only", COL["A"]), ("both", "#8844aa"),
                                    ("F023 only", COL["B"]), ("neither", "0.8")]):
        ax.bar(cats, segs[:, k], bottom=bottom, label=lab, color=col); bottom += segs[:, k]
    ax.set_ylabel("% of spurious non-in-row two_row"); ax.set_ylim(0, 100)
    ax.legend(ncol=4, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    ax.set_title("F022 & F023 (union) complementarity by category (mean over arms)")
    fig.tight_layout(); p = _out(bag, "mitigation", "fig12_complementarity.png"); fig.savefig(p); plt.close(fig)
    return p


# ================= locked figure set =================
def build(bag, only=None):
    done = []
    F = {
        "1":  lambda: plot_in_row_frame(bag, 10247, "A", anatomy=True, fname="fig1_anatomy_10247.png"),
        "2":  lambda: plot_arm_invariance(bag, 7397),
        "2b": lambda: fig_forest(bag),
        "3":  lambda: fig_tilt_sensor(bag),
        "4":  lambda: plot_in_row_frame(bag, 10247, "C", fname="fig4_mechanism_10247_C.png"),
        "4b": lambda: plot_in_row_frame(bag, 13820, "A", near_seed=True, fname="fig4b_abstention_13820.png"),
        "5":  lambda: plot_non_in_row_frame(bag, 6, "stationary", fname="fig5_stationary_6.png"),
        "5b": lambda: fig_dist_bars(bag),
        "6":  lambda: plot_non_in_row_frame(bag, 10111, "turn", fname="fig6_turn_10111.png"),
        "7":  lambda: plot_non_in_row_frame(bag, 11264, "transition", fname="fig7_transition_11264.png"),
        "8":  lambda: plot_non_in_row_frame(bag, 11264, "transition", driven=True, fname="fig8_driven_path_11264.png"),
        "9":  lambda: plot_mitigation_3up(bag, [(6, "stationary", "A"), (10111, "turn", "A"), (11264, "transition", "A")],
                                          "f022", "F022 state gate — reject per category (odometry: speed / |v_y| / heading-rate)", "fig9_f022_3up.png"),
        "10": lambda: plot_mitigation_3up(bag, [(423, "stationary", "A"), (12801, "turn", "A"), (653, "transition", "A")],
                                          "f023", "F023 geometry filter — off-nominal catches (firing in-row-p99 threshold labelled)", "fig10_f023_3up.png"),
        "11": lambda: plot_mitigation_frame(bag, 14987, "turn", "turn_blind", fname="fig11_turn_blind_14987.png"),
        "12": lambda: fig_complementarity(bag),
    }
    ids = [only] if only else list(F)
    for i in ids:
        p = F[i](); done.append((i, p)); print(f"  fig {i:>3} -> {Path(p).relative_to(PKG)}")
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--bag", default="march"); ap.add_argument("--only", default=None)
    a = ap.parse_args()
    print(f"[figures] bag={a.bag}  (CSV-consistency assertion active on every per-frame figure)")
    build(a.bag, a.only)
    print("done.")
