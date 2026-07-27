"""check_bag_complete.py --bag <name> — the authoritative "is this bag genuinely done?" gate.

A bag's evaluation is DONE only when BOTH (a) every expected committed artefact exists (analysis
JSONs + the 15 report figures) AND (b) docs/STATUS.md carries a consolidated "Confirmed on <bag>"
summary bullet. Artefacts-on-disk alone are NOT "done" — the summary is the step easy to forget
across a multi-session bag run (it was missed for may). This gate makes it impossible to declare a
bag done without one: loud banner + non-zero exit if either is missing. Wired as the final advisory
of scripts/control/control.py so the verdict prints at the natural end of every full bag run.

march is the design bag: its findings F010-F028 ARE the primary write-ups, so the "Confirmed on
<bag>" rule does not apply to it (artefacts are still checked)."""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bag_config import resolve, BAGS

PKG = Path(__file__).resolve().parents[2]
STATUS = PKG / "docs" / "STATUS.md"
DESIGN_BAG = "march"          # findings established here, not "confirmed" -> exempt from the summary rule

EVAL_JSONS = ["line_fit_report.json", "paired_crossarm.json", "config_analysis.json",
              "lidar_crosscheck.json", "single_row_analysis.json"]                    # Stage C (in-row)
FINAL_JSONS = ["non_in_row_evaluation/non_in_row_analysis.json",                      # Stage D
               "mitigation_evaluation/mitigation_analysis.json",
               "mitigation_evaluation/state_gate_native.json",
               "command_evaluation/command_summary.json",                            # Stage F (control)
               "command_evaluation/gain_kfold.json",
               "command_evaluation/command_smoothness.json"]
N_FIGURES = 15                                                                        # Stage E
CACHE_ADVISORY = ["detections.csv", "blob_audit.json"]    # gitignored/regenerable -> advisory only


def check(bag):
    B = resolve(bag, "eligible")
    final = B["out_dir"].parent                              # results/geometric/<bag>/final
    missing = []
    for j in EVAL_JSONS:
        if not (B["out_dir"] / j).exists(): missing.append(f"{bag}_evaluation/{j}")
    for j in FINAL_JSONS:
        if not (final / j).exists(): missing.append(j)
    figs = list((final / "figures").rglob("*.png")) if (final / "figures").is_dir() else []
    if len(figs) < N_FIGURES: missing.append(f"figures/ (found {len(figs)}/{N_FIGURES})")
    advisory = [c for c in CACHE_ADVISORY if not (B["cache_dir"] / c).exists()]
    txt = STATUS.read_text().lower() if STATUS.exists() else ""
    summary_required = bag != DESIGN_BAG
    has_summary = (f"confirmed on {bag}" in txt) if summary_required else True
    return missing, advisory, summary_required, has_summary


def main():
    ap = argparse.ArgumentParser(description="Verify a bag's evaluation is genuinely complete "
                                             "(all artefacts + the consolidated STATUS 'Confirmed on <bag>' summary).")
    ap.add_argument("--bag", required=True, choices=sorted(BAGS))
    ap.add_argument("--quiet-if-ok", action="store_true", help="print nothing when complete (control.py hook)")
    a = ap.parse_args()
    missing, advisory, summary_required, has_summary = check(a.bag)
    ok = (not missing) and has_summary
    bar = "=" * 74
    if ok:
        if not a.quiet_if_ok:
            print(f"\n{bar}\n  BAG '{a.bag}' COMPLETE — all artefacts present"
                  + (f" + STATUS 'Confirmed on {a.bag}' summary written" if summary_required else "") + f".\n{bar}")
            if advisory:
                print(f"  (advisory: regenerable cache absent — {', '.join(advisory)}; regenerate via extract_detections.py)")
        return 0
    print(f"\n{bar}\n  ⚠️  BAG '{a.bag}' NOT COMPLETE — do NOT consider this bag done.\n{bar}")
    if missing:
        print("  Missing committed artefacts:")
        for m in missing: print(f"      - {m}")
    if summary_required and not has_summary:
        print(f"  ⚠️  docs/STATUS.md has NO consolidated 'Confirmed on {a.bag}' summary bullet.")
        print(f"      A bag is not done until this is written (spot-checked, April/May style) — not just")
        print(f"      when the analysis artefacts exist. This is the step that was missed for may.")
    if advisory:
        print(f"  (advisory: regenerable cache absent — {', '.join(advisory)})")
    print(bar)
    return 1


if __name__ == "__main__":
    sys.exit(main())
