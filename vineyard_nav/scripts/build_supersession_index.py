"""Generate docs/SUPERSESSION_INDEX.md — which decisions and findings still stand.

WHY THIS EXISTS
---------------
DECISIONS.md and FINDINGS.md are append-only by convention: when a position changes, the original
text is left intact and a correction block is added beneath it. That is the right choice for an
audit trail — a marker can see what was believed, when, and what changed it — but it is a hazard for
anything that reads the file top-to-bottom, human or model. The superseded text reads exactly as
confidently as the text that replaced it.

This script does not change that convention. It emits a short index stating, per entry, whether the
original text still stands, and produces the machine-readable list that a retrieval system can hold
in context cheaply (~4 KB, against 160 KB + 248 KB for the two source files).

CLASSIFICATION (in precedence order)
    DEAD      an explicit status line retires the whole entry -> never cite the body
    PARTIAL   part of the entry is withdrawn, the rest stands -> cite only with the caveat
    AMENDED   original stands; a correction refines or extends it -> cite the corrected position
    CURRENT   no amendment recorded

Precedence matters: an entry can carry both a supersede status and later amendments, and the
supersede wins. Classification is by marker, not by interpretation — the script never tries to
summarise what a correction says, only to point at it.

    python3 scripts/build_supersession_index.py            # write the index
    python3 scripts/build_supersession_index.py --check     # CI mode: fail if stale
"""
import argparse
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"
OUT = DOCS / "SUPERSESSION_INDEX.md"
SOURCES = [("DECISIONS.md", r"D\d{3}"), ("FINDINGS.md", r"F\d{3}(?:-[A-Z])?")]

# DEAD: the entry as a whole is retired. Matched on a status marker only -- never on prose
# mentioning the word "superseded", because a LIVE entry routinely says what it supersedes
# (F017's body reads "supersedes the F015 camera-yaw form" and F017 is current).
#
# Three shapes are in use, all three load-bearing:
#   heading    "## D002 — ... — **SUPERSEDED by D022 (2 Jul 2026)**"
#   status     "**Status:** **SUPERSEDED (13 July 2026)** by **D036**"
#   body       "**Why superseded:** ..." / "**Original status:** LOCKED"  (the older entries
#              carry no Status line at all -- the past tense of "Original" IS the marker)
DEAD_HEAD = re.compile(r"\bSUPERSEDED\b", re.I)
DEAD = re.compile(
    r"^\s*>?\s*\*\*(?:Status:?\*\*:?\s*\*{0,2}|STATUS:\s*)\s*SUPERSEDED\b"
    r"|^\s*>?\s*\*\*(?:STATUS|Status):\s*SUPERSEDED\b"
    r"|^\s*>?\s*\*\*(?:Why superseded|Original status|Original decision|Superseded because)\b",
    re.I,
)
# REVISED is NOT retirement. F005's status reads "REVISED" but the entry is live -- its scope was
# narrowed (fg IoU kept as a per-arm characterisation metric, dropped as a cross-arm ranking one)
# and its heading already states the corrected position. Treat as AMENDED: cite the revised scope.
#
# PARTIAL must be anchored to the start of the line. An entry routinely describes the partial
# supersession of a DIFFERENT entry in its cross-references -- D042's body says "F022 (superseded
# *for the control strand only*)", which is a fact about F022 and says nothing about D042.
PARTIAL = re.compile(r"^\s*>?\s*\*{0,2}[^\w\s]{0,2}\s*\*{0,2}(?:Superseded in part|Partially superseded)\b", re.I)
AMENDED = re.compile(
    r"^\s*>?\s*\*\*(?:Status:?\*\*:?\s*\*{0,2}|STATUS:\s*)?\s*REVISED\b"
    r"|^\s*>?\s*\*\*(?:Correction|Amendment|Revised (?:scope|interpretation)|Update|Clarification|"
    r"Supplementary|Reporting scope|Canopy \w+ check|Canopy deployment gap|Confirmed on|"
    r"Whole-bag cross-check|The fault is)\b",
    re.I,
)
# "**SUPERSEDES:** D002, D003" -- the reverse edge, so the index can say what killed what.
KILLS = re.compile(r"\*\*SUPERSEDES:?\*\*:?\s*(.+)", re.I)


def parse(path, idpat):
    """-> list of dicts, one per entry, in document order."""
    head = re.compile(rf"^#{{2,4}}\s*(?:\*\*)?({idpat})\b[^\n]*")
    entries, cur = [], None
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        m = head.match(line)
        if m:
            cur = {
                "id": m.group(1),
                "title": re.sub(r"^#+\s*|\*\*", "", line).strip(),
                "line": lineno,
                "state": "DEAD" if DEAD_HEAD.search(line) else "CURRENT",
                "notes": [],   # (lineno, marker-text)
                "kills": [],
            }
            entries.append(cur)
            continue
        if cur is None:
            continue
        stripped = line.strip()
        if DEAD.match(line):
            cur["state"] = "DEAD"
            cur["notes"].append((lineno, _clip(stripped)))
        elif PARTIAL.search(line):
            if cur["state"] != "DEAD":
                cur["state"] = "PARTIAL"
            cur["notes"].append((lineno, _clip(stripped)))
        elif AMENDED.match(line):
            if cur["state"] == "CURRENT":
                cur["state"] = "AMENDED"
            cur["notes"].append((lineno, _clip(stripped)))
        k = KILLS.search(line)
        if k:
            cur["kills"] += re.findall(r"[DF]\d{3}", k.group(1))
    return entries


def _clip(s, n=150):
    s = re.sub(r"^>\s*|\*\*|\s+", lambda m: " " if m.group(0).strip() == "" else "", s).strip()
    return (s[: n - 1] + "…") if len(s) > n else s


def render(all_entries):
    L = []
    A = L.append
    A("# SUPERSESSION_INDEX.md — what still stands")
    A("")
    A("**Generated** by `scripts/build_supersession_index.py`. Do not edit by hand; edit the source")
    A("entry and regenerate. Regenerate after any change to `DECISIONS.md` or `FINDINGS.md`.")
    A("")
    A("---")
    A("")
    A("## The reading contract")
    A("")
    A("`DECISIONS.md` and `FINDINGS.md` are **append-only**. When a position changed, the original text")
    A("was left in place and a correction was added beneath it. Superseded text is therefore still")
    A("present, still fluent, and still reads as current. It is not.")
    A("")
    A("**Rules, in order of precedence:**")
    A("")
    A("1. **Where an entry carries a correction, amendment, or revision block, that block supersedes")
    A("   the text above it.** Cite the corrected position, never the original.")
    A("2. **Never cite the body of a `DEAD` entry as a current position.** These entries are retained")
    A("   as an audit trail — they record what was believed and why it was abandoned. They are")
    A("   legitimate material for a *narrative* of how the design evolved, and illegitimate as evidence")
    A("   for what the design *is*.")
    A("3. **For `PARTIAL`, the caveat travels with the claim.** Quoting the surviving part without the")
    A("   withdrawal is a misrepresentation.")
    A("4. **Silence is not currency.** An entry absent from this index has no recorded amendment; that")
    A("   is not a warranty that it is correct, only that nothing has contradicted it.")
    A("")
    A("**Using the history well.** The supersessions are not embarrassments to be hidden. A design that")
    A("visibly corrected itself under evidence — a refuted hypothesis, a rejected metric, a tightened")
    A("guard — is stronger evidence of rigour than one that never changed. Cite the *arc* (\"the initial")
    A("attribution was refuted by an independent cross-check, and the finding was rewritten\") in the")
    A("discussion; cite only the *endpoint* in the results.")
    A("")

    counts = {}
    for _, entries in all_entries:
        for e in entries:
            counts[e["state"]] = counts.get(e["state"], 0) + 1
    A("| state | count | how to use it |")
    A("|---|---|---|")
    A(f"| CURRENT | {counts.get('CURRENT', 0)} | cite freely |")
    A(f"| AMENDED | {counts.get('AMENDED', 0)} | cite the correction block, not the original text |")
    A(f"| PARTIAL | {counts.get('PARTIAL', 0)} | cite only with the withdrawal attached |")
    A(f"| DEAD | {counts.get('DEAD', 0)} | never cite as current; history only |")
    A("")

    for fname, entries in all_entries:
        live = [e for e in entries if e["state"] != "CURRENT"]
        A("---")
        A("")
        A(f"## {fname} — {len(live)} of {len(entries)} entries carry a status")
        A("")
        if not live:
            A("_No amendments recorded._")
            A("")
            continue
        for e in live:
            A(f"### {e['state']} · {e['title']}  <sub>(line {e['line']})</sub>")
            A("")
            for ln, note in e["notes"]:
                A(f"- L{ln} — {note}")
            if e["kills"]:
                A(f"- **retires:** {', '.join(sorted(set(e['kills'])))}")
            A("")

        A(f"**{fname} entries with no recorded amendment (cite freely):** ")
        A("`" + "`, `".join(e["id"] for e in entries if e["state"] == "CURRENT") + "`")
        A("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed index is stale")
    a = ap.parse_args()

    all_entries = [(f, parse(DOCS / f, pat)) for f, pat in SOURCES]
    text = render(all_entries)

    if a.check:
        if not OUT.exists():
            print(f"MISSING {OUT.relative_to(DOCS.parent)} — run without --check")
            return 1
        if OUT.read_text() != text:
            print(f"STALE {OUT.relative_to(DOCS.parent)} — regenerate after editing the source docs")
            return 1
        print(f"OK {OUT.relative_to(DOCS.parent)} matches the source docs")
        return 0

    OUT.write_text(text)
    for fname, entries in all_entries:
        by = {}
        for e in entries:
            by[e["state"]] = by.get(e["state"], 0) + 1
        print(f"  {fname:15} {len(entries):3} entries  " + "  ".join(f"{k}={v}" for k, v in sorted(by.items())))
    print(f"  wrote {OUT.relative_to(DOCS.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
