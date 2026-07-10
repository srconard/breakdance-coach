"""Build the move-organized breakdance library (moves.html).

The reference_library index groups by DATASET; this groups by MOVE — one card
per move (Flair, 1990, Swipes, Footwork, Freeze, Uprock, power moves...), each
linking its clean reference animation(s), and where we have them, the
tutorial-wiki breakdown and Shawn's own reconstruction. The long mocap
freestyles (AIST++/FineDance) are listed separately as multi-move runs.

Move tags: if output/reference_library/move_tags.json exists (produced by
vlm_movetag.py --merge-into; keyed by library stem), each freestyle run shows
its VLM-detected move segments with timestamps, and each move card gains a
"seen in freestyles" cross-reference. Timestamps are text only — the viewer
has no ?t= playback deep-link (its only URL param is the ?shot= QA hook).

Accurate by construction: the reference animations are mocap / Mixamo clean
animation, so they don't carry the single-view leg-extension limit that our
own video reconstructions do (see 2026-07-05-leg-extension-investigation).

Usage:
    python -m analyzer_3d.src.move_library [root] [-o moves.html]
"""

import argparse
import html
import json
import sys
from pathlib import Path
from urllib.parse import quote

VAULT = "Shawn's Vault"


def obsidian_uri(vault_relpath):
    return (f"obsidian://open?vault={quote(VAULT)}"
            f"&file={quote(vault_relpath)}")


# Each move: name, blurb, and the viewer files (relative to moves.html) that
# depict it. Optional: a tutorial-wiki Obsidian link and reconstruction viewers.
# Only entries whose files exist on disk are rendered.
MOVES = [
    {
        "name": "Flair",
        "blurb": "The signature power move — legs circle in a wide straddle while the hands alternate as support. The project's showcase move.",
        "anims": [("Mixamo · Beta mannequin", "mixamo_beta/Flair_viewer.html")],
        "tutorial": ("Flare Tutorial by Sambo (step-by-step)",
                     obsidian_uri("XX. Other/ProjectsFiles/flare_tutorial/Flare_Tutorial_by_Sambo")),
        "reconstructions": [
            ("Reconstruction — clean 1080p60 tutorial", "../3d/flare60/flare60_demo_refined_viewer.html"),
            ("Reconstruction — SAMBO flare", "../3d/phase1/flare_sambo_viewer.html"),
        ],
    },
    {
        "name": "Breakdance 1990",
        "blurb": "One-handed handstand spin — balance on a single hand and rotate.",
        "anims": [("Mixamo · Beta mannequin", "mixamo_beta/Breakdance 1990_viewer.html")],
    },
    {
        "name": "Swipes",
        "blurb": "Rotational power move — the chest twists first and the hands trade support while hips and legs whip around to follow, landing feet-to-hands again.",
        "anims": [("Mixamo · Beta mannequin", "mixamo_beta/Breakdance Swipes_viewer.html")],
    },
    {
        "name": "Footwork",
        "blurb": "Ground-level stepping patterns (6-step family) — the connective tissue of a set.",
        "anims": [
            ("Mixamo · Footwork 1", "mixamo_beta/Breakdance Footwork 1_viewer.html"),
            ("Mixamo · Footwork 2", "mixamo_beta/Breakdance Footwork 2_viewer.html"),
            ("Mixamo · Footwork 3", "mixamo_beta/Breakdance Footwork 3_viewer.html"),
            ("CMU mocap · Fancy Footwork", "cmu85/85_04_viewer.html"),
        ],
    },
    {
        "name": "Freezes",
        "blurb": "Held balance poses that punctuate a set — baby freeze, chair, and variations.",
        "anims": [
            ("Mixamo · Freezes", "mixamo_beta/Breakdance Freezes_viewer.html"),
            ("Mixamo · Freeze Var 1", "mixamo_beta/Breakdance Freeze Var 1_viewer.html"),
            ("Mixamo · Freeze Var 2", "mixamo_beta/Breakdance Freeze Var 2_viewer.html"),
            ("Mixamo · Freeze Var 3", "mixamo_beta/Breakdance Freeze Var 3_viewer.html"),
            ("Mixamo · Freeze Var 4", "mixamo_beta/Breakdance Freeze Var 4_viewer.html"),
        ],
    },
    {
        "name": "Uprock / Toprock",
        "blurb": "Standing dance that opens a set before dropping to the floor.",
        "anims": [
            ("Mixamo · Uprock", "mixamo_beta/Breakdance Uprock_viewer.html"),
            ("Mixamo · Ready (stance)", "mixamo_beta/Breakdance Ready_viewer.html"),
        ],
    },
    {
        "name": "Power moves (misc)",
        "blurb": "Real b-boy floorwork from CMU marker mocap — helicopter, handstand kicks, kick flips.",
        "anims": [
            ("CMU · Helicopter", "cmu85/85_08_viewer.html"),
            ("CMU · Handstand Kicks", "cmu85/85_05_viewer.html"),
            ("CMU · Kick Flip", "cmu85/85_06_viewer.html"),
        ],
    },
]

# Multi-move freestyle runs (not single moves — browse/scrub). Runs whose
# library stem appears in move_tags.json get per-segment move tags rendered.
FREESTYLE = [
    ("CMU · Long Break Sequence", "cmu85/85_12_viewer.html"),
    ("CMU · Break Sequence w/ Flips", "cmu85/85_14_viewer.html"),
    ("AIST++ · Advanced pro freestyle (48s)", "aistpp/gBR_sFM_cAll_d04_mBR0_ch01_viewer.html"),
]

# vlm_movetag.py MOVE_VOCAB label -> move card name above. Labels with no
# card (transition, kick-up, other) are shown on the run but not cross-linked.
TAG_TO_CARD = {
    "toprock": "Uprock / Toprock", "uprock": "Uprock / Toprock",
    "6-step": "Footwork", "footwork-other": "Footwork", "cc": "Footwork",
    "swipe": "Swipes",
    "flare": "Flair",
    "1990": "Breakdance 1990", "2000": "Breakdance 1990",
    "windmill": "Power moves (misc)", "headspin": "Power moves (misc)",
    "backspin": "Power moves (misc)", "halo": "Power moves (misc)",
    "power-combo": "Power moves (misc)",
    "baby-freeze": "Freezes", "chair-freeze": "Freezes",
    "airchair": "Freezes", "elbow-freeze": "Freezes",
    "handstand-freeze": "Freezes",
}


PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Breakdance Move Library</title><style>
:root{{--bg:#101114;--panel:#1b1d22;--fg:#d7dae0;--teal:#19b3a6;--orange:#ea6b2e;--mut:#8b919b;}}
*{{box-sizing:border-box;margin:0;}}body{{background:var(--bg);color:var(--fg);
font:14px/1.55 "Segoe UI",system-ui,sans-serif;padding:28px 32px;max-width:1000px;margin:0 auto;}}
h1{{font-size:25px;margin-bottom:4px;}}h1 .t{{color:var(--teal);}}h1 .o{{color:var(--orange);}}
.sub{{color:var(--mut);margin-bottom:26px;}}
.move{{background:var(--panel);border:1px solid #2a2d34;border-radius:12px;padding:16px 18px;margin-bottom:14px;}}
.move h2{{font-size:18px;margin-bottom:3px;}}
.move .blurb{{color:var(--mut);font-size:13px;margin-bottom:11px;}}
.links{{display:flex;flex-wrap:wrap;gap:8px;}}
a.pill{{display:inline-block;text-decoration:none;font-size:12.5px;padding:6px 12px;border-radius:20px;
background:#23262d;color:var(--fg);border:1px solid #343842;transition:.12s;}}
a.pill:hover{{border-color:var(--teal);background:#2b2f37;}}
a.tut{{border-color:var(--orange);color:#ffd9c2;}}a.tut:hover{{background:#2f261f;}}
a.rec{{border-style:dashed;color:#bcd;}}
.lbl{{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.05em;margin:8px 0 4px;}}
h2.section{{font-size:16px;margin:26px 0 10px;border-bottom:1px solid #2a2d34;padding-bottom:6px;}}
.frun{{padding:9px 0;border-bottom:1px solid #23262d;}}.frun:last-child{{border-bottom:0;}}
.tags{{margin-top:7px;font-size:12px;color:var(--mut);}}
.seg{{display:inline-block;background:#23262d;border:1px solid #343842;border-radius:4px;
padding:1px 7px;margin:3px 4px 0 0;color:#9fb3c8;font-size:11.5px;}}
.seg b{{color:var(--fg);font-weight:600;}}
a.free{{border-color:#3a5b57;color:#a9d6d1;}}a.free:hover{{background:#22302e;}}
footer{{color:var(--mut);font-size:12px;margin-top:30px;border-top:1px solid #2a2d34;padding-top:12px;}}
</style></head><body>
<h1><span class="t">Break</span><span class="o">dance</span> Move Library</h1>
<div class="sub">{count} moves. Each card links clean reference animations to scrub, plus tutorial breakdowns and reconstructions where we have them. Click any pill to open its study viewer.</div>
{moves}
<h2 class="section">Freestyle sequences (multiple moves — scrub to explore)</h2>
<div class="move">{freestyle}</div>
<footer>Reference animations are mocap / Mixamo clean animation (accurate — no single-view leg-extension limit).
Orange pill = tutorial wiki (opens in Obsidian). Dashed pill = a reconstruction from real video (has the synced source panel for form detail).
Detected-move tags come from <code>vlm_movetag.py</code> (Gemini video analysis of the source clips); timestamps are scrub positions in the linked viewer. Generated by <code>move_library.py</code>.</footer>
</body></html>"""


def exists(root, rel):
    return (root / rel).exists()


def pill(label, href, cls="pill"):
    return f'<a class="{cls}" href="{html.escape(href)}">{html.escape(label)}</a>'


def load_move_tags(root):
    """clips mapping from move_tags.json (vlm_movetag.py --merge-into), or {}."""
    p = root / "move_tags.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("clips", {})
    except (json.JSONDecodeError, OSError):
        return {}


def viewer_stem(rel):
    """Library stem for a viewer path — the move_tags.json join key."""
    stem = Path(rel).stem
    return stem[:-len("_viewer")] if stem.endswith("_viewer") else stem


def fmt_ts(sec):
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"


def freestyle_html(root, tags):
    """Render freestyle runs (+ per-run detected-move segments) and collect
    per-move-card cross-references: {card: [(run_label, rel, move, spans)]}."""
    blocks, seen = [], {}
    for label, rel in FREESTYLE:
        if not exists(root, rel):
            continue
        parts = [f'<div class="frun"><div class="links">{pill(label, rel)}</div>']
        clip = tags.get(viewer_stem(rel))
        segs = (clip or {}).get("segments") or []
        if segs:
            chips = "".join(
                f'<span class="seg" title="{html.escape(s.get("notes") or "")}">'
                f'<b>{html.escape(s["move"])}</b> '
                f'{fmt_ts(s["start_s"])}–{fmt_ts(s["end_s"])}</span>'
                for s in segs)
            cov = clip.get("coverage")
            meta = "detected moves (" + html.escape(clip.get("model", "VLM"))
            if cov is not None:
                meta += f", coverage {cov:.0%}"
            parts.append(f'<div class="tags">{meta}): {chips}</div>')
            per_move = {}
            for s in segs:
                card = TAG_TO_CARD.get(s["move"])
                if card:
                    per_move.setdefault((card, s["move"]), []).append(
                        (s["start_s"], s["end_s"]))
            for (card, move), spans in per_move.items():
                seen.setdefault(card, []).append((label, rel, move, spans))
        parts.append("</div>")
        blocks.append("".join(parts))
    return "\n".join(blocks), seen


def build(root, out):
    tags = load_move_tags(root)
    free, seen = freestyle_html(root, tags)
    move_html, count = [], 0
    for m in MOVES:
        anims = [(l, r) for l, r in m["anims"] if exists(root, r)]
        if not anims:
            continue
        count += 1
        parts = [f'<div class="move"><h2>{html.escape(m["name"])}</h2>'
                 f'<div class="blurb">{html.escape(m["blurb"])}</div>']
        parts.append('<div class="lbl">animations</div><div class="links">')
        parts += [pill(l, r) for l, r in anims]
        parts.append('</div>')
        if m.get("tutorial"):
            parts.append('<div class="lbl">tutorial</div><div class="links">')
            parts.append(pill(m["tutorial"][0], m["tutorial"][1], "pill tut"))
            parts.append('</div>')
        recs = [(l, r) for l, r in m.get("reconstructions", []) if exists(root, r)]
        if recs:
            parts.append('<div class="lbl">your reconstructions</div><div class="links">')
            parts += [pill(l, r, "pill rec") for l, r in recs]
            parts.append('</div>')
        refs = seen.get(m["name"], [])
        if refs:
            # No ?t= deep-link in the viewer (only the ?shot= QA hook), so
            # timestamps are label text — open the run and scrub to them.
            parts.append('<div class="lbl">seen in freestyles</div><div class="links">')
            for run_label, rel, move, spans in refs:
                ts = ", ".join(f"{fmt_ts(a)}–{fmt_ts(b)}" for a, b in spans[:4])
                extra = f" +{len(spans) - 4} more" if len(spans) > 4 else ""
                parts.append(pill(f"{run_label} · {move} @ {ts}{extra}",
                                  rel, "pill free"))
            parts.append('</div>')
        parts.append('</div>')
        move_html.append("".join(parts))

    out.write_text(PAGE.format(count=count, moves="\n".join(move_html),
                               freestyle=free), encoding="utf-8")
    print(f"[move-library] {count} move cards -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", nargs="?", default="output/reference_library")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    build(root, Path(args.out) if args.out else root / "moves.html")


if __name__ == "__main__":
    sys.exit(main() or 0)
