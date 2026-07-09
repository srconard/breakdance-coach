"""Build the move-organized breakdance library (moves.html).

The reference_library index groups by DATASET; this groups by MOVE — one card
per move (Flair, 1990, Footwork, Freeze, Uprock, power moves...), each linking
its clean reference animation(s), and where we have them, the tutorial-wiki
breakdown and Shawn's own reconstruction. The long mocap freestyles
(AIST++/FineDance) are listed separately as multi-move runs until tagged.

Accurate by construction: the reference animations are mocap / Mixamo clean
animation, so they don't carry the single-view leg-extension limit that our
own video reconstructions do (see 2026-07-05-leg-extension-investigation).

Usage:
    python -m analyzer_3d.src.move_library [root] [-o moves.html]
"""

import argparse
import html
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

# Multi-move freestyle runs (not single moves — browse/scrub; tag later)
FREESTYLE = [
    ("CMU · Long Break Sequence", "cmu85/85_12_viewer.html"),
    ("CMU · Break Sequence w/ Flips", "cmu85/85_14_viewer.html"),
    ("AIST++ · Advanced pro freestyle (48s)", "aistpp/gBR_sFM_cAll_d04_mBR0_ch01_viewer.html"),
]


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
footer{{color:var(--mut);font-size:12px;margin-top:30px;border-top:1px solid #2a2d34;padding-top:12px;}}
</style></head><body>
<h1><span class="t">Break</span><span class="o">dance</span> Move Library</h1>
<div class="sub">{count} moves. Each card links clean reference animations to scrub, plus tutorial breakdowns and reconstructions where we have them. Click any pill to open its study viewer.</div>
{moves}
<h2 class="section">Freestyle sequences (multiple moves — scrub to explore)</h2>
<div class="move"><div class="links">{freestyle}</div></div>
<footer>Reference animations are mocap / Mixamo clean animation (accurate — no single-view leg-extension limit).
Orange pill = tutorial wiki (opens in Obsidian). Dashed pill = a reconstruction from real video (has the synced source panel for form detail). Generated by <code>move_library.py</code>.</footer>
</body></html>"""


def exists(root, rel):
    return (root / rel).exists()


def pill(label, href, cls="pill"):
    return f'<a class="{cls}" href="{html.escape(href)}">{html.escape(label)}</a>'


def build(root, out):
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
        parts.append('</div>')
        move_html.append("".join(parts))

    free = [pill(l, r) for l, r in FREESTYLE if exists(root, r)]
    out.write_text(PAGE.format(count=count, moves="\n".join(move_html),
                               freestyle="".join(free)), encoding="utf-8")
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
