"""Build Shawn's Move Library — a phone-first static web app (2026-11-17).

Gathers everything the breakdance-coach project has produced into one
browsable place, one page per MOVE:

* the Instagram library (every saved + tagged clip, with its contact sheet,
  a link to the post, and its tags),
* his own video-to-3D extractions (the nightly NAS GVHMR runs over the
  Instagram clips + the early tutorial reconstructions), each with a plain
  quality badge (clean / usable / broken) and the reason, from the 2026-11-17
  audit (library_audit.py numbers + library_vision_judge.py verdicts),
* the motion-capture reference library (Mixamo, CMU subject 85, AIST++,
  FineDance) placed beside the moves they show — freestyle runs deep-link to
  the tagged segment (?t= / ?loop=),
* tutorial-wiki steps (per-step clips) and Dance Mirror lessons.

Everything streams through ONE shared viewer page (viewer/library.html +
app.bundle.js): ?glb=..&video=..&fps=.. — no 5-10 MB self-contained HTML per
move. Private: the pages contain other people's videos; serve on the tailnet
only (the NAS bridge's /webapps/move-library/), never publish.

Pure stdlib (runs on the NAS host python 3.8). Media are HARD-LINKED from
their NAS homes when possible (same btrfs volume: no extra space), copied
otherwise; the bridge refuses symlinks that leave the app folder.

Usage (NAS):
    python3 -m analyzer_3d.src.move_library_site --config library_site.json
or with explicit paths (see --help). Rebuild = run it again; it is idempotent.
"""

import argparse
import datetime as dt
import html
import json
import os
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import quote, urlencode

E = html.escape

# ---------------------------------------------------------------- vocabulary
# family slug -> (name, blurb). Order = display order.
FAMILIES = [
    ("toprock", "Toprock", "Standing dance that opens a set."),
    ("footwork", "Footwork", "Ground-level stepping on hands and feet — the connective tissue of a set."),
    ("freezes", "Freezes", "Held balances that punctuate a set."),
    ("power", "Power moves", "Continuous rotational moves — flares, windmills, spins."),
    ("transitions", "Drops, transitions & floorwork", "Getting to the floor, between moves and back up."),
    ("freestyle", "Freestyle runs", "Whole sets with many moves in a row (mostly motion capture)."),
    ("acro", "Acro & tricking", "Flips, kicks and acrobatics."),
    ("parkour", "Parkour", "Vaults, wall runs and moving through obstacles."),
    ("partner", "Partner work", "Lifts, partner dance and contact improvisation."),
    ("dance", "Other dance", "Contemporary, hip-hop, popping, tutting, capoeira and more."),
    ("conditioning", "Conditioning & mobility", "Strength, handstands, flexibility — the body that makes the moves."),
    ("outdoors", "Outdoors & other", "Climbing, canyoning, skating and everything else he saved."),
]
FAMILY_NAME = {k: n for k, n, _ in FAMILIES}

# move slug: (name, family, blurb, [raw tags that map here])
MOVES = [
    ("toprock", "Toprock / uprock", "toprock", "Standing steps before the drop: cross-steps, Indian step, uprock battles.", ["toprock", "uprock"]),
    ("six-step", "6-step", "footwork", "The basic circle of footwork on hands and feet.", ["6-step"]),
    ("cc", "CCs", "footwork", "Kick-through footwork drill switching legs under the body.", ["cc"]),
    ("footwork", "Footwork (other)", "footwork", "Other ground-level stepping patterns and floor flows.", ["footwork-other"]),
    ("baby-freeze", "Baby freeze", "freezes", "Side freeze balancing on hands and head with the knee on the elbow.", ["baby-freeze"]),
    ("chair-freeze", "Chair freeze", "freezes", "Freeze on one hand and the head with legs crossed like a chair.", ["chair-freeze"]),
    ("airchair", "Airchair", "freezes", "One-hand freeze with the elbow in the hip and the body tilted up.", ["airchair"]),
    ("elbow-freeze", "Elbow freeze", "freezes", "Freezes balanced on the elbows.", ["elbow-freeze"]),
    ("handstand-freeze", "Handstand & hand freezes", "freezes", "Handstands, one-arm handstands and hand-balance freezes.", ["handstand-freeze", "other:handbalancing"]),
    ("freeze-various", "Freezes (various)", "freezes", "Freeze variations from the motion-capture packs.", []),
    ("flare", "Flare", "power", "The signature power move: legs circle in a wide straddle while the hands alternate as support.", ["flare"]),
    ("windmill", "Windmill", "power", "Rolling across the upper back and chest with the legs in a V.", ["windmill"]),
    ("headspin", "Headspin", "power", "Spinning on the head, legs up.", ["headspin"]),
    ("backspin", "Backspin", "power", "Spinning on the upper back.", ["backspin"]),
    ("swipe", "Swipes", "power", "The chest twists first, the hands trade support, hips and legs whip around.", ["swipe"]),
    ("1990", "1990 / 2000", "power", "Spinning on one hand (1990) or two hands stacked (2000) in a handstand.", ["1990", "2000"]),
    ("halo", "Halo", "power", "Spinning around the head on the side, between headspin and windmill.", ["halo"]),
    ("power-combo", "Power combos", "power", "Several power moves chained together.", ["power-combo"]),
    ("power-misc", "Power moves (other)", "power", "Helicopter, handstand kicks, kick flips from the CMU b-boy capture.", []),
    ("transition", "Transitions", "transitions", "Moving between moves: drops, rolls, getting up.", ["transition"]),
    ("kick-up", "Kick-up", "transitions", "Kipping up from the back to the feet.", ["kick-up"]),
    ("floorwork", "Floorwork & slides", "transitions", "Floorwork, head slides and rolls.", ["other:floorwork", "other:headslide"]),
    ("freestyle-runs", "Breaking freestyle runs", "freestyle", "Whole sets from motion capture — scrub through, every move is in there.", ["other:freestyle", "other:breaking"]),
    ("acro", "Acro", "acro", "Acrobatic moves (flips, handsprings, aerials).", ["acro", "other:flip"]),
    ("tricking", "Tricking", "acro", "Martial-arts kicks with flips and twists.", ["other:tricking", "other:stunts", "other:trampoline"]),
    ("parkour", "Parkour", "parkour", "Vaults, precision jumps, wall work.", ["other:parkour", "other:vault", "other:wall-run", "other:wall-acrobatics", "other:scrambling"]),
    ("lift", "Lifts & acro-partner", "partner", "Partner lifts and acrobatic partner work (3D does not work on these yet: the flyer is lost).", ["lift"]),
    ("partner-dance", "Partner dance", "partner", "Social, ballroom, bachata and contemporary partner dance.", ["partner:social", "partner:contemporary", "partner:ballroom", "partner:bachata"]),
    ("contact-improv", "Contact improvisation", "partner", "Shared weight and rolling point of contact.", ["other:contact-improvisation"]),
    ("contemporary", "Contemporary", "dance", "Contemporary and release-based dance.", ["other:contemporary", "other:improvisation"]),
    ("hip-hop", "Hip-hop, popping & tutting", "dance", "Popping, tutting, hip-hop grooves and voguing.", ["other:hip-hop", "other:popping", "other:tutting", "other:voguing"]),
    ("capoeira", "Capoeira", "dance", "Capoeira movement: macaco, au, ginga.", ["other:capoeira", "other:martial-arts"]),
    ("ecstatic", "Ecstatic & free dance", "dance", "Free-form dance.", ["other:ecstatic-dance", "other:pole-dance", "other:contortion", "other:cyr-wheel"]),
    ("calisthenics", "Calisthenics & handstands", "conditioning", "Strength skills: levers, push-ups, handstand training.", ["other:calisthenics", "other:handstand-conditioning", "other:handstand-push-up", "other:v-up", "other:hanging", "other:bodybuilding-posing"]),
    ("mobility", "Mobility & stretching", "conditioning", "Joint mobility, flexibility and animal flow.", ["other:mobility", "other:stretching", "other:shoulder-mobility", "other:yoga", "other:animal-flow"]),
    ("fitness", "Fitness & movement training", "conditioning", "General conditioning drills.", ["other:fitness", "other:movement-training", "other:movement"]),
    ("flare-conditioning", "Flare conditioning", "conditioning", "The strength drills that make flares possible: V-sits, L-sits, straddle lifts, tuck planche.", []),
    ("climbing", "Climbing & trees", "outdoors", "Climbing, tree climbing, canyoning, slackline.", ["other:climbing", "other:tree-climbing", "other:canyoning", "other:slackline"]),
    ("outdoor-sports", "Board, ski & wheels", "outdoors", "Skiing, skating and other sports.", ["other:skiing", "other:inline-skating"]),
    ("object", "Juggling & objects", "outdoors", "Juggling and object manipulation.", ["other:juggling", "other:object-manipulation"]),
    ("other-sport", "Other", "outdoors", "Boxing, comedy, falls and everything else.", ["other:boxing", "other:physical-comedy", "other:fall", "other:pit-stop"]),
]
MOVE_BY_SLUG = {m[0]: m for m in MOVES}
TAG_TO_MOVE = {t: m[0] for m in MOVES for t in m[4]}
UNSORTED_TAGS = {"other", "other:none", ""}

# Motion-capture reference: (source, file stem in the assets, label, [move slugs])
REF_SOURCES = {
    "mixamo_beta": ("Mixamo", "Clean game animation (Adobe), skinned Beta character, 60 fps"),
    "cmu85": ("CMU mocap", "Optical marker capture of a real b-boy (CMU subject 85), free to use"),
    "aistpp": ("AIST++", "Multi-view reconstruction of pro breakers (Google), CC BY 4.0"),
    "finedance": ("FineDance", "Optical marker capture (ICCV 2023), research use"),
}
REF_SINGLE = [
    ("mixamo_beta", "Flair_beta.glb", "Mixamo · Flair", ["flare"]),
    ("mixamo_beta", "Breakdance 1990_beta.glb", "Mixamo · 1990", ["1990"]),
    ("mixamo_beta", "Breakdance Swipes_beta.glb", "Mixamo · Swipes", ["swipe"]),
    ("mixamo_beta", "Breakdance Footwork 1_beta.glb", "Mixamo · Footwork 1", ["footwork", "six-step"]),
    ("mixamo_beta", "Breakdance Footwork 2_beta.glb", "Mixamo · Footwork 2", ["footwork"]),
    ("mixamo_beta", "Breakdance Footwork 3_beta.glb", "Mixamo · Footwork 3", ["footwork"]),
    ("mixamo_beta", "Breakdance Freezes_beta.glb", "Mixamo · Freezes", ["freeze-various", "baby-freeze"]),
    ("mixamo_beta", "Breakdance Freeze Var 1_beta.glb", "Mixamo · Freeze var 1", ["freeze-various"]),
    ("mixamo_beta", "Breakdance Freeze Var 2_beta.glb", "Mixamo · Freeze var 2", ["freeze-various"]),
    ("mixamo_beta", "Breakdance Freeze Var 3_beta.glb", "Mixamo · Freeze var 3", ["freeze-various"]),
    ("mixamo_beta", "Breakdance Freeze Var 4_beta.glb", "Mixamo · Freeze var 4", ["freeze-various"]),
    ("mixamo_beta", "Breakdance Uprock_beta.glb", "Mixamo · Uprock", ["toprock"]),
    ("mixamo_beta", "Breakdance Ready_beta.glb", "Mixamo · Ready stance", ["toprock"]),
    ("cmu85", "85_04_mannequin.glb", "CMU · Fancy footwork", ["footwork"]),
    ("cmu85", "85_05_mannequin.glb", "CMU · Handstand kicks", ["power-misc", "handstand-freeze"]),
    ("cmu85", "85_06_mannequin.glb", "CMU · Kick flip", ["power-misc", "acro"]),
    ("cmu85", "85_07_mannequin.glb", "CMU · Breakdance (short)", ["power-misc"]),
    ("cmu85", "85_08_mannequin.glb", "CMU · Helicopter", ["power-misc"]),
    ("cmu85", "85_12_mannequin.glb", "CMU · Long break sequence", ["freestyle-runs"]),
    ("cmu85", "85_14_mannequin.glb", "CMU · Break sequence with flips", ["freestyle-runs", "acro"]),
]
# vlm_movetag vocabulary label -> move slug (segments inside freestyle runs)
SEG_TO_MOVE = dict(TAG_TO_MOVE)
SEG_TO_MOVE.update({"uprock": "toprock", "2000": "1990"})

TIMELINE = [
    ("2024-07-02", "The idea, in his notes: \"take a video and use AI to turn it to a 3D model and play it in VR and slow motion\"."),
    ("2026-01/03", "Tutorial Wiki Generator built: a tutorial video becomes steps with per-step clips (the Sambo flare tutorial, FLARE workout)."),
    ("2026-07-04", "3D accuracy stack: GVHMR + Gemini contact gates + contact-clamped refinement; floating support hand fixed on the flare (28 → 2.4 cm)."),
    ("2026-07-05", "Motion-capture reference library: AIST++ (10), FineDance (13), CMU b-boy (7), Mixamo (13); segmented mannequin + move-study viewer."),
    ("2026-07-09", "moves.html v1.1: 22-move vocabulary, Gemini move tags on the AIST++ freestyles."),
    ("2026-07-10/20", "Instagram export parsed (848 saved posts); 695 clips downloaded and tagged on the NAS for $0 (723 tags with the PC's)."),
    ("2026-07-21/27", "Nightly NAS 3D extraction of 63 chosen clips: 52 worked, 11 failed (all three footwork clips among them). $1.54 of GPU."),
    ("2026-07-22", "Flare training plan built from the tutorial clips (AGENTS/desks/dance/flare)."),
    ("2026-08-04", "Dance Mirror: a clip becomes a practice page with loops, speeds and mirror (tutting demo)."),
    ("2026-09-01", "First look at the 52 results: the flare's support hand never reaches the floor; phase traces (rotations, tempo) for all 52."),
    ("2026-09-01", "Dance Mirror lesson #2: Sambo's flare in 7 parts."),
    ("2026-11-17", "This library + the full 3D audit (every extraction checked against its video)."),
]


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "x"


def jload(p, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


# ---------------------------------------------------------------- media

class Media:
    """Places files under OUT/media: hard link when possible, else copy."""

    def __init__(self, out):
        self.out = out
        self.linked = self.copied = 0
        self.linked_bytes = self.copied_bytes = 0

    def put(self, src, rel):
        src = Path(src)
        if not src.exists():
            return None
        dst = self.out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        size = src.stat().st_size
        if dst.exists():
            if dst.stat().st_size == size and os.path.samefile(src, dst):
                self.linked += 1
                self.linked_bytes += size
                return rel
            if dst.stat().st_size == size and dst.stat().st_mtime >= src.stat().st_mtime:
                self.copied += 1
                self.copied_bytes += size
                return rel
            dst.unlink()
        try:
            os.link(src, dst)
            self.linked += 1
            self.linked_bytes += size
        except OSError:
            shutil.copy2(src, dst)
            self.copied += 1
            self.copied_bytes += size
        return rel


# ---------------------------------------------------------------- data

def grade_of(audit, vision, override=None):
    """Final badge: the vision verdict, calibrated against the numbers.

    Calibration (2026-11-17, from a human look at 8 sheets): the vision pass
    is right about floating and about legs reconstructed wrong behind the
    body, but about one grade too harsh where nothing floats and the fit is
    tight (a plank, a macaco, popping). So:
      * broken -> usable when floating < 1 s, fit <= 7.5 % and row-1 pose >= 6/10;
      * usable -> clean when both pose scores >= 7/10, floating < 0.5 s, fit <= 7.5 %;
      * clean -> usable when the numbers show >= 2 s floating or fit > 12 %.
    A per-clip `override` (config "grade_overrides", dated) wins over all.
    """
    if not audit:
        return None, ""
    if override:
        return override["grade"], override["reason"]
    v = vision if vision and vision.get("grade") in ("clean", "usable", "broken") else None
    g = v["grade"] if v else audit.get("draft_grade", "usable")
    reason = (v or {}).get("reason") or "; ".join(audit.get("draft_reasons") or []) or "no visible defects"
    fl, fit = audit.get("float_s", 0) or 0, audit.get("reproj_med", 1) or 1
    p1, p2 = (v or {}).get("pose_match_row1") or 0, (v or {}).get("world_plausible_row2") or 0
    if g == "broken" and fl < 1.0 and fit <= 0.075 and p1 >= 6:
        g = "usable"
        reason += " (kept as usable: nothing floats and the fit to the video is tight)"
    elif g == "usable" and p1 >= 7 and p2 >= 7 and fl < 0.5 and fit <= 0.075:
        g = "clean"
    if g == "clean" and (fl >= 2.0 or fit > 0.12):
        g = "usable"
        reason += " (numbers: " + "; ".join(audit.get("draft_reasons") or []) + ")"
    return g, reason


def tags_of(audit, vision):
    t = set((vision or {}).get("tags") or []) - {"none"}
    if audit:
        if audit.get("float_s", 0) >= 0.5:
            t.add("floating")
        if audit.get("slide_cm_s", 0) > 6:
            t.add("foot_sliding")
        if audit.get("jitter_deg_s2", 0) > 700:
            t.add("jitter")
        if (audit.get("valid_frac") or 1) < 0.6:
            t.add("fast_cuts")
    return sorted(t)


def build_data(cfg):
    tags_doc = jload(cfg["tags"], {})
    raw = tags_doc.get("clips", [])
    by_sc = {}
    for c in raw:
        sc = c.get("shortcode")
        if not sc:
            continue
        prev = by_sc.get(sc)
        if prev is None or (c.get("owner") and not prev.get("owner")):
            by_sc[sc] = c
    posts = {}
    for p in (jload(cfg.get("ig_queue"), {}) or {}).get("posts", []):
        posts[p.get("shortcode")] = p

    results = Path(cfg["results"])
    audit_dir = Path(cfg["audit"])
    overrides = cfg.get("grade_overrides", {})
    clips = []
    for sc, c in sorted(by_sc.items()):
        moves_raw = [m for m in (c.get("moves") or []) if m]
        slugs = []
        for m in moves_raw:
            s = TAG_TO_MOVE.get(m)
            if s is None and m not in UNSORTED_TAGS:
                s = "other-sport" if m.startswith("other:") else None
            if s and s not in slugs:
                slugs.append(s)
        if c.get("partner_or_lift") and "lift" not in slugs and not any(MOVE_BY_SLUG[s][2] == "partner" for s in slugs):
            slugs.append("lift")
        rec = {
            "id": sc,
            "owner": c.get("owner") or "",
            "url": c.get("url") or (posts.get(sc) or {}).get("url") or f"https://www.instagram.com/p/{sc}/",
            "saved": ((posts.get(sc) or {}).get("saved_at") or "")[:10],
            "dur": round(float(c.get("duration_s") or 0), 1),
            "tags": moves_raw,
            "style": c.get("style") or "",
            "desc": c.get("description") or "",
            "worth": c.get("library_worthiness") or "",
            "partner": bool(c.get("partner_or_lift")),
            "moves": slugs,
        }
        mdir = results / sc
        metrics = jload(mdir / f"{sc}_metrics.json")
        if metrics:
            rec["x3d"] = metrics.get("status")
            if metrics.get("status") != "ok":
                rec["x3d_error"] = metrics.get("error")
            audit = jload(audit_dir / f"{sc}_audit.json")
            vision = jload(audit_dir / f"{sc}_vision.json")
            if audit:
                g, reason = grade_of(audit, vision, overrides.get(sc))
                rec["grade"], rec["reason"] = g, reason
                rec["fail_tags"] = tags_of(audit, vision)
                rec["fps"] = audit.get("fps")
                rec["metrics"] = {k: audit.get(k) for k in (
                    "reproj_med", "float_s", "feet_off_s", "slide_cm_s", "jitter_deg_s2",
                    "valid_frac", "n_cycles", "travel_m_per_s", "contact_summary")}
                if vision:
                    rec["vision"] = {k: vision.get(k) for k in ("grade", "tags", "reason", "pose_match_row1", "world_plausible_row2", "error")}
        clips.append(rec)
    return clips


# ---------------------------------------------------------------- html

CSS = """
:root{--bg:#101114;--panel:#1b1d22;--panel2:#23262d;--fg:#e2e5ea;--mut:#959ba5;--teal:#19b3a6;--orange:#ea6b2e;
--clean:#2fbf71;--usable:#e0b43a;--broken:#e5534b;--line:#2c3038}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,"Segoe UI",sans-serif;padding:14px 14px 60px;max-width:1100px;margin:0 auto}
a{color:var(--teal)}
h1{font-size:24px;line-height:1.2}h1 .t{color:var(--teal)}h1 .o{color:var(--orange)}
h2{font-size:18px;margin:26px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:15px;margin:14px 0 6px}
.mut{color:var(--mut)}.small{font-size:12.5px}
nav.top{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:12px;font-size:14px}
nav.top a{text-decoration:none;padding:6px 10px;border:1px solid var(--line);border-radius:18px;color:var(--fg);background:var(--panel)}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;margin:12px 0}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px 12px}
.tile b{display:block;font-size:22px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px;overflow:hidden}
.card img.thumb{width:100%;height:150px;object-fit:cover;border-radius:8px;background:#000;display:block}
.card .ttl{font-weight:600;margin:6px 0 2px}
.btns{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
a.btn{display:inline-block;text-decoration:none;font-size:13px;padding:7px 11px;border-radius:18px;background:var(--panel2);color:var(--fg);border:1px solid #3a3e47}
a.btn.go{border-color:var(--teal);color:#bff3ee}
.badge{display:inline-block;font-size:11.5px;font-weight:700;padding:2px 8px;border-radius:10px;color:#111;text-transform:uppercase;letter-spacing:.03em}
.badge.clean{background:var(--clean)}.badge.usable{background:var(--usable)}.badge.broken{background:var(--broken)}
.badge.failed{background:#777}.badge.ref{background:#7aa7ff}
.chip{display:inline-block;font-size:12px;padding:2px 8px;border-radius:10px;background:var(--panel2);border:1px solid #3a3e47;margin:2px 3px 0 0;color:#cfd3da;text-decoration:none}
.fam{margin-bottom:8px}
.fam a.chip{font-size:13px;padding:4px 10px}
.steps{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px}
video{width:100%;border-radius:8px;background:#000;max-height:60vh}
table{border-collapse:collapse;width:100%;font-size:13px}
td,th{border-bottom:1px solid var(--line);padding:6px 5px;text-align:left;vertical-align:top}
.scroll{overflow-x:auto}
input[type=search],select{background:var(--panel);color:var(--fg);border:1px solid #3a3e47;border-radius:10px;padding:9px 10px;font-size:15px}
.filters{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}
.filters input{flex:1 1 220px}
.note{background:#1d2422;border:1px solid #2d4a45;border-radius:12px;padding:10px 12px;margin:10px 0}
.warn{background:#2a1f1b;border-color:#5a3a2a}
ul.tl{list-style:none;padding:0}ul.tl li{padding:4px 0;border-bottom:1px solid var(--line)}ul.tl b{color:var(--orange);font-weight:600;margin-right:6px}
"""


def page(title, body, depth=0, extra_head=""):
    up = "../" * depth
    nav = (f'<nav class="top"><a href="{up}index.html">⌂ Library</a><a href="{up}clips.html">All clips</a>'
           f'<a href="{up}audit.html">3D audit</a><a href="{up}reference.html">Mocap</a>'
           f'<a href="{up}clips.html#unsorted">Not yet sorted</a></nav>')
    return (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<meta name="robots" content="noindex,nofollow">'
            f'<title>{E(title) if title == "Move Library" else E(title) + " · Move Library"}</title>'
            f'<link rel="icon" href="data:,"><link rel="stylesheet" href="{up}style.css">{extra_head}</head>'
            f'<body>{nav}{body}</body></html>')


def viewer_href(depth, glb, video="", fps=30, title="", t=None, loop=None, back=""):
    q = {"glb": glb, "fps": f"{float(fps or 30):.4f}", "title": title}
    if video:
        q["video"] = video
    if t:
        q["t"] = f"{t:.2f}"
    if loop:
        q["loop"] = f"{loop[0]:.2f},{loop[1]:.2f}"
    if back:
        q["back"] = back
    return "../" * depth + "viewer.html?" + urlencode(q, quote_via=quote)


def badge(g):
    return f'<span class="badge {E(g)}">{E(g)}</span>' if g else ""


def clip_card(c, depth, show3d=True, back=""):
    up = "../" * depth
    thumb = c.get("sheet")
    img = f'<img class="thumb" loading="lazy" src="{up}{E(thumb)}" alt="">' if thumb else ""
    who = f"@{c['owner']}" if c.get("owner") else c["id"]
    tags = "".join(f'<span class="chip">{E(t)}</span>' for t in c["tags"][:4])
    btns = []
    if c.get("video"):
        btns.append(f'<a class="btn" href="{up}{E(c["video"])}">▶ Video</a>')
    if show3d and c.get("glb"):
        btns.append(f'<a class="btn go" href="{E(viewer_href(depth, c["glb"], c.get("video", ""), c.get("fps", 30), (who + " · " + c["id"]) if c.get("owner") else c["id"], back=back))}">3D viewer</a>')
    if c.get("audit_sheet"):
        btns.append(f'<a class="btn" href="{up}{E(c["audit_sheet"])}">3D vs video</a>')
    btns.append(f'<a class="btn" href="{E(c["url"])}" rel="noreferrer">Instagram</a>')
    g = ""
    if c.get("grade"):
        g = f'<div class="small">{badge(c["grade"])} <span class="mut">{E(c.get("reason", ""))}</span></div>'
    elif c.get("x3d") and c["x3d"] != "ok":
        g = f'<div class="small"><span class="badge failed">3D failed</span> <span class="mut">{E(c.get("x3d_error") or "")}</span></div>'
    return (f'<div class="card">{img}<div class="ttl">{E(who)} <span class="mut small">{c["dur"]}s</span></div>'
            f'<div class="small mut">{E(c["desc"])}</div><div>{tags}</div>{g}<div class="btns">{"".join(btns)}</div></div>')


def build_site(cfg):
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    media = Media(out)
    clips = build_data(cfg)
    audit_dir = Path(cfg["audit"])
    assets = Path(cfg["assets"])

    # -- media for clips
    clip_dirs = [Path(p) for p in cfg["clip_dirs"]]
    sheet_dirs = [Path(p) for p in cfg["sheet_dirs"]]
    sheet_index = {}
    for sd in sheet_dirs:
        if sd.exists():
            for f in sd.iterdir():
                if f.suffix == ".jpg" and f.name.endswith("_sheet.jpg"):
                    sheet_index.setdefault(f.name.split("_dur")[0].replace("_sheet.jpg", ""), f)
    for c in clips:
        sc = c["id"]
        for cd in clip_dirs:
            if (cd / f"{sc}.mp4").exists():
                c["video"] = media.put(cd / f"{sc}.mp4", f"media/ig/{sc}.mp4")
                break
        if sc in sheet_index:
            c["sheet"] = media.put(sheet_index[sc], f"media/ig/{sc}_sheet.jpg")
        if c.get("x3d") == "ok":
            g = media.put(audit_dir / f"{sc}_mannequin.glb", f"media/3d/{sc}.glb")
            if g:
                c["glb"] = g
            s = media.put(audit_dir / f"{sc}_audit.jpg", f"media/audit/{sc}_audit.jpg")
            if s:
                c["audit_sheet"] = s

    # -- tutorial reconstructions (earlier PC runs, judged 2026-07-04 by vlm_judge)
    recon = []
    for r in cfg.get("tutorial_recon", []):
        glb = media.put(assets / r["glb"], f"media/3d/{slugify(r['label'])}.glb")
        vid = media.put(assets / r["video"], f"media/3d/{slugify(r['label'])}.mp4") if r.get("video") else None
        if glb:
            recon.append(dict(r, glb_rel=glb, video_rel=vid or ""))

    # -- reference mocap
    refs = []
    for src, fname, label, moves in REF_SINGLE:
        rel = media.put(assets / "reference_library" / src / fname, f"media/ref/{src}/{slugify(Path(fname).stem)}.glb")
        if rel:
            refs.append({"src": src, "label": label, "glb": rel, "moves": moves, "fps": 60 if src != "cmu85" else 60})
    tagdoc = (jload(assets / "reference_library" / "move_tags.json", {}) or {}).get("clips", {})
    for src in ("aistpp", "finedance"):
        d = assets / "reference_library" / src
        if not d.exists():
            continue
        for f in sorted(d.glob("*_mannequin.glb")):
            stem = f.name[:-len("_mannequin.glb")]
            rel = media.put(f, f"media/ref/{src}/{slugify(stem)}.glb")
            segs = (tagdoc.get(stem) or {}).get("segments") or []
            if src == "aistpp":
                kind = "freestyle" if "_sFM_" in stem else "basic"
                label = f"AIST++ · {'advanced freestyle' if kind == 'freestyle' else 'basic break'} {stem.split('_')[-1]} ({stem.split('_')[3]})"
            else:
                label = "FineDance · " + stem.split("_", 1)[1].replace("_", " ")
            mv = sorted({SEG_TO_MOVE[s["move"]] for s in segs if s.get("move") in SEG_TO_MOVE})
            refs.append({"src": src, "label": label, "glb": rel, "moves": ["freestyle-runs"] + mv,
                         "fps": 60 if src == "aistpp" else 30, "segments": segs})

    # -- tutorials (tutorial generator output)
    tutorials = []
    for t in cfg.get("tutorials", []):
        md = assets / t["md"]
        steps = parse_tutorial_md(md) if md.exists() else []
        meta = jload(assets / t["meta"]) if t.get("meta") else None
        if meta and not steps:
            steps = [{"n": s["step_number"], "title": s["label"], "time": f"{s['start_time']}–{s['end_time']}",
                      "desc": s.get("description", ""), "clip": s["clip_filename"]} for s in meta["steps"]]
        for s in steps:
            src = md.parent / s["clip"]
            s["video"] = media.put(src, f"media/tutorials/{slugify(t['title'])}/{slugify(Path(s['clip']).stem)}.mp4") if src.exists() else None
        tutorials.append(dict(t, steps=[s for s in steps if s.get("video")]))
    lessons = []
    for l in cfg.get("lessons", []):
        rel = media.put(assets / l["file"], f"media/lessons/{slugify(l['title'])}.html")
        if rel:
            lessons.append(dict(l, rel=rel))

    # -- viewer
    for f, rel in (("viewer_html", "viewer.html"), ("viewer_bundle", "app.bundle.js")):
        shutil.copy2(cfg[f], out / rel)
    (out / "style.css").write_text(CSS, encoding="utf-8")

    # -- index by move
    by_move = {m[0]: [] for m in MOVES}
    unsorted = []
    for c in clips:
        if not c["moves"]:
            unsorted.append(c)
        for s in c["moves"]:
            by_move.setdefault(s, []).append(c)
    ref_by_move = {}
    for r in refs:
        for s in r["moves"]:
            ref_by_move.setdefault(s, []).append(r)

    (out / "m").mkdir(exist_ok=True)
    move_pages = []
    for slug, name, fam, blurb, _ in MOVES:
        mc = by_move.get(slug, [])
        mr = ref_by_move.get(slug, [])
        tut = [t for t in tutorials if slug in t["moves"]]
        les = [l for l in lessons if slug in l["moves"]]
        rec = [r for r in recon if slug in r["moves"]]
        if not (mc or mr or tut or les or rec):
            continue
        move_pages.append((slug, name, fam, len(mc), len(mr), sum(1 for c in mc if c.get("glb")) + len(rec)))
        (out / "m" / f"{slug}.html").write_text(
            move_page(slug, name, fam, blurb, mc, mr, tut, les, rec), encoding="utf-8")

    # -- browse data + pages
    lite = [{k: c.get(k) for k in ("id", "owner", "url", "dur", "tags", "desc", "worth", "moves", "video", "sheet",
                                    "glb", "fps", "grade", "reason", "x3d", "audit_sheet", "saved", "partner")}
            for c in clips]
    for c in lite:
        c["fam"] = sorted({MOVE_BY_SLUG[s][2] for s in c["moves"]})
    (out / "data").mkdir(exist_ok=True)
    (out / "data" / "clips.json").write_text(json.dumps(lite, separators=(",", ":")), encoding="utf-8")
    (out / "clips.html").write_text(clips_page(move_pages), encoding="utf-8")
    (out / "audit.html").write_text(audit_page(clips, recon), encoding="utf-8")
    (out / "reference.html").write_text(reference_page(refs), encoding="utf-8")
    stats = compute_stats(clips, refs, tutorials, recon, move_pages, unsorted)
    (out / "index.html").write_text(index_page(stats, move_pages, cfg), encoding="utf-8")
    (out / "app.json").write_text(json.dumps({
        "title": "Move Library",
        "description": "Breaking & movement library: his Instagram clips by move, 3D viewers with quality badges, mocap reference, tutorials. Private (tailnet only)."}),
        encoding="utf-8")
    stats["media"] = {"linked_files": media.linked, "linked_mb": round(media.linked_bytes / 1e6, 1),
                      "copied_files": media.copied, "copied_mb": round(media.copied_bytes / 1e6, 1)}
    (out / "data" / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return stats


def parse_tutorial_md(md):
    text = md.read_text(encoding="utf-8")
    steps = []
    for m in re.finditer(r"## Step (\d+): (.+?)\n\*\*(.+?)\*\*\s*\n+!\[\[(.+?)\]\]\s*\n+(.*?)(?=\n---|\Z)", text, re.S):
        steps.append({"n": int(m.group(1)), "title": m.group(2).strip(), "time": m.group(3).strip(),
                      "clip": m.group(4).strip(), "desc": m.group(5).strip()})
    return steps


def compute_stats(clips, refs, tutorials, recon, move_pages, unsorted):
    x = [c for c in clips if c.get("x3d")]
    ok = [c for c in x if c.get("x3d") == "ok"]
    grades = {g: sum(1 for c in ok if c.get("grade") == g) for g in ("clean", "usable", "broken")}
    return {
        "built": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "clips_saved": len(clips),
        "clips_with_video": sum(1 for c in clips if c.get("video")),
        "clips_tagged": sum(1 for c in clips if c["tags"]),
        "clips_sorted": sum(1 for c in clips if c["moves"]),
        "clips_unsorted": len(unsorted),
        "chosen_for_3d": len(x), "3d_ok": len(ok), "3d_failed": len(x) - len(ok),
        "grades": grades, "tutorial_recon": len(recon),
        "mocap_refs": len(refs), "tutorial_steps": sum(len(t["steps"]) for t in tutorials),
        "moves": len(move_pages),
    }


def move_page(slug, name, fam, blurb, mc, mr, tut, les, rec):
    d = 1
    back = f"m/{slug}.html"
    parts = [f'<h1>{E(name)}</h1><div class="mut">{E(FAMILY_NAME.get(fam, fam))} · {E(blurb)}</div>']
    counts = [f"{len(mc)} clips", f"{sum(1 for c in mc if c.get('glb')) + len(rec)} of yours in 3D", f"{len(mr)} motion-capture"]
    parts.append(f'<div class="small mut" style="margin-top:4px">{" · ".join(counts)}</div>')
    for t in tut:
        parts.append(f'<h2>Tutorial: {E(t["title"])}</h2><div class="small mut">{E(t.get("source", ""))}</div><div class="steps">')
        for s in t["steps"]:
            parts.append(f'<div class="card"><div class="ttl">{s["n"]}. {E(s["title"])} <span class="mut small">{E(s.get("time", ""))}</span></div>'
                         f'<video controls playsinline preload="none" src="../{E(s["video"])}"></video>'
                         f'<div class="small mut">{E(s.get("desc", ""))}</div></div>')
        parts.append("</div>")
    for l in les:
        parts.append(f'<div class="note">🪞 <b>Dance Mirror lesson:</b> <a href="../{E(l["rel"])}">{E(l["title"])}</a> — '
                     f'per-part loops, slow speeds, mirror. <span class="mut small">{E(l.get("note", ""))}</span></div>')
    yours = [c for c in mc if c.get("glb")] + []
    if yours or rec:
        parts.append('<h2>Your 3D (from video)</h2><div class="small mut">Badge = how far you can trust it '
                     '(<a href="../audit.html">how it was judged</a>). Open the viewer for slow motion, any angle, A–B loop, mirror.</div><div class="grid">')
        order = {"clean": 0, "usable": 1, "broken": 2}
        for r in rec:
            href = viewer_href(d, r["glb_rel"], r.get("video_rel", ""), r.get("fps", 30), r["label"], back=back)
            parts.append(f'<div class="card"><div class="ttl">{E(r["label"])}</div><div class="small">{badge(r["grade"])} '
                         f'<span class="mut">{E(r["reason"])}</span></div><div class="btns"><a class="btn go" href="{E(href)}">3D viewer</a></div></div>')
        for c in sorted(yours, key=lambda c: order.get(c.get("grade"), 3)):
            parts.append(clip_card(c, d, back=back))
        parts.append("</div>")
    if mr:
        parts.append('<h2>Motion-capture reference</h2><div class="small mut">Real capture or clean animation — '
                     'accurate, no video guesswork. Freestyle runs open at the part where this move happens, looping it.</div><div class="grid">')
        for r in mr:
            segs = [s for s in r.get("segments", []) if SEG_TO_MOVE.get(s.get("move")) == slug]
            src_name = REF_SOURCES.get(r["src"], (r["src"],))[0]
            if segs:
                links = "".join(
                    f'<a class="btn go" href="{E(viewer_href(d, r["glb"], "", r["fps"], r["label"], loop=(s["start_s"], s["end_s"]), back=back))}">'
                    f'{fmt_ts(s["start_s"])}–{fmt_ts(s["end_s"])}</a>' for s in segs[:6])
                parts.append(f'<div class="card"><div class="ttl">{E(r["label"])}</div><div class="small"><span class="badge ref">{E(src_name)}</span> '
                             f'<span class="mut">this move at:</span></div><div class="btns">{links}'
                             f'<a class="btn" href="{E(viewer_href(d, r["glb"], "", r["fps"], r["label"], back=back))}">whole run</a></div></div>')
            else:
                parts.append(f'<div class="card"><div class="ttl">{E(r["label"])}</div><div class="small"><span class="badge ref">{E(src_name)}</span></div>'
                             f'<div class="btns"><a class="btn go" href="{E(viewer_href(d, r["glb"], "", r["fps"], r["label"], back=back))}">3D viewer</a></div></div>')
        parts.append("</div>")
    rest = [c for c in mc if not c.get("glb")]
    if rest:
        parts.append(f'<h2>Clips ({len(rest)} more, not in 3D)</h2><div class="grid">')
        for c in sorted(rest, key=lambda c: ({"yes": 0, "maybe": 1}.get(c["worth"], 2), c["id"])):
            parts.append(clip_card(c, d, show3d=False))
        parts.append("</div>")
    return page(name, "".join(parts), depth=d)


def fmt_ts(sec):
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"


def index_page(st, move_pages, cfg):
    g = st["grades"]
    tiles = [
        (st["clips_saved"], "clips saved", "clips.html"),
        (st["clips_tagged"], "tagged by AI", "clips.html"),
        (st["clips_sorted"], "sorted into a move", "clips.html"),
        (st["chosen_for_3d"], "chosen for 3D", "audit.html"),
        (st["3d_ok"], "turned into 3D", "audit.html"),
        (g["clean"], "clean 3D", "audit.html#clean"),
        (g["usable"], "usable 3D", "audit.html#usable"),
        (g["broken"] + st["3d_failed"], f"broken ({g['broken']}) or failed ({st['3d_failed']})", "audit.html#broken"),
        (st["mocap_refs"], "motion-capture sequences", "reference.html"),
        (st["tutorial_steps"], "tutorial step clips", "m/flare.html"),
        (st["clips_unsorted"], "not yet sorted", "clips.html#unsorted"),
    ]
    t_html = "".join(f'<a class="tile" href="{h}" style="text-decoration:none;color:inherit"><b>{n}</b><span class="mut small">{E(l)}</span></a>'
                     for n, l, h in tiles)
    fam_html = []
    for fslug, fname, fblurb in FAMILIES:
        ms = [m for m in move_pages if m[2] == fslug]
        if not ms:
            continue
        chips = "".join(f'<a class="chip" href="m/{m[0]}.html">{E(m[1])} <span class="mut">{m[3]}{" · 3D " + str(m[5]) if m[5] else ""}{" · mocap " + str(m[4]) if m[4] else ""}</span></a>' for m in ms)
        fam_html.append(f'<div class="fam"><h3>{E(fname)}</h3><div>{chips}</div></div>')
    tl = "".join(f"<li><b>{E(d)}</b>{E(t)}</li>" for d, t in TIMELINE)
    weak = "".join(f"<li>{E(w)}</li>" for w in cfg.get("weak", []))
    body = f"""
<h1><span class="t">Move</span> <span class="o">Library</span></h1>
<p class="mut">Everything the breakdance-coach project made, in one place: the Instagram moves you saved, sorted by move;
your video-to-3D models with an honest quality badge; real motion capture of the same moves beside them; and the tutorial clips.
Tap a move. In any 3D viewer: drag to turn it, ¼× speed, A–B loop, mirror.</p>
<div class="filters"><input id="q" type="search" placeholder="Search moves, people, descriptions…"></div>
<div id="results"></div>
<div class="tiles">{t_html}</div>
<h2>Moves</h2>
<div class="small mut">Number = clips · 3D = your 3D models · mocap = reference sequences.</div>
{''.join(fam_html)}
<h2>What is weak</h2><ul>{weak}</ul>
<div class="note warn small">Private: this holds other people's Instagram videos. It lives on the NAS for you only (tailnet); never publish it.</div>
<h2>What was built when</h2><ul class="tl">{tl}</ul>
<p class="small mut">Built {E(st['built'])} by <code>analyzer_3d/src/move_library_site.py</code> (breakdance-coach). Rebuild by running it again.</p>
<script>
const MOVES={json.dumps([[m[0], m[1], FAMILY_NAME.get(m[2], m[2])] for m in move_pages])};
let CLIPS=null;
const q=document.getElementById('q'),out=document.getElementById('results');
q.addEventListener('input',async()=>{{
  const s=q.value.trim().toLowerCase(); if(!s){{out.innerHTML='';return;}}
  if(!CLIPS) CLIPS=await (await fetch('data/clips.json')).json();
  const mv=MOVES.filter(m=>(m[1]+' '+m[2]).toLowerCase().includes(s)).slice(0,12);
  const cl=CLIPS.filter(c=>(c.owner+' '+c.desc+' '+c.tags.join(' ')+' '+c.id).toLowerCase().includes(s));
  out.innerHTML='<div class="card">'+mv.map(m=>`<a class="chip" href="m/${{m[0]}}.html">${{m[1]}}</a>`).join('')+
   `<div class="small" style="margin-top:6px"><a href="clips.html#q=${{encodeURIComponent(s)}}">${{cl.length}} clips match →</a></div></div>`;
}});
</script>"""
    return page("Move Library", body, depth=0)


def clips_page(move_pages):
    fam_opts = "".join(f'<option value="{k}">{E(n)}</option>' for k, n, _ in FAMILIES)
    move_opts = "".join(f'<option value="{m[0]}">{E(m[1])}</option>' for m in move_pages)
    body = f"""<h1>All clips</h1>
<div class="small mut">Every Instagram clip you saved. Filter by move family, move, quality of the 3D, or the "not yet sorted" shelf.</div>
<div class="filters">
<input id="q" type="search" placeholder="Search…">
<select id="fam"><option value="">All families</option>{fam_opts}<option value="__unsorted">Not yet sorted</option></select>
<select id="mv"><option value="">All moves</option>{move_opts}</select>
<select id="qual"><option value="">Any 3D</option><option value="has3d">Has 3D</option><option value="clean">Clean</option><option value="usable">Usable</option><option value="broken">Broken</option><option value="failed">3D failed</option></select>
<select id="worth"><option value="">Any rating</option><option value="yes">Library-worthy: yes</option><option value="maybe">maybe</option><option value="no">no</option></select>
</div>
<div id="count" class="small mut"></div>
<div id="grid" class="grid"></div>
<div style="text-align:center;margin:16px"><a class="btn" id="more" href="#" style="display:none">Show more</a></div>
<script>
let CLIPS=[],shown=60;
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function vhref(c){{const p=new URLSearchParams({{glb:c.glb,fps:String(c.fps||30),title:(c.owner?'@'+c.owner:c.id),back:'clips.html'}});if(c.video)p.set('video',c.video);return 'viewer.html?'+p.toString();}}
function card(c){{
  let b='';
  if(c.video) b+=`<a class="btn" href="${{esc(c.video)}}">▶ Video</a>`;
  if(c.glb) b+=`<a class="btn go" href="${{esc(vhref(c))}}">3D viewer</a>`;
  if(c.audit_sheet) b+=`<a class="btn" href="${{esc(c.audit_sheet)}}">3D vs video</a>`;
  b+=`<a class="btn" href="${{esc(c.url)}}" rel="noreferrer">Instagram</a>`;
  let g='';
  if(c.grade) g=`<div class="small"><span class="badge ${{c.grade}}">${{c.grade}}</span> <span class="mut">${{esc(c.reason)}}</span></div>`;
  else if(c.x3d&&c.x3d!=='ok') g=`<div class="small"><span class="badge failed">3D failed</span></div>`;
  const mv=c.moves.map(m=>`<a class="chip" href="m/${{m}}.html">${{esc(m)}}</a>`).join('');
  return `<div class="card">${{c.sheet?`<img class="thumb" loading="lazy" src="${{esc(c.sheet)}}" alt="">`:''}}<div class="ttl">${{esc(c.owner?'@'+c.owner:c.id)}} <span class="mut small">${{c.dur}}s</span></div><div class="small mut">${{esc(c.desc)}}</div><div>${{mv||'<span class="chip">not yet sorted</span>'}}</div>${{g}}<div class="btns">${{b}}</div></div>`;
}}
function apply(){{
  const s=$('q').value.trim().toLowerCase(),f=$('fam').value,m=$('mv').value,ql=$('qual').value,w=$('worth').value;
  const r=CLIPS.filter(c=>{{
    if(s&&!(c.owner+' '+c.desc+' '+c.tags.join(' ')+' '+c.id).toLowerCase().includes(s))return false;
    if(f==='__unsorted'&&c.moves.length)return false;
    if(f&&f!=='__unsorted'&&!c.fam.includes(f))return false;
    if(m&&!c.moves.includes(m))return false;
    if(ql==='has3d'&&!c.glb)return false;
    if(ql==='failed'&&!(c.x3d&&c.x3d!=='ok'))return false;
    if(['clean','usable','broken'].includes(ql)&&c.grade!==ql)return false;
    if(w&&c.worth!==w)return false;
    return true;}});
  $('count').textContent=r.length+' clips';
  $('grid').innerHTML=r.slice(0,shown).map(card).join('');
  $('more').style.display=r.length>shown?'':'none';
}}
$('more').onclick=e=>{{e.preventDefault();shown+=60;apply();}};
for(const id of ['q','fam','mv','qual','worth'])$(id).addEventListener('input',()=>{{shown=60;apply();}});
fetch('data/clips.json').then(r=>r.json()).then(d=>{{CLIPS=d;
  const h=decodeURIComponent(location.hash.slice(1));
  if(h==='unsorted')$('fam').value='__unsorted';
  else if(h.startsWith('q='))$('q').value=h.slice(2);
  else if(h.startsWith('move='))$('mv').value=h.slice(5);
  apply();}});
</script>"""
    return page("All clips", body, depth=0)


def audit_page(clips, recon):
    x = [c for c in clips if c.get("x3d")]
    order = {"clean": 0, "usable": 1, "broken": 2}
    rows = []
    for c in sorted(x, key=lambda c: (order.get(c.get("grade"), 3), c["id"])):
        m = c.get("metrics") or {}
        mv = ", ".join(c["moves"]) or "—"
        if c.get("x3d") != "ok":
            rows.append(f'<tr id="{c["id"]}"><td><span class="badge failed">failed</span></td><td>{E(c["id"])}<br><span class="mut small">{E(mv)}</span></td>'
                        f'<td colspan="3" class="small">{E(c.get("x3d_error") or "")}</td><td></td></tr>')
            continue
        sheet = f'<a href="{E(c["audit_sheet"])}">sheet</a>' if c.get("audit_sheet") else ""
        view = f'<a href="{E(viewer_href(0, c["glb"], c.get("video", ""), c.get("fps", 30), c["id"], back="audit.html"))}">3D</a>' if c.get("glb") else ""
        rows.append(f'<tr><td>{badge(c.get("grade"))}</td><td>{E(c["id"])}<br><span class="mut small">{E(mv)}</span></td>'
                    f'<td class="small">{E(c.get("reason", ""))}<br><span class="mut">{E(", ".join(c.get("fail_tags") or []))}</span></td>'
                    f'<td class="small">fit {m.get("reproj_med", 0):.0%} · float {m.get("float_s")}s · slide {m.get("slide_cm_s")} · jitter {m.get("jitter_deg_s2")}</td>'
                    f'<td class="small">{sheet} {view}</td><td></td></tr>')
    for r in recon:
        href = viewer_href(0, r["glb_rel"], r.get("video_rel", ""), r.get("fps", 30), r["label"], back="audit.html")
        rows.append(f'<tr><td>{badge(r["grade"])}</td><td>{E(r["label"])}<br><span class="mut small">tutorial video, contact-refined (July)</span></td>'
                    f'<td class="small">{E(r["reason"])}</td><td class="small">Gemini judge 2026-07-04</td><td class="small"><a href="{E(href)}">3D</a></td><td></td></tr>')
    body = f"""<h1>3D audit</h1>
<div class="note small">Every clip that went through video-to-3D, checked against its video on 2026-11-17.
<b>clean</b> = study it as-is · <b>usable</b> = right move and shape, visible defects · <b>broken</b> = misleading.
Each verdict comes from a vision model looking at the "3D vs video" sheet (top: the video with the 3D skeleton drawn on it;
bottom: the 3D in the world with the floor line), weighed against measured numbers:
<b>fit</b> = how far the 3D joints sit from the joints seen in the video (share of body size; ~4% is as good as it gets),
<b>float</b> = seconds with no body part on the floor for longer than a jump lasts, <b>slide</b> = foot sliding (cm/s), <b>jitter</b> (deg/s²).
The full write-up and the plan to fix it are in the vault: AGENTS/dev/breakdance-coach/docs/2026-11-17 move-library-and-3d-audit.md.</div>
<div class="scroll"><table><tr><th>grade</th><th>clip · moves</th><th>why</th><th>numbers</th><th>look</th><th></th></tr>{''.join(rows)}</table></div>"""
    return page("3D audit", body, depth=0)


def reference_page(refs):
    parts = ['<h1>Motion-capture reference</h1><div class="small mut">Clean 3D of breaking from real capture or animation — '
             'what your own 3D should look like. Each also appears on its move page.</div>']
    for src, (name, desc) in REF_SOURCES.items():
        rs = [r for r in refs if r["src"] == src]
        if not rs:
            continue
        parts.append(f'<h2>{E(name)} <span class="mut small">({len(rs)})</span></h2><div class="small mut">{E(desc)}</div><div class="grid">')
        for r in rs:
            mv = "".join(f'<a class="chip" href="m/{E(m)}.html">{E(MOVE_BY_SLUG[m][1])}</a>' for m in r["moves"] if m in MOVE_BY_SLUG)
            parts.append(f'<div class="card"><div class="ttl">{E(r["label"])}</div><div>{mv}</div>'
                         f'<div class="btns"><a class="btn go" href="{E(viewer_href(0, r["glb"], "", r["fps"], r["label"], back="reference.html"))}">3D viewer</a></div></div>')
        parts.append("</div>")
    return page("Motion-capture reference", "".join(parts), depth=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="JSON with paths (see analyzer_3d/nas_deploy/move_library_site.json)")
    a = ap.parse_args()
    cfg = jload(a.config)
    if not cfg:
        print("bad config", file=sys.stderr)
        return 2
    st = build_site(cfg)
    print(json.dumps(st, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
