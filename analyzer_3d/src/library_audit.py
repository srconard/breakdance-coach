"""Move-library audit of the Instagram 3D extractions (2026-11-17).

For every clip under a results tree (the NAS `ig-extract/results/<id>/` layout:
`<id>_smpl_clean.pkl`, `<id>_metrics.json`, `<id>_contacts.json`,
`<id>_phases.json`) this builds what the move library needs and what a
reviewer needs to judge the 3D against its source video:

* `<id>_mannequin.glb` — the segmented study mannequin (Blender, via
  `mannequin.export_mannequin`), streamed by the library's viewer.
* `<id>_audit.jpg` — an audit sheet, 2 rows x N columns at the same instants:
  row 1 = the source frame with the reconstructed 3D skeleton projected onto it
  (GVHMR in-camera parameters + intrinsics; LEFT teal, RIGHT orange), row 2 =
  the world-frame reconstruction seen from the same camera heading,
  gravity-aligned, with the floor line — this is where floating hands, missing
  floor contact, sinking and drift show. Columns are evenly spaced, plus the
  frames where the body is worst "floating" (feet off the floor and nothing
  else near it).
* `<id>_audit.json` — measured numbers (nothing fitted):
    - `reproj_med` — median 2D distance between the projected 3D joints and
      the ViTPose keypoints GVHMR saw (shoulders, elbows, wrists, knees,
      ankles; confidence >= 0.5), as a fraction of the person's bbox size.
      Tells whether the 3D pose matches the picture at all.
    - `float_s` / `float_frac` — seconds (and fraction of the valid span)
      where the LOWEST MESH VERTEX stays > 5 cm off the floor for >= 0.5 s:
      nothing touches the floor for longer than a real jump stays in the air
      (the "floating hands" failure of hand-supported moves). `feet_off_s` =
      seconds with both feet > 12 cm up; `hands_support_floating_s` = the
      overlap (feet up AND floating — the support hand never reaches).
    - cleanup metrics after (`jitter_deg_s2`, `slide_cm_s`), palm contact
      windows, `valid_frac` and rotation cycles from `phases.json`, pelvis
      travel per second (drift), and a draft grade with its reasons.

Pure numpy + PIL + ffmpeg for the sheets; Blender only for the GLB.
CPU-heavy: run it on the PC (PC-first lane), never on the NAS.

Usage:
    python -m analyzer_3d.src.library_audit RESULTS_DIR --clips CLIPS_DIR
        --out OUT_DIR [--only ID ...] [--no-glb] [--no-sheet] [--cols 6]
"""

import argparse
import json
import pickle
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import numpy as np

from .motion_cleanup import PARENTS24, fk, lbs_min_height, rodrigues

LEFT = {1, 4, 7, 10, 13, 16, 18, 20, 22}
TEAL = (25, 179, 166)
ORANGE = (234, 107, 46)
GREY = (200, 204, 210)
RED = (235, 64, 52)
# ViTPose COCO-17 index -> SMPL-24 joint (hips left out: SMPL hip joints sit
# inside the pelvis, COCO hips on the surface — a constant bias, not error).
COCO_TO_SMPL = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 13: 4, 14: 5, 15: 7, 16: 8}
FEET_OFF_M = 0.12      # both feet (ankle/foot joints) this far up = "feet off"
FLOAT_M = 0.05         # lowest mesh vertex this far up = nothing touches the floor
FLOAT_MIN_S = 0.5      # ...for longer than a real jump/flip stays airborne
ROW_H = 400


def load(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def joints(params, rest):
    jw, _, _ = fk(params["body_pose"], params["global_orient"], params["transl"], rest)
    return jw


def project(j_cam, K):
    z = np.clip(j_cam[..., 2:3], 1e-3, None)
    uv = (j_cam[..., :2] / z) @ K[:2, :2].T + K[:2, 2]
    return uv


def valid_mask(ph, n, fps):
    m = np.zeros(n, bool)
    spans = (ph or {}).get("valid_span", {}).get("spans_s") or [[0, n / fps]]
    for a, b in spans:
        m[int(a * fps):int(np.ceil(b * fps)) + 1] = True
    return m


def measure(pkl, ph, contacts, metrics):
    rest = pkl["mesh"]["joints"].astype(np.float64)
    fps = float(pkl["fps"])
    g = pkl["smpl_params_global"]
    jw, G, pos = fk(g["body_pose"], g["global_orient"], g["transl"], rest)
    jc = joints(pkl["smpl_params_incam"], rest)
    n = jw.shape[0]
    K = np.asarray(pkl["K_fullimg"], float)
    uv = project(jc, K)
    kp = np.asarray(pkl["keypoints_2d"], float)
    size = np.asarray(pkl["bbx_xys"], float)[:, 2]
    errs = []
    for ci, si in COCO_TO_SMPL.items():
        ok = kp[:, ci, 2] >= 0.5
        d = np.linalg.norm(uv[:, si] - kp[:, ci, :2], axis=-1) / np.maximum(size, 1)
        errs.append(np.where(ok, d, np.nan))
    errs = np.stack(errs, 1)
    allnan = np.isnan(errs).all(1)
    per_frame = np.full(n, np.nan)
    per_frame[~allnan] = np.nanmedian(errs[~allnan], 1)
    vm = valid_mask(ph, n, fps)

    h = jw[..., 1]
    feet = np.min(h[:, [7, 8, 10, 11]], axis=1)
    lowest = np.asarray(lbs_min_height(G, pos, g["transl"], pkl["mesh"], rest), float)
    support = lowest
    feet_off = (feet > FEET_OFF_M) & vm
    airborne = (lowest > FLOAT_M) & vm
    floating = np.zeros(n, bool)
    i = 0
    while i < n:            # only sustained airborne runs count as floating
        if airborne[i]:
            j = i
            while j < n and airborne[j]:
                j += 1
            if (j - i) / fps >= FLOAT_MIN_S:
                floating[i:j] = True
            i = j
        else:
            i += 1
    pel = jw[:, 0, [0, 2]]
    step = np.linalg.norm(np.diff(pel, axis=0), axis=1)
    travel = float(step.sum())
    dur = n / fps

    cw = (contacts or {}).get("clips", {})
    cdoc = next(iter(cw.values()), {}) if isinstance(cw, dict) else {}
    windows = cdoc.get("windows") or cdoc.get("contact_windows") or []
    palm = [w for w in windows if str(w.get("part", "")).startswith("palm")]
    after = ((metrics or {}).get("cleanup") or {}).get("after") or {}
    summ = (ph or {}).get("summary", {})
    out = {
        "fps": round(fps, 3), "frames": n, "duration_s": round(dur, 2),
        "reproj_med": round(float(np.nanmedian(per_frame[vm])) if np.isfinite(per_frame[vm]).any() else -1, 4),
        "reproj_p90": round(float(np.nanpercentile(per_frame[vm], 90)) if np.isfinite(per_frame[vm]).any() else -1, 4),
        "feet_off_s": round(float(feet_off.sum() / fps), 2),
        "float_s": round(float(floating.sum() / fps), 2),
        "float_frac": round(float(floating.sum() / max(vm.sum(), 1)), 3),
        "hands_support_floating_s": round(float((floating & feet_off).sum() / fps), 2),
        "jitter_deg_s2": round(float(after.get("rot_jitter_deg_s2", -1)), 1),
        "slide_cm_s": round(float(after.get("contact_slide_cm_s", -1)), 2),
        "palm_windows": len(palm),
        "contact_windows": metrics.get("contact_windows") if metrics else None,
        "contact_summary": metrics.get("contact_summary") if metrics else None,
        "valid_frac": summ.get("valid_frac"),
        "n_cycles": summ.get("n_cycles"),
        "inverted_frac": summ.get("inverted_frac"),
        "travel_m_per_s": round(travel / dur, 3),
        "net_displacement_m": round(float(np.linalg.norm(pel[-1] - pel[0])), 2),
    }
    out["draft_grade"], out["draft_reasons"] = draft_grade(out)
    arrays = dict(jw=jw, uv=uv, floating=floating, feet_off=feet_off, support=support,
                  vm=vm, per_frame=per_frame)
    return out, arrays


def draft_grade(m):
    """Numbers-only first guess; the vision pass and a human look decide."""
    reasons = []
    bad = 0
    if m["reproj_med"] > 0.15:
        reasons.append(f"3D pose does not match the video (2D error {m['reproj_med']:.0%} of body size)")
        bad += 2
    elif m["reproj_med"] > 0.09:
        reasons.append(f"loose fit to the video (2D error {m['reproj_med']:.0%})")
        bad += 1
    if m["float_s"] >= 2.0 or m["float_frac"] > 0.15:
        reasons.append(f"floating: {m['float_s']:.1f}s with no body part on the floor")
        bad += 2
    elif m["float_s"] >= 0.5:
        reasons.append(f"briefly floating ({m['float_s']:.1f}s with nothing on the floor)")
        bad += 1
    if m["slide_cm_s"] > 6:
        reasons.append(f"foot sliding {m['slide_cm_s']:.1f} cm/s")
        bad += 1
    if m["jitter_deg_s2"] > 700:
        reasons.append(f"jittery ({m['jitter_deg_s2']:.0f} deg/s²)")
        bad += 1
    if (m.get("valid_frac") or 1) < 0.6:
        reasons.append(f"only {m['valid_frac']:.0%} of the clip is trackable (cuts/close-ups)")
        bad += 1
    grade = "clean" if bad == 0 else ("usable" if bad <= 2 else "broken")
    return grade, reasons


# ------------------------------------------------------------------ sheet

def grab_frame(video, t, height=ROW_H):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
           "-vf", f"scale=-2:{height}", "-f", "image2pipe", "-vcodec", "png", "-"]
    r = subprocess.run(cmd, capture_output=True, timeout=60)
    from PIL import Image
    if r.returncode != 0 or not r.stdout:
        return None
    return Image.open(BytesIO(r.stdout)).convert("RGB")


def draw_skel(draw, pts, width=3, colors=True):
    for i in range(1, 24):
        p = PARENTS24[i]
        col = (TEAL if i in LEFT else ORANGE) if colors else GREY
        draw.line([tuple(pts[p]), tuple(pts[i])], fill=col, width=width)
    for i in (15, 20, 21, 7, 8):
        x, y = pts[i]
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 255, 255))


def world_view_axes(pkl):
    """Unit right/forward vectors (world xz) of the source camera, gravity-aligned."""
    gw = rodrigues(np.asarray(pkl["smpl_params_global"]["global_orient"], float))
    gc = rodrigues(np.asarray(pkl["smpl_params_incam"]["global_orient"], float))
    Rcw = gc @ np.transpose(gw, (0, 2, 1))          # world -> cam, per frame
    R = Rcw[len(Rcw) // 2]
    right = R.T @ np.array([1.0, 0, 0])
    right[1] = 0
    right /= max(np.linalg.norm(right), 1e-6)
    return right


def build_sheet(pkl, arrays, video, out_jpg, cols=6, title=""):
    from PIL import Image, ImageDraw
    fps = float(pkl["fps"])
    n = arrays["jw"].shape[0]
    idx = [int((i + 0.5) * n / cols) for i in range(cols)]
    fl = np.where(arrays["floating"])[0]
    if len(fl):
        worst = fl[np.argsort(-arrays["support"][fl])][:1]
        idx += [int(w) for w in worst if all(abs(int(w) - i) > fps * 0.5 for i in idx)]
    idx = sorted(idx)
    vw, vh = pkl.get("video_wh") or (None, None)

    jw = arrays["jw"]
    right = world_view_axes(pkl)
    xs = jw[..., [0, 2]] @ right[[0, 2]]
    ys = jw[..., 1]
    span_x = np.percentile(xs, 99.5) - np.percentile(xs, 0.5)
    lo_x = float(np.percentile(xs, 0.5))
    top = max(float(np.percentile(ys, 99.5)), 1.0)
    cells = []
    for f in idx:
        t = f / fps
        img = grab_frame(video, t) if video and Path(video).exists() else None
        if img is None:
            img = Image.new("RGB", (225, ROW_H), (40, 40, 40))
        s = img.height / float(vh or img.height)
        d = ImageDraw.Draw(img)
        draw_skel(d, arrays["uv"][f] * s, width=3)
        cw = max(img.width, 225)
        wimg = Image.new("RGB", (cw, ROW_H), (25, 27, 31))
        wd = ImageDraw.Draw(wimg)
        scale = min((cw - 20) / max(span_x, 1.0), (ROW_H - 40) / (top + 0.1))
        floor_y = ROW_H - 20
        wd.line([(0, floor_y), (cw, floor_y)], fill=(120, 126, 135), width=2)
        pts = np.stack([10 + (xs[f] - lo_x) * scale, floor_y - ys[f] * scale], -1)
        draw_skel(wd, pts, width=3)
        sup = arrays["support"][f]
        flag = arrays["floating"][f]
        wd.text((4, 4), f"t={t:.2f}s", fill=GREY)
        wd.text((4, 16), f"lowest body point {sup * 100:.0f} cm", fill=RED if flag else GREY)
        if flag:
            wd.text((4, 28), "FLOATING (nothing on floor)", fill=RED)
        cells.append((img, wimg))
    W = sum(max(a.width, b.width) for a, b in cells)
    sheet = Image.new("RGB", (W, ROW_H * 2 + 22), (16, 17, 20))
    ImageDraw.Draw(sheet).text((6, 4), title, fill=GREY)
    x = 0
    for a, b in cells:
        sheet.paste(a, (x, 22))
        sheet.paste(b, (x, 22 + ROW_H))
        x += max(a.width, b.width)
    sheet.save(out_jpg, quality=82)
    return [round(i / fps, 2) for i in idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--clips", required=True, help="folder of <id>.mp4 source clips")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--no-glb", action="store_true")
    ap.add_argument("--no-sheet", action="store_true")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for d in sorted(p for p in res.iterdir() if p.is_dir()):
        cid = d.name
        if a.only and cid not in a.only:
            continue
        mpath = d / f"{cid}_metrics.json"
        metrics = json.loads(mpath.read_text()) if mpath.exists() else {}
        pkl_path = d / f"{cid}_smpl_clean.pkl"
        if not pkl_path.exists():
            summary[cid] = {"status": metrics.get("status", "missing"), "error": metrics.get("error")}
            print(f"[audit] {cid}: no 3D ({summary[cid]['status']})")
            continue
        pkl = load(pkl_path)
        ph = json.loads((d / f"{cid}_phases.json").read_text()) if (d / f"{cid}_phases.json").exists() else None
        ct = json.loads((d / f"{cid}_contacts.json").read_text()) if (d / f"{cid}_contacts.json").exists() else None
        m, arrays = measure(pkl, ph, ct, metrics)
        m["status"] = "ok"
        video = Path(a.clips) / f"{cid}.mp4"
        if not a.no_sheet:
            m["sheet_times_s"] = build_sheet(pkl, arrays, video, out / f"{cid}_audit.jpg", a.cols,
                                             title=f"{cid}  draft={m['draft_grade']}  reproj={m['reproj_med']:.3f}  "
                                                   f"floating {m['float_s']}s  feet-up {m['feet_off_s']}s")
        if not a.no_glb and not (out / f"{cid}_mannequin.glb").exists():
            from .mannequin import export_mannequin
            try:
                export_mannequin(pkl_path, out / f"{cid}_mannequin.glb")
            except Exception as e:  # keep going; the page shows "no viewer"
                m["glb_error"] = str(e)[:300]
        (out / f"{cid}_audit.json").write_text(json.dumps(m, indent=1))
        summary[cid] = m
        print(f"[audit] {cid}: {m['draft_grade']} reproj={m['reproj_med']} float={m['float_s']}/{m['feet_off_s']}s "
              f"slide={m['slide_cm_s']} jit={m['jitter_deg_s2']} :: {'; '.join(m['draft_reasons'])}")
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
