# Scene GLB (`--scene`) and the scan → GLB pipeline

## How `--scene` works

```
python -m analyzer_3d.src.make_viewer move.glb --scene space.glb -o out.html
```

`make_viewer.py` base64-embeds the environment GLB into the self-contained
viewer HTML (`sceneB64` in `window.__VIEWER__`). The viewer (`viewer/src/app.js`)
renders it as a **static** environment around the animated mannequin — no
animation is played on it, meshes are `receiveShadow` only (no casting — cheap).
When a scene is present, the synthetic studio floor/grid are hidden and fog /
camera far-plane are pushed out to the scene's bounds. A **Scene** button in the
header toggles it (hiding the scene restores the studio floor). Camera auto-fit
still frames the *dancer* (whole-animation bounds), not the scene.

## Scale / origin contract

glTF standard conventions, assumed everywhere: **1 unit = 1 meter, Y-up**.
Author/export the scene so:

- the ground surface of the dance spot sits at **y = 0**
- the dance spot is **near the origin** (mannequin animates around origin on y=0)
- real-world scale (a doorway ≈ 2 m; if the scan looks dollhouse- or giant-sized
  next to the ~1.7 m mannequin, rescale)

## Scan → GLB path (scanned outdoor spaces)

Goal: "a ghost dancing in a scanned scene." Splat/NeRF captures must be
exported as **textured mesh** — the viewer renders meshes, not splats.

- **Phone apps** (fastest):
  - *Polycam* — LiDAR/photo mode, exports GLTF/GLB directly; real-world scale on
    LiDAR iPhones/iPads.
  - *Scaniverse* — free, LiDAR + splats; mesh export as GLB/OBJ/FBX, metric scale.
  - *Luma AI* — NeRF/splat capture; export "Mesh" as GLTF/OBJ (choose highest
    poly count, then decimate yourself).
- **Drone / photo sets**: *WebODM* (self-hostable) → `odm_texturing/odm_textured_model_geo.obj`
  → convert to GLB with Blender (`File > Import OBJ`, then `Export glTF 2.0`) or
  `obj2gltf`. WebODM outputs are **Z-up** — glTF export from Blender handles the
  Y-up conversion automatically. With GPS/GCPs the scale is metric already.

## Cleanup checklist (Blender, 5 minutes)

1. Import mesh; set origin: move the dance-spot ground point to world origin,
   ground at y=0 (in Blender: z=0 — the glTF exporter converts).
2. Verify scale against a known dimension; `S` to correct.
3. **Decimate** — viewer embeds the GLB base64 in one HTML (~+33% size); target
   **≤ 25 MB / ≤ 500k tris** (Decimate modifier, ratio 0.1–0.5 is usually fine
   for photogrammetry). Texture 2–4k JPEG, not PNG.
4. Delete floating debris / sky shells; flip inverted normals
   (`Mesh > Normals > Recalculate Outside`).
5. Export glTF 2.0 (.glb), +Y up (default). Test:
   `python -m analyzer_3d.src.make_viewer <move.glb> --scene <scan.glb>`.

QA reference: `output/reference_library/qa/make_test_scene.py` generates the
synthetic `test_scene.glb` (16 m slab, crates, walls, ramp) used by
`scene_demo_viewer.html`.
