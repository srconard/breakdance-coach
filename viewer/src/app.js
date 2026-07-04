/* Move-study viewer (quality plan Phase 3+4).
 *
 * Self-contained: the packer (analyzer_3d/src/make_viewer.py) inlines this
 * bundle plus base64 GLB (+ optional source MP4) into one HTML file.
 * Expects window.__VIEWER__ = { title, fps, glbB64, videoB64, videoOffset }.
 */

import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const DATA = window.__VIEWER__;

// ---------------------------------------------------------------- helpers

function b64ToBuffer(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

const $ = (id) => document.getElementById(id);
const fmt = (t) => `${t.toFixed(2)}s`;

// ---------------------------------------------------------------- scene

const canvasWrap = $("scene");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
canvasWrap.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x191b1f);
scene.fog = new THREE.Fog(0x191b1f, 12, 30);

const camera = new THREE.PerspectiveCamera(42, 1, 0.05, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.HemisphereLight(0xf0f4ff, 0x33363c, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.position.set(3, 6, 2);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = sun.shadow.camera.bottom = -4;
sun.shadow.camera.right = sun.shadow.camera.top = 4;
sun.shadow.bias = -0.0004;
scene.add(sun);

const floor = new THREE.Mesh(
  new THREE.CircleGeometry(7, 64).rotateX(-Math.PI / 2),
  new THREE.MeshStandardMaterial({ color: 0x232629, roughness: 0.95 })
);
floor.receiveShadow = true;
scene.add(floor);
const grid = new THREE.GridHelper(14, 28, 0x4a5058, 0x2e3237);
grid.position.y = 0.001;
scene.add(grid);

// ---------------------------------------------------------------- model

const modelRoot = new THREE.Group();
scene.add(modelRoot);

let mixer, actions = [], duration = 0;
let trackables = {};    // name -> Object3D (wrists/ankles for trails)

new GLTFLoader().parse(b64ToBuffer(DATA.glbB64), "", (gltf) => {
  const model = gltf.scene;
  model.traverse((o) => {
    if (o.isMesh) {
      o.castShadow = true;
      o.material.side = THREE.DoubleSide; // survives mirror mode
    }
    if (/^joint_(7|8|20|21)$/.test(o.name)) trackables[o.name] = o;
  });
  modelRoot.add(model);

  mixer = new THREE.AnimationMixer(model);
  for (const clip of gltf.animations) {
    const a = mixer.clipAction(clip);
    a.play();
    actions.push(a);
    duration = Math.max(duration, clip.duration);
  }
  $("scrub").max = duration;
  $("btime").textContent = fmt(duration);
  fitCamera("threequarter");
  seek(0);
});

function motionBounds() {
  const box = new THREE.Box3().setFromObject(modelRoot);
  if (box.isEmpty()) box.set(new THREE.Vector3(-1, 0, -1), new THREE.Vector3(1, 2, 1));
  return box;
}

const PRESETS = {
  front: [0, 0.35, 1],
  side: [1, 0.35, 0],
  top: [0, 1.8, 0.001],
  threequarter: [0.75, 0.45, 0.75],
};

function fitCamera(preset) {
  const box = motionBounds();
  const center = box.getCenter(new THREE.Vector3());
  const r = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1);
  const d = PRESETS[preset] || PRESETS.threequarter;
  const dir = new THREE.Vector3(...d).normalize();
  camera.position.copy(center).addScaledVector(dir, r * 2.6);
  controls.target.copy(center);
  controls.update();
}

// ---------------------------------------------------------------- trails

const TRAIL_LEN = 70;
const TRAIL_COLORS = { joint_20: 0x19b3a6, joint_7: 0x19b3a6, joint_21: 0xea6b2e, joint_8: 0xea6b2e };
let trails = {}, trailsOn = false;

function initTrails() {
  for (const name of Object.keys(trackables)) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(TRAIL_LEN * 3), 3));
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({
      color: TRAIL_COLORS[name] || 0xffffff, transparent: true, opacity: 0.85,
    }));
    line.frustumCulled = false;
    line.visible = trailsOn;
    scene.add(line);
    trails[name] = { line, pts: [] };
  }
}

function updateTrails() {
  if (!trailsOn) return;
  for (const [name, ob] of Object.entries(trackables)) {
    const t = trails[name];
    if (!t) continue;
    const p = new THREE.Vector3();
    ob.getWorldPosition(p);
    t.pts.push(p);
    if (t.pts.length > TRAIL_LEN) t.pts.shift();
    const attr = t.line.geometry.attributes.position;
    for (let i = 0; i < TRAIL_LEN; i++) {
      const q = t.pts[Math.min(i, t.pts.length - 1)] || p;
      attr.setXYZ(i, q.x, q.y, q.z);
    }
    attr.needsUpdate = true;
    t.line.geometry.setDrawRange(0, Math.max(t.pts.length, 1));
  }
}

function clearTrails() {
  for (const t of Object.values(trails)) t.pts.length = 0;
}

// ---------------------------------------------------------------- video

const video = $("vid");
const videoPanel = $("videopanel");
if (DATA.videoB64) {
  const blob = new Blob([b64ToBuffer(DATA.videoB64)], { type: "video/mp4" });
  video.src = URL.createObjectURL(blob);
  video.muted = true;
} else {
  videoPanel.style.display = "none";
  $("togglevideo").style.display = "none";
}

function syncVideo(hard) {
  if (!DATA.videoB64 || videoPanel.style.display === "none") return;
  const target = time + (DATA.videoOffset || 0);
  if (hard || Math.abs(video.currentTime - target) > 0.12) video.currentTime = target;
  video.playbackRate = Math.max(speed, 0.0625);
  if (playing && video.paused) video.play().catch(() => {});
  if (!playing && !video.paused) video.pause();
}

// ---------------------------------------------------------------- transport

let playing = true, time = 0, speed = 1.0;
let loopA = null, loopB = null;

function seek(t) {
  time = Math.max(0, Math.min(t, duration || 0));
  if (mixer) {
    for (const a of actions) a.time = time;
    mixer.update(0);
  }
  $("scrub").value = time;
  $("atime").textContent = fmt(time);
  syncVideo(true);
}

function setPlaying(p) {
  playing = p;
  $("play").textContent = p ? "❚❚" : "▶";
  syncVideo(true);
}

// UI wiring
$("play").onclick = () => setPlaying(!playing);
$("scrub").oninput = (e) => { setPlaying(false); seek(parseFloat(e.target.value)); };
$("speed").oninput = (e) => {
  speed = parseFloat(e.target.value);
  $("speedval").textContent = speed.toFixed(2) + "×";
  syncVideo(true);
};
for (const s of [0.1, 0.25, 0.5, 1]) {
  $("sp" + String(s).replace(".", "")).onclick = () => {
    speed = s; $("speed").value = s; $("speedval").textContent = s.toFixed(2) + "×";
    syncVideo(true);
  };
}
$("seta").onclick = () => { loopA = time; $("abstate").textContent = `A=${fmt(loopA)}${loopB !== null ? " B=" + fmt(loopB) : ""}`; };
$("setb").onclick = () => { loopB = time; $("abstate").textContent = `${loopA !== null ? "A=" + fmt(loopA) + " " : ""}B=${fmt(loopB)}`; };
$("clearab").onclick = () => { loopA = loopB = null; $("abstate").textContent = ""; };
for (const p of ["front", "side", "top", "threequarter"]) $("cam_" + p).onclick = () => fitCamera(p);
$("mirror").onclick = () => {
  modelRoot.scale.x *= -1;
  $("mirror").classList.toggle("on");
  clearTrails();
};
$("trails").onclick = () => {
  trailsOn = !trailsOn;
  $("trails").classList.toggle("on");
  for (const t of Object.values(trails)) t.line.visible = trailsOn;
  clearTrails();
};
$("togglevideo").onclick = () => {
  videoPanel.style.display = videoPanel.style.display === "none" ? "" : "none";
  resize();
};
window.addEventListener("keydown", (e) => {
  if (e.code === "Space") { e.preventDefault(); setPlaying(!playing); }
  if (e.code === "ArrowRight") { setPlaying(false); seek(time + 1 / DATA.fps); }
  if (e.code === "ArrowLeft") { setPlaying(false); seek(time - 1 / DATA.fps); }
});

// deterministic screenshot hook for headless QA: viewer.html?shot=SECONDS
const shotT = new URLSearchParams(location.search).get("shot");

// ---------------------------------------------------------------- loop

function resize() {
  const w = canvasWrap.clientWidth, h = canvasWrap.clientHeight;
  renderer.setSize(w, h, false);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

const clock = new THREE.Clock();
let shotDone = false;

function tick() {
  requestAnimationFrame(tick);
  const dt = clock.getDelta();
  if (mixer && duration > 0) {
    if (shotT !== null && !shotDone) {
      setPlaying(false);
      seek(parseFloat(shotT));
      shotDone = true;
      document.title = "SHOT_READY";
    } else if (playing) {
      time += dt * speed;
      const a = loopA ?? 0;
      const b = (loopB !== null && loopB > a) ? loopB : duration;
      if (time >= b) { time = a; syncVideo(true); }
      if (time < a) time = a;
      for (const ac of actions) ac.time = time;
      mixer.update(0);
      $("scrub").value = time;
      $("atime").textContent = fmt(time);
      syncVideo(false);
    }
    if (Object.keys(trails).length === 0 && Object.keys(trackables).length) initTrails();
    updateTrails();
  }
  controls.update();
  renderer.render(scene, camera);
}
tick();
