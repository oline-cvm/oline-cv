/**
 * On-field immersive view powered by MediaPipe 3D world landmarks + real film.
 * Data comes from whatever clip was analyzed — nothing clip-hardcoded.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const params = new URLSearchParams(location.search);
const jobId = params.get("job") || "demo";

const boot = document.getElementById("boot");
const bootCard = boot.querySelector(".boot-card");
const bootMsg = document.getElementById("boot-msg");
const bootFill = document.getElementById("boot-fill");
const bootPct = document.getElementById("boot-pct");
const root = document.getElementById("field-root");
const canvas = document.getElementById("field-canvas");
const film = document.getElementById("film");
const scrub = document.getElementById("scrub");
const scrubLabel = document.getElementById("scrub-label");
const cueEl = document.getElementById("field-cue");
const titleEl = document.getElementById("field-title");

function setBootProgress(percent, message, stage) {
  const pct = Math.max(0, Math.min(100, Math.round(percent || 0)));
  bootMsg.textContent = message || "Working…";
  bootPct.textContent = `${pct}%`;
  bootFill.style.width = `${pct}%`;
  const order = ["start", "model", "lift", "ready"];
  const active = stage || (pct < 8 ? "start" : pct < 20 ? "model" : pct < 95 ? "lift" : "ready");
  const ai = order.indexOf(active);
  document.querySelectorAll("#boot-stages li").forEach((li) => {
    const i = order.indexOf(li.getAttribute("data-stage"));
    li.classList.toggle("done", i >= 0 && i < ai);
    li.classList.toggle("active", i === ai);
  });
}

function setBootError(message) {
  bootCard?.classList.add("is-error");
  setBootProgress(100, message, "ready");
}

async function waitForPose3d() {
  const url =
    jobId === "demo" ? "/api/demo/pose3d" : `/api/jobs/${encodeURIComponent(jobId)}/pose3d`;
  setBootProgress(2, "Starting 3D lift…", "start");
  for (;;) {
    const res = await fetch(url);
    const st = await res.json().catch(() => ({}));
    if (!res.ok && st.status !== "running") {
      throw new Error(st.message || st.error || `pose3d ${res.status}`);
    }
    const pct = Number(st.percent) || 0;
    const msg = st.message || "Building 3D pose…";
    let stage = "lift";
    if (pct < 8) stage = "start";
    else if (/load|model|mediapipe/i.test(msg)) stage = "model";
    else if (st.status === "done") stage = "ready";
    setBootProgress(pct, msg, stage);
    if (st.status === "done") return st;
    if (st.status === "error") throw new Error(st.message || "3D lift failed");
    await new Promise((r) => setTimeout(r, 450));
  }
}

/** Soft-body football player: thick limbs + pads driven by MediaPipe joints. */
const LIMBS = [
  // Upper-arm sleeves in jersey color (sits over skin arm mesh)
  { a: "left_shoulder", b: "left_elbow", r0: 0.095, r1: 0.078, mat: "jersey", sleeve: true },
  { a: "right_shoulder", b: "right_elbow", r0: 0.095, r1: 0.078, mat: "jersey", sleeve: true },
  { a: "left_shoulder", b: "left_elbow", r0: 0.078, r1: 0.062, mat: "skin" },
  { a: "left_elbow", b: "left_wrist", r0: 0.06, r1: 0.048, mat: "skin" },
  { a: "right_shoulder", b: "right_elbow", r0: 0.078, r1: 0.062, mat: "skin" },
  { a: "right_elbow", b: "right_wrist", r0: 0.06, r1: 0.048, mat: "skin" },
  { a: "left_hip", b: "left_knee", r0: 0.115, r1: 0.092, mat: "pants" },
  { a: "left_knee", b: "left_ankle", r0: 0.082, r1: 0.055, mat: "pants" },
  { a: "right_hip", b: "right_knee", r0: 0.115, r1: 0.092, mat: "pants" },
  { a: "right_knee", b: "right_ankle", r0: 0.082, r1: 0.055, mat: "pants" },
];

let data = null;
let frames3d = [];
let frameIndex = 0;
let playing = true;
let lastTick = performance.now();
let videoReady = false;
let jerseyNum = null;

const keys = new Set();
window.addEventListener("keydown", (e) => {
  keys.add(e.code);
  if (e.code === "Space") {
    e.preventDefault();
    togglePlay();
  }
});
window.addEventListener("keyup", (e) => keys.delete(e.code));

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.08;

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x6f8a62, 0.016);

const camera = new THREE.PerspectiveCamera(58, window.innerWidth / window.innerHeight, 0.05, 250);
camera.position.set(0, 2.2, 8);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.target.set(0, 1.2, 0);
controls.maxPolarAngle = Math.PI * 0.49;
controls.minDistance = 0.8;
controls.maxDistance = 50;

scene.add(new THREE.HemisphereLight(0xc9ddff, 0x2a3a22, 1.0));
const sun = new THREE.DirectionalLight(0xffe6bf, 1.4);
sun.position.set(10, 20, 8);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.near = 1;
sun.shadow.camera.far = 60;
sun.shadow.camera.left = -20;
sun.shadow.camera.right = 20;
sun.shadow.camera.top = 20;
sun.shadow.camera.bottom = -20;
scene.add(sun);
scene.add(new THREE.AmbientLight(0x405038, 0.4));

buildSky();
buildField();

// Cinema — real analyzed film
const filmTex = new THREE.VideoTexture(film);
filmTex.colorSpace = THREE.SRGBColorSpace;
filmTex.minFilter = THREE.LinearFilter;
filmTex.magFilter = THREE.LinearFilter;
const cinema = new THREE.Mesh(
  new THREE.PlaneGeometry(26, 14.6),
  new THREE.MeshBasicMaterial({ map: filmTex })
);
cinema.position.set(0, 8.6, -15);
scene.add(cinema);
const bezel = new THREE.Mesh(
  new THREE.BoxGeometry(26.5, 15.1, 0.3),
  new THREE.MeshStandardMaterial({ color: 0x141612, roughness: 0.9 })
);
bezel.position.set(0, 8.6, -15.2);
scene.add(bezel);

// Volumetric OL player (not a stick figure)
const athleteRoot = new THREE.Group();
scene.add(athleteRoot);

const mats = {
  skin: new THREE.MeshStandardMaterial({
    color: 0xc68642,
    roughness: 0.72,
    metalness: 0.02,
  }),
  jersey: new THREE.MeshStandardMaterial({
    color: 0x1a3a6e,
    roughness: 0.55,
    metalness: 0.08,
  }),
  pants: new THREE.MeshStandardMaterial({
    color: 0xe8e4d8,
    roughness: 0.65,
    metalness: 0.04,
  }),
  pad: new THREE.MeshStandardMaterial({
    color: 0x163059,
    roughness: 0.45,
    metalness: 0.12,
  }),
  helmet: new THREE.MeshStandardMaterial({
    color: 0x132a4f,
    roughness: 0.28,
    metalness: 0.35,
  }),
  facemask: new THREE.MeshStandardMaterial({
    color: 0xb8bcc4,
    roughness: 0.35,
    metalness: 0.75,
  }),
  cleat: new THREE.MeshStandardMaterial({
    color: 0x1a1a1a,
    roughness: 0.55,
    metalness: 0.1,
  }),
  glove: new THREE.MeshStandardMaterial({
    color: 0x2a2a2a,
    roughness: 0.7,
    metalness: 0.05,
  }),
};

function makeLimb(r0, r1, matKey) {
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(r1, r0, 1, 12), mats[matKey]);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  athleteRoot.add(mesh);
  return mesh;
}

const limbMeshes = LIMBS.map((L) => makeLimb(L.r0, L.r1, L.mat));

const torso = new THREE.Mesh(
  new THREE.CapsuleGeometry(0.22, 0.42, 8, 16),
  mats.jersey
);
torso.castShadow = true;
athleteRoot.add(torso);

const jerseyDecal = makeJerseyNumberMesh();
jerseyDecal.position.set(0, 0.06, 0.2);
torso.add(jerseyDecal);

const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.06, 0.12, 10), mats.skin);
neck.castShadow = true;
athleteRoot.add(neck);

const helmet = new THREE.Mesh(new THREE.SphereGeometry(0.135, 20, 16), mats.helmet);
helmet.castShadow = true;
athleteRoot.add(helmet);
const visor = new THREE.Mesh(
  new THREE.BoxGeometry(0.16, 0.05, 0.08),
  new THREE.MeshStandardMaterial({
    color: 0x111820,
    roughness: 0.15,
    metalness: 0.4,
    transparent: true,
    opacity: 0.85,
  })
);
helmet.add(visor);
visor.position.set(0, 0.01, 0.1);

const facemask = new THREE.Group();
for (let i = -1; i <= 1; i++) {
  const bar = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.16, 6), mats.facemask);
  bar.rotation.z = Math.PI / 2;
  bar.position.set(0, i * 0.035, 0.12);
  facemask.add(bar);
}
helmet.add(facemask);

const shoulderPadL = new THREE.Mesh(new THREE.SphereGeometry(0.12, 14, 12), mats.pad);
const shoulderPadR = new THREE.Mesh(new THREE.SphereGeometry(0.12, 14, 12), mats.pad);
shoulderPadL.scale.set(1.15, 0.85, 1.05);
shoulderPadR.scale.set(1.15, 0.85, 1.05);
shoulderPadL.castShadow = true;
shoulderPadR.castShadow = true;
athleteRoot.add(shoulderPadL, shoulderPadR);

const handL = new THREE.Mesh(new THREE.SphereGeometry(0.055, 12, 10), mats.glove);
const handR = new THREE.Mesh(new THREE.SphereGeometry(0.055, 12, 10), mats.glove);
handL.scale.set(1.1, 0.7, 1.35);
handR.scale.set(1.1, 0.7, 1.35);
athleteRoot.add(handL, handR);

const footL = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.06, 0.22), mats.cleat);
const footR = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.06, 0.22), mats.cleat);
footL.castShadow = true;
footR.castShadow = true;
athleteRoot.add(footL, footR);

const hipPad = new THREE.Mesh(new THREE.SphereGeometry(0.16, 12, 10), mats.pants);
hipPad.scale.set(1.35, 0.55, 0.9);
athleteRoot.add(hipPad);

const ring = new THREE.Mesh(
  new THREE.RingGeometry(0.42, 0.55, 48),
  new THREE.MeshBasicMaterial({
    color: 0xd4b56a,
    transparent: true,
    opacity: 0.55,
    side: THREE.DoubleSide,
  })
);
ring.rotation.x = -Math.PI / 2;
ring.position.y = 0.02;
scene.add(ring);

function makeJerseyNumberMesh() {
  const c = document.createElement("canvas");
  c.width = 128;
  c.height = 128;
  const g = c.getContext("2d");
  g.clearRect(0, 0, 128, 128);
  g.fillStyle = "#f2f4f8";
  g.font = "bold 78px IBM Plex Sans, Arial, sans-serif";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText("OL", 64, 68);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(0.28, 0.28),
    new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false })
  );
  mesh.userData.canvas = c;
  mesh.userData.ctx = g;
  mesh.userData.tex = tex;
  return mesh;
}

function setJerseyNumber(n) {
  jerseyNum = n;
  const g = jerseyDecal.userData.ctx;
  const c = jerseyDecal.userData.canvas;
  g.clearRect(0, 0, c.width, c.height);
  g.fillStyle = "#f2f4f8";
  const label = n != null ? String(n) : "OL";
  g.font = label.length > 2 ? "bold 58px IBM Plex Sans, Arial, sans-serif" : "bold 78px IBM Plex Sans, Arial, sans-serif";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText(label, 64, 68);
  jerseyDecal.userData.tex.needsUpdate = true;
}

function buildSky() {
  const skyMat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    uniforms: {
      top: { value: new THREE.Color(0x6f9fd0) },
      mid: { value: new THREE.Color(0xb8c9a6) },
      bot: { value: new THREE.Color(0x4e6a40) },
    },
    vertexShader: `varying vec3 vW; void main(){ vec4 p=modelMatrix*vec4(position,1.0); vW=p.xyz; gl_Position=projectionMatrix*viewMatrix*p; }`,
    fragmentShader: `uniform vec3 top,mid,bot; varying vec3 vW; void main(){ float h=normalize(vW).y; vec3 c=mix(bot,mid,smoothstep(-0.1,0.2,h)); c=mix(c,top,smoothstep(0.15,0.8,h)); gl_FragColor=vec4(c,1.0); }`,
  });
  scene.add(new THREE.Mesh(new THREE.SphereGeometry(120, 32, 16), skyMat));
}

function makeGrassTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 512;
  const g = c.getContext("2d");
  g.fillStyle = "#356530";
  g.fillRect(0, 0, 512, 512);
  for (let i = 0; i < 9000; i++) {
    g.fillStyle = Math.random() > 0.5 ? "#3f7538" : "#2c5528";
    g.fillRect(Math.random() * 512, Math.random() * 512, 2, 5 + Math.random() * 9);
  }
  for (let y = 0; y < 512; y += 64) {
    g.fillStyle = "rgba(255,255,255,0.035)";
    g.fillRect(0, y, 512, 32);
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(16, 12);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function buildField() {
  const grass = new THREE.Mesh(
    new THREE.PlaneGeometry(56, 40),
    new THREE.MeshStandardMaterial({ map: makeGrassTexture(), roughness: 0.92 })
  );
  grass.rotation.x = -Math.PI / 2;
  grass.receiveShadow = true;
  scene.add(grass);
  const lineMat = new THREE.LineBasicMaterial({ color: 0xf2f5ea });
  for (let z = -15; z <= 15; z += 5) {
    scene.add(
      new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(-24, 0.04, z),
          new THREE.Vector3(24, 0.04, z),
        ]),
        lineMat
      )
    );
  }
  scene.add(
    new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-24, 0.05, 0),
        new THREE.Vector3(24, 0.05, 0),
      ]),
      new THREE.LineBasicMaterial({ color: 0xd4b56a })
    )
  );
}

/** MediaPipe world (x right, y up, z toward camera) → Three.js local athlete space. */
function mpToLocal(j) {
  return new THREE.Vector3(j.x, j.y, -j.z);
}

function mid(a, b) {
  return a.clone().add(b).multiplyScalar(0.5);
}

function placeLimb(mesh, a, b, rScale = 1) {
  const dir = new THREE.Vector3().subVectors(b, a);
  const len = dir.length();
  if (len < 1e-4) {
    mesh.visible = false;
    return;
  }
  mesh.visible = true;
  mesh.scale.set(rScale, len, rScale);
  mesh.position.copy(a).add(b).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
}

function applyPose3d(fr) {
  if (!fr?.joints3d || !fr?.root) return;
  const root = fr.root;
  athleteRoot.position.set(root.x, 0, root.z);

  const J = {};
  for (const [name, j] of Object.entries(fr.joints3d)) {
    J[name] = mpToLocal(j);
  }

  // Yaw whole athlete from shoulder line, then pose limbs in local space
  if (J.left_shoulder && J.right_shoulder) {
    const across = new THREE.Vector3().subVectors(J.right_shoulder, J.left_shoulder);
    across.y = 0;
    if (across.lengthSq() > 1e-6) {
      across.normalize();
      const forward = new THREE.Vector3(-across.z, 0, across.x).normalize();
      athleteRoot.rotation.y = Math.atan2(forward.x, forward.z);
    }
  }

  const invYaw = -athleteRoot.rotation.y;
  const cos = Math.cos(invYaw);
  const sin = Math.sin(invYaw);
  const local = {};
  for (const [name, p] of Object.entries(J)) {
    local[name] = new THREE.Vector3(p.x * cos - p.z * sin, p.y, p.x * sin + p.z * cos);
  }

  LIMBS.forEach((L, i) => {
    const a = local[L.a];
    const b = local[L.b];
    if (!a || !b) {
      limbMeshes[i].visible = false;
      return;
    }
    if (L.sleeve) {
      placeLimb(limbMeshes[i], a, a.clone().lerp(b, 0.55));
    } else if (L.mat === "skin" && (L.a === "left_shoulder" || L.a === "right_shoulder")) {
      // Skin forearm-side of upper arm under the sleeve end
      placeLimb(limbMeshes[i], a.clone().lerp(b, 0.45), b);
    } else {
      placeLimb(limbMeshes[i], a, b);
    }
  });

  const shoulderMid =
    local.left_shoulder && local.right_shoulder
      ? mid(local.left_shoulder, local.right_shoulder)
      : null;
  const hipMid =
    local.left_hip && local.right_hip ? mid(local.left_hip, local.right_hip) : null;

  if (shoulderMid && hipMid) {
    const torsoLen = shoulderMid.distanceTo(hipMid);
    const shoulderWidth = local.left_shoulder.distanceTo(local.right_shoulder);
    torso.visible = true;
    torso.position.copy(mid(shoulderMid, hipMid));
    const base = 0.86;
    torso.scale.set(
      1.05 + Math.min(0.45, shoulderWidth * 0.4),
      Math.max(0.5, torsoLen / base),
      1.1
    );
    torso.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      new THREE.Vector3().subVectors(shoulderMid, hipMid).normalize()
    );
    jerseyDecal.visible = true;
    // Keep number readable when torso is non-uniformly scaled
    jerseyDecal.scale.set(1 / torso.scale.x, 1 / torso.scale.y, 1);
    hipPad.visible = true;
    hipPad.position.copy(hipMid);
  } else {
    torso.visible = false;
    jerseyDecal.visible = false;
    hipPad.visible = false;
  }

  if (shoulderMid && local.nose) {
    neck.visible = true;
    placeLimb(neck, shoulderMid, local.nose, 0.9);
    helmet.visible = true;
    helmet.position.copy(local.nose);
    helmet.position.y += 0.05;
    // Face slightly forward (+Z local) with a nod from neck direction
    const neckDir = new THREE.Vector3().subVectors(local.nose, shoulderMid).normalize();
    helmet.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), neckDir);
  } else if (local.nose) {
    neck.visible = false;
    helmet.visible = true;
    helmet.position.copy(local.nose);
    helmet.position.y += 0.05;
  } else {
    neck.visible = false;
    helmet.visible = false;
  }

  if (local.left_shoulder) {
    shoulderPadL.visible = true;
    shoulderPadL.position.copy(local.left_shoulder);
  } else shoulderPadL.visible = false;
  if (local.right_shoulder) {
    shoulderPadR.visible = true;
    shoulderPadR.position.copy(local.right_shoulder);
  } else shoulderPadR.visible = false;

  if (local.left_wrist) {
    handL.visible = true;
    handL.position.copy(local.left_wrist);
    if (local.left_elbow) {
      handL.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        new THREE.Vector3().subVectors(local.left_wrist, local.left_elbow).normalize()
      );
    }
  } else handL.visible = false;
  if (local.right_wrist) {
    handR.visible = true;
    handR.position.copy(local.right_wrist);
    if (local.right_elbow) {
      handR.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        new THREE.Vector3().subVectors(local.right_wrist, local.right_elbow).normalize()
      );
    }
  } else handR.visible = false;

  placeFoot(footL, local.left_ankle, local.left_knee);
  placeFoot(footR, local.right_ankle, local.right_knee);

  ring.position.x = root.x;
  ring.position.z = root.z;
}

function placeFoot(mesh, ankle, knee) {
  if (!ankle) {
    mesh.visible = false;
    return;
  }
  mesh.visible = true;
  mesh.position.set(ankle.x, Math.max(0.03, Math.min(ankle.y, 0.08)), ankle.z);
  if (knee) {
    const fwd = new THREE.Vector3().subVectors(ankle, knee);
    fwd.y = 0;
    if (fwd.lengthSq() > 1e-6) {
      fwd.normalize();
      mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), fwd);
    }
  }
}

function setView(name) {
  const root = frames3d[Math.floor(frameIndex)]?.root || { x: 0, y: 0, z: 1 };
  if (name === "cinema") {
    camera.position.set(0, 8.8, 5);
    controls.target.set(0, 8.4, -15);
  } else if (name === "pocket") {
    camera.position.set(root.x, 1.7, root.z + 3.2);
    controls.target.set(root.x, 1.25, root.z - 0.5);
  } else if (name === "sideline") {
    camera.position.set(root.x + 6, 2.4, root.z + 2);
    controls.target.set(root.x, 1.2, root.z);
  } else if (name === "los") {
    camera.position.set(root.x + 0.2, 1.55, root.z + 1.4);
    controls.target.set(root.x, 1.2, root.z - 1.5);
  } else if (name === "endzone") {
    camera.position.set(root.x, 2.8, root.z - 8);
    controls.target.set(root.x, 1.4, root.z);
  }
  controls.update();
}

function moveCamera(dt) {
  const speed = 6.5 * dt;
  const forward = new THREE.Vector3();
  camera.getWorldDirection(forward);
  forward.y = 0;
  if (forward.lengthSq() < 1e-6) forward.set(0, 0, -1);
  forward.normalize();
  const right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0, 1, 0)).normalize();
  const delta = new THREE.Vector3();
  if (keys.has("KeyW")) delta.addScaledVector(forward, speed);
  if (keys.has("KeyS")) delta.addScaledVector(forward, -speed);
  if (keys.has("KeyA")) delta.addScaledVector(right, -speed);
  if (keys.has("KeyD")) delta.addScaledVector(right, speed);
  if (keys.has("KeyQ")) delta.y -= speed;
  if (keys.has("KeyE")) delta.y += speed;
  camera.position.add(delta);
  controls.target.add(delta);
}

function togglePlay() {
  playing = !playing;
  if (playing) film.play().catch(() => {});
  else film.pause();
}

function nearestFrame(t) {
  let best = 0;
  let bestD = 1e9;
  for (let i = 0; i < frames3d.length; i++) {
    const d = Math.abs((frames3d[i].t || 0) - t);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

async function load() {
  setBootProgress(1, "Checking analysis…", "start");
  await waitForPose3d();
  setBootProgress(96, "Loading field scene…", "ready");

  const url =
    jobId === "demo" ? "/api/demo/field-data" : `/api/jobs/${encodeURIComponent(jobId)}/field-data`;
  const res = await fetch(url);
  if (!res.ok) {
    setBootError("No analysis for this rep. Run Analyze on a clip first.");
    return;
  }
  data = await res.json();
  frames3d = data.pose3d?.frames || [];
  if (!data.video_url) {
    setBootError("No overlay film. Re-run Analyze.");
    return;
  }
  if (!frames3d.length) {
    const err = data.pose3d?.error ? ` (${data.pose3d.error})` : "";
    setBootError(`3D lift produced no frames${err}. Re-run Analyze and try again.`);
    return;
  }

  const label = data.jersey != null ? `#${data.jersey}` : "OL";
  titleEl.textContent = `${label} — 3D on the field`;
  setJerseyNumber(data.jersey);
  const brief = data.brief || {};
  const top = (brief.fix && brief.fix[0]) || (brief.keep && brief.keep[0]);
  cueEl.innerHTML = top
    ? `<strong>${top.title}</strong><br/>${top.detail}<br/><span style="opacity:.7;font-size:.75rem">Engine: MediaPipe 3D · ${frames3d.length} frames</span>`
    : `MediaPipe 3D pose · ${frames3d.length} frames from your upload`;

  film.src = data.video_url;
  film.load();
  await new Promise((resolve, reject) => {
    film.onloadeddata = () => resolve();
    film.onerror = () => reject(new Error("video load failed"));
  });
  videoReady = true;
  film.pause();
  film.currentTime = frames3d[0].t || 0;

  scrub.max = Math.max(0, frames3d.length - 1);
  setBootProgress(100, "Ready", "ready");
  boot.hidden = true;
  root.hidden = false;
  applyPose3d(frames3d[0]);
  setView("pocket");
  film.play().catch(() => {});
}

scrub.addEventListener("input", () => {
  playing = false;
  film.pause();
  frameIndex = Number(scrub.value) || 0;
  const fr = frames3d[frameIndex];
  if (!fr) return;
  if (Math.abs(film.currentTime - (fr.t || 0)) > 0.05) film.currentTime = fr.t || 0;
  applyPose3d(fr);
  scrubLabel.textContent = `${(fr.t || 0).toFixed(1)}s`;
});

document.getElementById("btn-play").addEventListener("click", togglePlay);
document.querySelectorAll("[data-view]").forEach((btn) => {
  btn.addEventListener("click", () => setView(btn.getAttribute("data-view")));
});

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function tick(now) {
  const dt = Math.min(0.05, (now - lastTick) / 1000);
  lastTick = now;
  moveCamera(dt);
  controls.update();

  if (playing && frames3d.length && videoReady) {
    const t = film.currentTime || 0;
    frameIndex = nearestFrame(t);
    applyPose3d(frames3d[frameIndex]);
    scrub.value = String(frameIndex);
    scrubLabel.textContent = `${t.toFixed(1)}s`;
    if (film.ended || t >= (film.duration || 0) - 0.05) {
      film.currentTime = frames3d[0].t || 0;
      film.play().catch(() => {});
    }
  }

  filmTex.needsUpdate = videoReady;
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

load()
  .then(() => requestAnimationFrame(tick))
  .catch((err) => {
    setBootError(`Could not open 3D field: ${err.message || err}`);
  });
