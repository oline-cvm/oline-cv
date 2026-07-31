/**
 * Phase 3: SMPL body in Three.js.
 *
 * Mesh vertices arrive already converted to Three.js coords (Z negated once at
 * bake time). Do not flip axes here — a second flip would mirror the athlete.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const CLIP = new URLSearchParams(location.search).get("clip") || "footage";
const MESH_URL = `/outputs/motion3d/${CLIP}/mesh_threejs.bin`;
const CONTACT_URL = `/outputs/motion3d/${CLIP}/mesh_contact.bin`;
const META_URL = `/api/motion3d/${CLIP}`;

const boot = document.getElementById("boot");
const bootMsg = document.getElementById("boot-msg");
const scrub = document.getElementById("scrub");
const btnPlay = document.getElementById("btn-play");
const timeEl = document.getElementById("time");
const speedEl = document.getElementById("speed");
const pillStatus = document.getElementById("pill-status");
const pillInterp = document.getElementById("pill-interp");
const metaEl = document.getElementById("meta");

function setBoot(msg) {
  bootMsg.textContent = msg;
}

function hideBoot() {
  boot.hidden = true;
}

async function loadMeshPack(url) {
  setBoot("Downloading SMPL mesh…");
  const res = await fetch(url);
  if (!res.ok) throw new Error(`mesh pack missing (${res.status}). Run bake_mesh.py first.`);
  const buf = await res.arrayBuffer();
  const view = new DataView(buf);
  const magic = String.fromCharCode(...new Uint8Array(buf, 0, 8));
  if (magic !== "OLMESH01") throw new Error(`bad mesh magic: ${magic}`);

  let o = 8;
  const nFrames = view.getUint32(o, true); o += 4;
  const nVerts = view.getUint32(o, true); o += 4;
  const nFaces = view.getUint32(o, true); o += 4;
  const fps = view.getFloat32(o, true); o += 4;

  const frameIndices = new Int32Array(buf, o, nFrames); o += nFrames * 4;
  const interpolated = new Uint8Array(buf, o, nFrames); o += nFrames;
  o += (4 - (o % 4)) % 4; // match bake_smpl_mesh padding
  const confidence = new Float32Array(buf, o, nFrames); o += nFrames * 4;
  const faces = new Uint32Array(buf, o, nFaces * 3); o += nFaces * 3 * 4;
  const verts = new Float32Array(buf, o, nFrames * nVerts * 3);

  return { nFrames, nVerts, nFaces, fps, frameIndices, interpolated, confidence, faces, verts };
}

function createScene(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight, false);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;

  const scene = new THREE.Scene();
  // Cool studio backdrop — bright enough to read form, not a black void.
  scene.background = new THREE.Color(0xd8dde4);
  scene.fog = new THREE.Fog(0xd8dde4, 22, 48);

  const camera = new THREE.PerspectiveCamera(40, innerWidth / innerHeight, 0.05, 80);
  camera.position.set(3.2, 1.8, 4.8);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 0.95, 0);
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.minDistance = 1.2;
  controls.maxDistance = 18;

  const ambient = new THREE.AmbientLight(0xffffff, 0.55);
  scene.add(ambient);
  const hemi = new THREE.HemisphereLight(0xffffff, 0xb8c4a8, 0.85);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xfffaf0, 1.55);
  sun.position.set(5, 10, 4);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.near = 0.5;
  sun.shadow.camera.far = 30;
  sun.shadow.camera.left = -6;
  sun.shadow.camera.right = 6;
  sun.shadow.camera.top = 6;
  sun.shadow.camera.bottom = -6;
  sun.shadow.bias = -0.0002;
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0xe8f0ff, 0.65);
  fill.position.set(-6, 4, -3);
  scene.add(fill);
  const kick = new THREE.DirectionalLight(0xffffff, 0.35);
  kick.position.set(0, 3, 6);
  scene.add(kick);

  const ground = new THREE.Mesh(
    new THREE.CircleGeometry(14, 64),
    new THREE.MeshStandardMaterial({
      color: 0x3d8f55,
      roughness: 0.88,
      metalness: 0.0,
    })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);

  const grid = new THREE.GridHelper(16, 32, 0xffffff, 0x2f7a45);
  grid.position.y = 0.002;
  grid.material.transparent = true;
  grid.material.opacity = 0.35;
  scene.add(grid);

  const markMat = new THREE.MeshBasicMaterial({ color: 0xf4f7f2, transparent: true, opacity: 0.85 });
  for (let i = -4; i <= 4; i++) {
    const line = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.012, 11), markMat);
    line.position.set(i * 1.0, 0.006, 0);
    scene.add(line);
  }

  return { renderer, scene, camera, controls };
}

const BODY_COLORS = {
  white: 0xf4f4f2,
  chalk: 0xe8e6e1,
  skin: 0xc9a07a,
  slate: 0x8a93a0,
};

function buildBody(pack, colorHex = BODY_COLORS.white) {
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(pack.nVerts * 3);
  pos.set(pack.verts.subarray(0, pack.nVerts * 3));
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setIndex(new THREE.BufferAttribute(pack.faces, 1));
  geo.computeVertexNormals();

  const mat = new THREE.MeshPhysicalMaterial({
    color: colorHex,
    roughness: 0.42,
    metalness: 0.0,
    clearcoat: 0.28,
    clearcoatRoughness: 0.45,
    sheen: 0.2,
    sheenRoughness: 0.55,
    sheenColor: new THREE.Color(0xffffff),
  });

  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.frustumCulled = false;
  mesh.userData.bodyColor = colorHex;
  mesh.userData.bridgedColor = 0xffd089;
  return mesh;
}

/** Map source frame_index → pack row, for meshes that only cover a contact window. */
function indexByFrame(pack) {
  const map = new Map();
  for (let i = 0; i < pack.nFrames; i++) map.set(pack.frameIndices[i], i);
  return map;
}

function applyFrame(mesh, pack, i) {
  const attr = mesh.geometry.getAttribute("position");
  const src = pack.verts;
  const base = i * pack.nVerts * 3;
  for (let v = 0; v < pack.nVerts * 3; v++) attr.array[v] = src[base + v];
  attr.needsUpdate = true;
  mesh.geometry.computeVertexNormals();
}

function frameCenter(pack, i) {
  const base = i * pack.nVerts * 3;
  let sx = 0, sy = 0, sz = 0;
  for (let v = 0; v < pack.nVerts; v++) {
    sx += pack.verts[base + v * 3];
    sy += pack.verts[base + v * 3 + 1];
    sz += pack.verts[base + v * 3 + 2];
  }
  return new THREE.Vector3(sx / pack.nVerts, sy / pack.nVerts, sz / pack.nVerts);
}

function setCamera(mode, camera, controls, pack, frame) {
  const c = frameCenter(pack, frame);
  controls.target.copy(c);
  if (mode === "side") {
    camera.position.set(c.x + 5.5, c.y + 1.1, c.z);
  } else if (mode === "endzone") {
    camera.position.set(c.x, c.y + 1.4, c.z + 6.2);
  } else if (mode === "reset" || mode === "orbit") {
    camera.position.set(c.x + 3.0, c.y + 1.5, c.z + 4.2);
  }
  controls.update();
}

async function main() {
  try {
    setBoot("Reading reconstruction metadata…");
    let meta = null;
    try {
      const mr = await fetch(META_URL);
      if (mr.ok) meta = await mr.json();
    } catch { /* optional */ }

    const pack = await loadMeshPack(MESH_URL);
    setBoot("Loading contact opponent…");
    let contactPack = null;
    let contactMap = null;
    try {
      const head = await fetch(CONTACT_URL, { method: "HEAD" });
      if (head.ok) {
        contactPack = await loadMeshPack(CONTACT_URL);
        contactMap = indexByFrame(contactPack);
      }
    } catch { /* optional */ }

    setBoot("Building scene…");

    const canvas = document.getElementById("c");
    const { renderer, scene, camera, controls } = createScene(canvas);
    const body = buildBody(pack, BODY_COLORS.white);
    scene.add(body);

    // Contact opponent: darker slate so the two bodies are easy to tell apart.
    let contactBody = null;
    if (contactPack) {
      contactBody = buildBody(contactPack, BODY_COLORS.slate);
      contactBody.visible = false;
      scene.add(contactBody);
    }

    scrub.max = String(pack.nFrames - 1);
    scrub.value = "0";
    let frame = 0;
    let playing = true;
    let accum = 0;
    let last = performance.now();
    let speed = 1;

    const jersey = meta?.target?.jersey ?? "?";
    const extra = contactPack
      ? ` · +contact (${contactPack.nFrames}f)`
      : "";
    metaEl.textContent = `jersey ${jersey} · ${pack.nFrames} frames · ${pack.fps.toFixed(0)} fps · ${pack.nVerts.toLocaleString()} verts${extra}`;
    pillStatus.textContent = meta?.metadata?.wham?.world_grounded ? "world grounded" : "local";
    pillStatus.classList.add(meta?.metadata?.wham?.world_grounded ? "ok" : "warn");

    function showFrame(i) {
      frame = Math.max(0, Math.min(pack.nFrames - 1, i | 0));
      applyFrame(body, pack, frame);
      const srcFrame = pack.frameIndices[frame];

      if (contactBody && contactMap) {
        const ci = contactMap.get(srcFrame);
        if (ci !== undefined) {
          contactBody.visible = true;
          applyFrame(contactBody, contactPack, ci);
        } else {
          contactBody.visible = false;
        }
      }

      const c = frameCenter(pack, frame);
      controls.target.lerp(new THREE.Vector3(c.x, Math.max(0.9, c.y), c.z), 0.12);
      scrub.value = String(frame);
      timeEl.textContent = `${frame + 1} / ${pack.nFrames}`;
      const bridged = pack.interpolated[frame] > 0;
      pillInterp.hidden = !bridged;
      body.material.color.setHex(
        bridged ? body.userData.bridgedColor : body.userData.bodyColor
      );
      btnPlay.textContent = playing ? "❚❚" : "▶";
    }

    showFrame(0);
    setCamera("orbit", camera, controls, pack, 0);
    hideBoot();

    btnPlay.addEventListener("click", () => {
      playing = !playing;
      btnPlay.textContent = playing ? "❚❚" : "▶";
    });
    scrub.addEventListener("input", () => {
      playing = false;
      showFrame(Number(scrub.value));
    });
    speedEl.addEventListener("change", () => {
      speed = Number(speedEl.value) || 1;
    });
    document.querySelectorAll(".side button[data-cam]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".side button[data-cam]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        setCamera(btn.dataset.cam, camera, controls, pack, frame);
      });
    });
    document.querySelectorAll(".side button[data-color]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".side button[data-color]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const key = btn.dataset.color;
        body.userData.bodyColor = BODY_COLORS[key] ?? BODY_COLORS.white;
        if (!(pack.interpolated[frame] > 0)) {
          body.material.color.setHex(body.userData.bodyColor);
        }
      });
    });

    addEventListener("resize", () => {
      camera.aspect = innerWidth / innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(innerWidth, innerHeight, false);
    });

    addEventListener("keydown", (e) => {
      if (e.code === "Space") {
        e.preventDefault();
        playing = !playing;
        btnPlay.textContent = playing ? "❚❚" : "▶";
      } else if (e.code === "ArrowRight") {
        playing = false;
        showFrame(frame + 1);
      } else if (e.code === "ArrowLeft") {
        playing = false;
        showFrame(frame - 1);
      }
    });

    function tick(now) {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      if (playing) {
        accum += dt * speed * pack.fps;
        while (accum >= 1) {
          accum -= 1;
          const next = frame + 1;
          if (next >= pack.nFrames) {
            frame = 0;
          } else {
            frame = next;
          }
          showFrame(frame);
        }
      }
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  } catch (err) {
    console.error(err);
    setBoot(err.message || String(err));
    boot.querySelector(".boot-spinner")?.remove();
    boot.querySelector(".boot-title").textContent = "Could not open 3D replay";
  }
}

main();
