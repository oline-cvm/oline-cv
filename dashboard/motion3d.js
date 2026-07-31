const ARRAY_MEANING = {
  frame_indices: "source frame index in the original film",
  timestamps: "seconds from clip start",
  pose_cam: "SMPL axis-angle, 24 joints, root in camera space",
  pose_world: "same body pose, root in gravity-aligned world space",
  betas: "SMPL shape parameters",
  trans_cam: "root translation in camera space (pelvis offset removed)",
  trans_world: "root translation in gravity-aligned world space",
  contact: "per-foot contact probability (4 values)",
  joints_cam: "regressed 3D joints in camera space",
  verts_cam: "SMPL vertices in camera space",
  keypoints_2d: "ViTPose 2D keypoints on the associated box",
  bbox_cxcys: "WHAM crop box as [cx, cy, scale]",
  frame_confidence: "association confidence for this frame",
  interpolated: "1 when the box was bridged rather than observed",
};

const fmt = (v, d = 2) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));

function stat(k, v, note, cls) {
  return `<div class="stat"><div class="k">${k}</div>
    <div class="v ${cls || ""}">${v}</div>
    ${note ? `<div class="n">${note}</div>` : ""}</div>`;
}

async function load() {
  const res = await fetch("/api/motion3d/footage");
  if (!res.ok) {
    document.getElementById("hdr-sub").textContent = "no reconstruction found";
    return;
  }
  const d = await res.json();
  const meta = d.metadata || {};
  const a = meta.association || {};
  const w = d.world || {};

  document.getElementById("hdr-sub").innerHTML =
    `jersey <b>${d.target?.jersey ?? "?"}</b> · track ${d.target?.target_id ?? "?"} ·
     frames ${meta.frame_range?.join("–") ?? "?"} · ${fmt(meta.fps, 0)} fps`;

  const grounded = meta.wham?.world_grounded;
  document.getElementById("stats").innerHTML = [
    stat("status", meta.status ?? "—", `${fmt(meta.runtime_seconds, 0)}s runtime`,
         meta.status === "ok" ? "ok" : "warn"),
    stat("frames", meta.frame_count ?? "—",
         `${a.observed ?? "?"} observed · ${a.bridged ?? 0} bridged`),
    stat("world grounded", grounded ? "yes" : "no",
         grounded ? "DPVO camera trajectory used" : "no camera compensation", grounded ? "ok" : "bad"),
    stat("gravity alignment", `${fmt(w.world_up_mean, 1)}°`,
         `std ${fmt(w.world_up_std, 2)}° from +Y`, w.gravity_ok ? "ok" : "bad"),
    stat("association", `${fmt(a.mean_confidence, 3)}`,
         `mean conf · min ${fmt(a.min_confidence, 3)}`),
    stat("mean IoU", `${fmt(a.mean_iou, 3)}`, `min ${fmt(a.min_iou, 3)}`),
    stat("candidates / frame", fmt(a.mean_candidates, 1), "people the detector saw"),
    stat("horizontal travel", `${fmt(w.path_length_m)} m`,
         `net ${fmt(w.net_displacement_m)} m · peak ${fmt(w.peak_speed_ms)} m/s`),
  ].join("");

  document.getElementById("axis-note").innerHTML =
    `Gravity check: the body's up axis sits <b>${fmt(w.world_up_mean, 1)}°</b> from +Y in world space
     (std ${fmt(w.world_up_std, 2)}°), versus <b>${fmt(w.cam_up_mean, 1)}°</b> in camera space —
     the y-down flip between the two conventions. Low variance means the trajectory is
     gravity-aligned rather than tumbling with the camera.`;

  const flagged = [];
  if (a.bridged_frames?.length) flagged.push(`bridged: ${a.bridged_frames.join(", ")}`);
  if (a.unmatched_frames?.length) flagged.push(`unmatched: ${a.unmatched_frames.join(", ")}`);
  if (a.ambiguous) flagged.push(`${a.ambiguous} ambiguous frame(s)`);
  document.getElementById("flagged-note").textContent =
    flagged.length ? flagged.join("  ·  ") : "nothing flagged";

  const warns = document.getElementById("warns");
  warns.innerHTML = (meta.warnings || []).length
    ? meta.warnings.map((x) => `<li>${x}</li>`).join("")
    : "<li>none</li>";

  const tbody = document.querySelector("#arrays tbody");
  tbody.innerHTML = (d.arrays || [])
    .map((r) => `<tr><td><code>${r.name}</code></td>
      <td class="num">${r.shape.join(" × ")}</td>
      <td>${ARRAY_MEANING[r.name] || ""}</td></tr>`)
    .join("");

  drawConfidence(d.confidence || [], d.interpolated || []);
}

function drawConfidence(conf, interp) {
  const cv = document.getElementById("conf");
  const dpr = window.devicePixelRatio || 1;
  cv.width = cv.clientWidth * dpr;
  cv.height = 150 * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  const W = cv.clientWidth;
  const H = 150;
  const pad = 24;

  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = "#22302a";
  ctx.lineWidth = 1;
  [0, 0.5, 1].forEach((t) => {
    const y = H - pad - t * (H - 2 * pad);
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(W - 6, y);
    ctx.stroke();
    ctx.fillStyle = "#8fa398";
    ctx.font = "10px system-ui";
    ctx.fillText(t.toFixed(1), 4, y + 3);
  });

  if (!conf.length) return;
  const bw = (W - pad - 6) / conf.length;
  conf.forEach((c, i) => {
    ctx.fillStyle = interp[i] ? "#f0aa3c" : "#6edc8a";
    const h = Math.max(1, c * (H - 2 * pad));
    ctx.fillRect(pad + i * bw, H - pad - h, Math.max(1, bw - 0.5), h);
  });
}

load();
