const $ = (id) => document.getElementById(id);

let pickXY = null; // normalized 0–1
let currentResult = null;
let pinnedResult = null;

try {
  const raw = localStorage.getItem("oline_pinned");
  if (raw) pinnedResult = JSON.parse(raw);
} catch (_) {}

$("file").addEventListener("change", () => {
  const f = $("file").files?.[0];
  if (!f) return;
  $("file-label").textContent = f.name;
  loadPickPreview(f);
});

function loadPickPreview(file) {
  const url = URL.createObjectURL(file);
  const v = $("pick-video");
  v.src = url;
  v.onloadeddata = () => {
    v.currentTime = Math.min(0.2, (v.duration || 1) * 0.05);
    $("pick-empty").classList.add("hide");
    v.classList.add("on");
    drawPickOverlay();
    $("pick-hint").textContent = "Click the offensive lineman to lock";
  };
}

$("pick-stage").addEventListener("click", (e) => {
  const v = $("pick-video");
  if (!v.videoWidth) {
    setStatus("Load a film first");
    return;
  }
  const rect = v.getBoundingClientRect();
  // Map click through object-fit: contain letterboxing
  const { x, y } = mapClickToVideo(e.clientX, e.clientY, rect, v.videoWidth, v.videoHeight);
  if (x == null) return;
  pickXY = { x, y };
  $("pick-coord").textContent = `lock ${x.toFixed(3)}, ${y.toFixed(3)}`;
  $("pick-hint").textContent = "OL locked — Run to analyze";
  drawPickOverlay();
});

$("clear-pick").addEventListener("click", () => {
  pickXY = null;
  $("pick-coord").textContent = "auto lock";
  $("pick-hint").textContent = "Load film, then click the OL on the preview";
  drawPickOverlay();
});

function mapClickToVideo(cx, cy, rect, vw, vh) {
  const elW = rect.width;
  const elH = rect.height;
  const scale = Math.min(elW / vw, elH / vh);
  const dispW = vw * scale;
  const dispH = vh * scale;
  const offX = (elW - dispW) / 2;
  const offY = (elH - dispH) / 2;
  const lx = cx - rect.left - offX;
  const ly = cy - rect.top - offY;
  if (lx < 0 || ly < 0 || lx > dispW || ly > dispH) return { x: null, y: null };
  return { x: lx / dispW, y: ly / dispH };
}

function drawPickOverlay() {
  const v = $("pick-video");
  const c = $("pick-canvas");
  if (!v.videoWidth) {
    c.width = c.height = 0;
    return;
  }
  const rect = v.getBoundingClientRect();
  c.width = Math.round(rect.width);
  c.height = Math.round(rect.height);
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  if (!pickXY) return;
  const scale = Math.min(c.width / v.videoWidth, c.height / v.videoHeight);
  const dispW = v.videoWidth * scale;
  const dispH = v.videoHeight * scale;
  const offX = (c.width - dispW) / 2;
  const offY = (c.height - dispH) / 2;
  const px = offX + pickXY.x * dispW;
  const py = offY + pickXY.y * dispH;
  ctx.strokeStyle = "#c4a35a";
  ctx.fillStyle = "rgba(196,163,90,0.35)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(px, py, 14, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(px - 22, py);
  ctx.lineTo(px + 22, py);
  ctx.moveTo(px, py - 22);
  ctx.lineTo(px, py + 22);
  ctx.stroke();
}

window.addEventListener("resize", drawPickOverlay);

$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!$("file").files?.[0]) {
    setStatus("Pick a video first");
    return;
  }
  const fd = new FormData();
  fd.append("file", $("file").files[0]);
  fd.append("jersey", $("jersey").value || "");
  fd.append("play_type", $("play_type").value || "pass");
  const snap = $("snap").value;
  if (snap !== "") fd.append("snap_frame", snap);
  if (pickXY) {
    fd.append("pick_x", String(pickXY.x));
    fd.append("pick_y", String(pickXY.y));
  }

  setBusy(true, "Uploading…");
  try {
    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed");
    await poll(data.job_id);
  } catch (err) {
    setBusy(false, err.message);
  }
});

$("demo").addEventListener("click", async () => {
  setBusy(true, "Loading…");
  try {
    const res = await fetch("/api/demo");
    const data = await res.json();
    if (!res.ok) throw new Error("No local result yet");
    render(data);
    setBusy(false, "");
  } catch (err) {
    setBusy(false, err.message);
  }
});

$("pin-compare").addEventListener("click", () => {
  if (!currentResult) {
    setStatus("Run an analysis first");
    return;
  }
  pinnedResult = currentResult;
  try {
    localStorage.setItem("oline_pinned", JSON.stringify(pinnedResult));
  } catch (_) {}
  setStatus("Pinned as compare A");
  renderCompare();
});

async function poll(id) {
  for (;;) {
    const res = await fetch(`/api/jobs/${id}`);
    const job = await res.json();
    setStatus(job.progress || job.status);
    if (job.status === "done") {
      render(job.result);
      setBusy(false, "");
      return;
    }
    if (job.status === "error") throw new Error(job.error || "Failed");
    await new Promise((r) => setTimeout(r, 1200));
  }
}

function render(r) {
  currentResult = r;
  const label =
    r.jersey != null && r.jersey !== ""
      ? `#${r.jersey}`
      : r.ol_lock?.method === "manual_pick_xy"
        ? "OL (click)"
        : "OL";
  $("title").textContent = label;
  const lock = r.ol_lock?.method
    ? ` · lock ${r.ol_lock.method}${r.ol_lock.jersey != null ? " #" + r.ol_lock.jersey : ""}`
    : "";
  $("subtitle").textContent = `${r.play_type || "pass"} · snap ${r.snap_frame ?? "—"} · ${
    r.video_fps ? Number(r.video_fps).toFixed(0) + " fps" : ""
  }${lock}`;

  const trust = r.trust?.overall;
  if (trust) {
    $("trust-overall").textContent = `${Math.round(trust.score * 100)}% ${trust.level}`;
    $("trust-pill").className = `trust-pill ${trust.level}`;
  } else {
    $("trust-overall").textContent = "—";
    $("trust-pill").className = "trust-pill";
  }

  const ms = r.reaction_time_ms;
  $("reaction").textContent = ms == null ? "—" : `${Math.round(ms)} ms`;
  $("reaction-s").textContent =
    r.initiated_by && r.initiated_by !== "unknown"
      ? `${r.initiated_by}-first · ${r.reaction_time_frames ?? "—"} fr${r.late_off_the_ball ? " · late" : ""}`
      : "snap → first move";

  const cells = [
    ["Posture", pretty(r.posture_classification)],
    ["Knee flex", deg(r.mean_knee_flexion_deg)],
    ["Torso", deg(r.mean_torso_angle_deg)],
    ["Hip low", num(r.hip_height_at_lowest, 2)],
    ["Cadence", r.step_cadence_hz == null ? "—" : `${Number(r.step_cadence_hz).toFixed(1)} Hz`],
    ["Set depth", num(r.set_depth, 2)],
    ["Set width", num(r.set_width, 2)],
    ["Base", num(r.mean_base_width, 2)],
    ["Mirror r", num(r.lateral_match, 2)],
    ["Anchor give", num(r.anchor_give, 2)],
    ["Punch", r.punch_ms == null ? "—" : `${Math.round(r.punch_ms)} ms`],
    ["Sustain", r.engagement_ms == null ? "—" : `${Math.round(r.engagement_ms)} ms`],
  ];
  $("summary-grid").innerHTML = cells
    .map(([k, v]) => `<div class="cell"><div class="k">${k}</div><div class="v">${v}</div></div>`)
    .join("");

  const tags = $("tags");
  tags.innerHTML = "";
  for (const f of r.coach_language || []) {
    const el = document.createElement("span");
    el.className = "tag";
    if (/late|loss|skates|gives|narrow|overset|occluded|slow|early|issue/.test(f)) el.classList.add("bad");
    else if (/quicks|mirror|redirect|anchor|sustain|balance|wide|punch|get_off|movement/.test(f)) el.classList.add("ok");
    else if (/bender|placement|pull|climb/.test(f)) el.classList.add("hot");
    el.textContent = f.replaceAll("_", " ");
    tags.appendChild(el);
  }

  const mods = r.modules || {};
  const trustMods = r.trust?.modules || {};
  const order = [
    ["initial_quicks", "1. Initial quicks / get-off"],
    ["footwork", "2. Footwork"],
    ["mirror_redirect", "3. Mirror / redirect"],
    ["anchor", "4. Anchor"],
    ["body_position", "5. Body position"],
    ["hands", "6. Hands"],
    ["sustain", "7. Sustain"],
    ["point_of_attack", "POA (run)"],
    ["movement_in_space", "Movement in space"],
    ["balance", "Balance"],
  ];
  $("modules").innerHTML = order
    .map(([key, title]) => moduleCard(title, mods[key], trustMods[key]))
    .join("");

  if (r.overlay_url) {
    $("preview").src = r.overlay_url;
    $("preview").classList.add("on");
    $("empty").classList.add("hide");
    $("preview").load();
  }

  renderCompare();
}

function moduleCard(title, data, trust) {
  const trustBar = trust
    ? `<div class="trust-bar ${trust.level}" title="${(trust.reasons || []).join(", ")}">
         <span>Trust ${Math.round(trust.score * 100)}%</span>
         <i style="width:${Math.round(trust.score * 100)}%"></i>
       </div>`
    : "";
  if (!data) {
    return `<article class="mod"><h3>${title}</h3>${trustBar}<p class="na">not computed</p></article>`;
  }
  if (data.available === false) {
    return `<article class="mod"><h3>${title}</h3>${trustBar}<p class="na">${(data.notes || ["unavailable"]).join(", ")}</p></article>`;
  }
  const skip = new Set([
    "available",
    "notes",
    "coach_flags",
    "mode",
    "displacement_direction",
    "posture_frame_counts",
  ]);
  const rows = Object.entries(data)
    .filter(([k, v]) => !skip.has(k) && v !== null && v !== undefined && typeof v !== "object")
    .slice(0, 8)
    .map(([k, v]) => {
      let show = v;
      if (typeof v === "number") show = Number.isInteger(v) ? v : Number(v).toFixed(3);
      if (typeof v === "boolean") show = v ? "yes" : "no";
      return `<dt>${k.replaceAll("_", " ")}</dt><dd>${show}</dd>`;
    })
    .join("");
  const flags = (data.coach_flags || []).join(", ");
  return `<article class="mod"><h3>${title}</h3>${trustBar}<dl>${rows || "<p class='na'>no scalars</p>"}</dl>${
    flags ? `<p class="muted" style="margin:.6rem 0 0;font-size:.75rem">${flags}</p>` : ""
  }</article>`;
}

function renderCompare() {
  const grid = $("compare-grid");
  const a = pinnedResult;
  const b = currentResult;
  if (!a && !b) {
    grid.innerHTML = "";
    $("compare-hint").textContent = "Pin a result, run another rep, then compare";
    return;
  }
  if (a && !b) {
    $("compare-hint").textContent = "A pinned — run another analysis for B";
  } else if (a && b && a.id === b.id) {
    $("compare-hint").textContent = "Pin a different rep to compare against this one";
  } else {
    $("compare-hint").textContent = "A = pinned · B = current";
  }

  const rows = [
    ["Label", (r) => (r.jersey != null ? `#${r.jersey}` : r.ol_lock?.method || "OL")],
    ["Play", (r) => r.play_type || "—"],
    ["Reaction ms", (r) => (r.reaction_time_ms == null ? "—" : Math.round(r.reaction_time_ms))],
    ["Posture", (r) => pretty(r.posture_classification)],
    ["Knee flex", (r) => (r.mean_knee_flexion_deg == null ? "—" : `${Math.round(r.mean_knee_flexion_deg)}°`)],
    ["Set depth", (r) => num(r.set_depth, 2)],
    ["Set width", (r) => num(r.set_width, 2)],
    ["Mirror r", (r) => num(r.lateral_match, 2)],
    ["Anchor give", (r) => num(r.anchor_give, 2)],
    ["Sustain ms", (r) => (r.engagement_ms == null ? "—" : Math.round(r.engagement_ms))],
    ["Trust", (r) => (r.trust?.overall ? `${Math.round(r.trust.overall.score * 100)}%` : "—")],
    ["Flags", (r) => (r.coach_language || []).slice(0, 4).join(", ") || "—"],
  ];

  grid.innerHTML = `
    <div class="cmp-col head">Metric</div>
    <div class="cmp-col head">A ${a ? "" : "(empty)"}</div>
    <div class="cmp-col head">B ${b ? "" : "(empty)"}</div>
    ${rows
      .map(([name, fn]) => {
        const va = a ? fn(a) : "—";
        const vb = b ? fn(b) : "—";
        const delta = numericDelta(va, vb);
        return `<div class="cmp-col">${name}</div>
          <div class="cmp-col">${va}</div>
          <div class="cmp-col">${vb}${delta}</div>`;
      })
      .join("")}
  `;
}

function numericDelta(a, b) {
  const na = parseFloat(String(a).replace(/[^\d.-]/g, ""));
  const nb = parseFloat(String(b).replace(/[^\d.-]/g, ""));
  if (!Number.isFinite(na) || !Number.isFinite(nb) || a === "—" || b === "—") return "";
  const d = nb - na;
  if (Math.abs(d) < 1e-6) return `<span class="delta flat"> =</span>`;
  const sign = d > 0 ? "+" : "";
  return `<span class="delta ${d > 0 ? "up" : "down"}"> ${sign}${d.toFixed(1)}</span>`;
}

function pretty(v) {
  return v ? String(v).replaceAll("_", " ") : "—";
}
function deg(v) {
  return v == null ? "—" : `${Math.round(v)}°`;
}
function num(v, d) {
  return v == null ? "—" : Number(v).toFixed(d);
}
function setBusy(busy, msg) {
  $("run").disabled = busy;
  $("demo").disabled = busy;
  $("pin-compare").disabled = busy;
  setStatus(msg || "");
}
function setStatus(msg) {
  $("status").textContent = msg || "";
}

renderCompare();
