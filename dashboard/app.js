const $ = (id) => document.getElementById(id);

$("file").addEventListener("change", () => {
  if ($("file").files?.[0]) $("file-label").textContent = $("file").files[0].name;
});

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
  const label =
    r.jersey != null && r.jersey !== ""
      ? `#${r.jersey}`
      : "OL";
  $("title").textContent = label;
  const lock = r.ol_lock?.method
    ? ` · lock ${r.ol_lock.method}${r.ol_lock.jersey != null ? " #" + r.ol_lock.jersey : ""}`
    : "";
  $("subtitle").textContent = `${r.play_type || "pass"} · snap ${r.snap_frame ?? "—"} · ${
    r.video_fps ? Number(r.video_fps).toFixed(0) + " fps" : ""
  }${lock} · full frame`;

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
    .map(([key, title]) => moduleCard(title, mods[key]))
    .join("");

  if (r.overlay_url) {
    $("preview").src = r.overlay_url;
    $("preview").classList.add("on");
    $("empty").classList.add("hide");
    $("preview").load();
  }
}

function moduleCard(title, data) {
  if (!data) {
    return `<article class="mod"><h3>${title}</h3><p class="na">not computed</p></article>`;
  }
  if (data.available === false) {
    return `<article class="mod"><h3>${title}</h3><p class="na">${(data.notes || ["unavailable"]).join(", ")}</p></article>`;
  }
  const skip = new Set(["available", "notes", "coach_flags", "mode", "displacement_direction"]);
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
  return `<article class="mod"><h3>${title}</h3><dl>${rows || "<p class='na'>no scalars</p>"}</dl>${
    flags ? `<p class="muted" style="margin:.6rem 0 0;font-size:.75rem">${flags}</p>` : ""
  }</article>`;
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
  setStatus(msg || "");
}
function setStatus(msg) {
  $("status").textContent = msg || "";
}
