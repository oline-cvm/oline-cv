const form = document.getElementById("analyze-form");
const fileInput = document.getElementById("file");
const dropzone = document.getElementById("dropzone");
const dropLabel = document.getElementById("drop-label");
const runBtn = document.getElementById("run-btn");
const demoBtn = document.getElementById("demo-btn");
const statusEl = document.getElementById("status");
const statusText = document.getElementById("status-text");
const preview = document.getElementById("preview");
const stageEmpty = document.getElementById("stage-empty");

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("drag");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag");
  if (e.dataTransfer.files?.[0]) {
    fileInput.files = e.dataTransfer.files;
    dropLabel.textContent = e.dataTransfer.files[0].name;
  }
});
fileInput.addEventListener("change", () => {
  if (fileInput.files?.[0]) dropLabel.textContent = fileInput.files[0].name;
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!fileInput.files?.[0]) {
    dropLabel.textContent = "Pick a video first";
    return;
  }
  const fd = new FormData();
  fd.append("file", fileInput.files[0]);
  fd.append("jersey", document.getElementById("jersey").value || "76");
  const snap = document.getElementById("snap").value;
  if (snap !== "") fd.append("snap_frame", snap);

  setBusy(true, "Uploading film…");
  try {
    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed");
    await pollJob(data.job_id);
  } catch (err) {
    setBusy(false, err.message || "Failed");
    statusEl.hidden = false;
    statusText.textContent = err.message || "Failed";
  }
});

demoBtn.addEventListener("click", async () => {
  setBusy(true, "Loading #76 result…");
  try {
    const res = await fetch("/api/demo");
    const data = await res.json();
    if (!res.ok) throw new Error("No local result yet — run an analysis first");
    renderResult(data);
    setBusy(false, "Loaded");
    statusEl.hidden = true;
  } catch (err) {
    setBusy(false, err.message);
    statusEl.hidden = false;
    statusText.textContent = err.message;
  }
});

async function pollJob(jobId) {
  for (;;) {
    const res = await fetch(`/api/jobs/${jobId}`);
    const job = await res.json();
    statusText.textContent = job.progress || job.status;
    if (job.status === "done") {
      renderResult(job.result);
      setBusy(false, "Done");
      statusEl.hidden = true;
      return;
    }
    if (job.status === "error") {
      throw new Error(job.error || "Analysis failed");
    }
    await sleep(1200);
  }
}

function renderResult(r) {
  document.getElementById("player-title").textContent = `#${r.jersey ?? 76}`;
  document.getElementById("posture-pill").textContent =
    (r.posture_classification || "—").replaceAll("_", " ");

  const ms = r.reaction_time_ms;
  document.getElementById("reaction-ms").textContent =
    ms == null ? "—" : `${Math.round(ms)} ms`;
  document.getElementById("reaction-sub").textContent =
    r.initiated_by && r.initiated_by !== "unknown"
      ? `${r.initiated_by}-first · ${r.reaction_time_frames ?? "—"} frames`
      : "from snap to first move";

  document.getElementById("initiated").textContent = r.initiated_by || "—";
  document.getElementById("snap-out").textContent =
    r.snap_frame == null ? "—" : String(r.snap_frame);
  document.getElementById("knee").textContent = fmtDeg(r.mean_knee_flexion_deg);
  document.getElementById("torso").textContent = fmtDeg(r.mean_torso_angle_deg);
  document.getElementById("hip").textContent =
    r.hip_height_at_lowest == null ? "—" : Number(r.hip_height_at_lowest).toFixed(2);
  document.getElementById("fps").textContent =
    r.video_fps == null ? "—" : Number(r.video_fps).toFixed(1);

  const counts = r.posture_frame_counts || {};
  const bars = document.getElementById("bars");
  const wrap = document.getElementById("posture-bars");
  const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
  bars.innerHTML = "";
  if (Object.keys(counts).length) {
    wrap.hidden = false;
    for (const [label, n] of Object.entries(counts)) {
      const pct = Math.round((n / total) * 100);
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = `<span>${label.replaceAll("_", " ")}</span>
        <div class="bar-track"><div class="bar-fill" style="width:0%"></div></div>
        <span>${pct}%</span>`;
      bars.appendChild(row);
      requestAnimationFrame(() => {
        row.querySelector(".bar-fill").style.width = `${pct}%`;
      });
    }
  } else {
    wrap.hidden = true;
  }

  if (r.overlay_url) {
    preview.src = r.overlay_url;
    preview.classList.add("visible");
    stageEmpty.classList.add("hidden");
    preview.load();
  }
}

function fmtDeg(v) {
  return v == null ? "—" : `${Math.round(v)}°`;
}

function setBusy(busy, msg) {
  runBtn.disabled = busy;
  demoBtn.disabled = busy;
  if (busy) {
    statusEl.hidden = false;
    statusText.textContent = msg || "Working…";
  }
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}
