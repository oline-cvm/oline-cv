const $ = (id) => document.getElementById(id);

let currentResult = null;
let pinnedResult = null;

try {
  const raw = localStorage.getItem("oline_pinned");
  if (raw) pinnedResult = JSON.parse(raw);
} catch (_) {}

$("show-advanced").addEventListener("change", () => {
  $("snap-wrap").hidden = !$("show-advanced").checked;
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

function makePicker(ids, hintDefault) {
  const els = {
    stage: $(ids.stage),
    video: $(ids.video),
    canvas: $(ids.canvas),
    empty: $(ids.empty),
    hint: $(ids.hint),
    coord: $(ids.coord),
    clear: $(ids.clear),
  };
  let pick = null;

  function draw() {
    const v = els.video;
    const c = els.canvas;
    if (!v.videoWidth) {
      c.width = c.height = 0;
      return;
    }
    const rect = v.getBoundingClientRect();
    c.width = Math.round(rect.width);
    c.height = Math.round(rect.height);
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    if (!pick) return;
    const scale = Math.min(c.width / v.videoWidth, c.height / v.videoHeight);
    const dispW = v.videoWidth * scale;
    const dispH = v.videoHeight * scale;
    const offX = (c.width - dispW) / 2;
    const offY = (c.height - dispH) / 2;
    const px = offX + pick.x * dispW;
    const py = offY + pick.y * dispH;
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

  function load(file) {
    const url = URL.createObjectURL(file);
    els.video.src = url;
    els.video.onloadeddata = () => {
      els.video.currentTime = Math.min(0.2, (els.video.duration || 1) * 0.05);
      els.empty.classList.add("hide");
      els.video.classList.add("on");
      draw();
      els.hint.textContent = "Tap the tackle to lock (best results)";
    };
  }

  els.stage.addEventListener("click", (e) => {
    const v = els.video;
    if (!v.videoWidth) {
      setStatus("Load a film first");
      return;
    }
    const rect = v.getBoundingClientRect();
    const { x, y } = mapClickToVideo(e.clientX, e.clientY, rect, v.videoWidth, v.videoHeight);
    if (x == null) return;
    pick = { x, y, t: v.currentTime };
    els.coord.textContent = "player locked";
    els.hint.textContent = "Locked — hit Analyze rep";
    draw();
  });

  els.clear.addEventListener("click", () => {
    pick = null;
    els.coord.textContent = "auto lock";
    els.hint.textContent = hintDefault;
    draw();
  });

  return { load, draw, getPick: () => pick };
}

const picker = makePicker(
  {
    stage: "pick-stage",
    video: "pick-video",
    canvas: "pick-canvas",
    empty: "pick-empty",
    hint: "pick-hint",
    coord: "pick-coord",
    clear: "clear-pick",
  },
  "Load film, then tap the tackle"
);

$("file").addEventListener("change", () => {
  const f = $("file").files?.[0];
  if (!f) return;
  $("file-label").textContent = f.name;
  $("pick-box").hidden = false;
  picker.load(f);
});

window.addEventListener("resize", () => {
  picker.draw();
});

$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const file1 = $("file").files?.[0];
  if (!file1) {
    setStatus("Upload a clip first");
    return;
  }

  const fd = new FormData();
  fd.append("jersey", $("jersey").value || "");
  fd.append("play_type", $("play_type").value || "pass");
  const snap = $("snap").value;
  if ($("show-advanced").checked && snap !== "") fd.append("snap_frame", snap);

  fd.append("file", file1);
  const pick1 = picker.getPick();
  if (pick1) {
    fd.append("pick_x", String(pick1.x));
    fd.append("pick_y", String(pick1.y));
    fd.append("pick_t", String(pick1.t ?? 0));
  }

  setBusy(true, "Uploading clip…", 2, "ingest");
  try {
    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed");
    setProgressUI(6, "Queued — starting analysis…", "ingest", []);
    await poll(data.job_id);
  } catch (err) {
    setBusy(false, err.message);
  }
});

$("demo").addEventListener("click", async () => {
  setBusy(true, "Loading…", 10, "ingest");
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
    setStatus("Analyze a rep first");
    return;
  }
  pinnedResult = currentResult;
  try {
    localStorage.setItem("oline_pinned", JSON.stringify(pinnedResult));
  } catch (_) {}
  setStatus("Saved as compare A");
  renderCompare();
});

$("export-pdf").addEventListener("click", () => {
  if (!currentResult) return;
  const url = currentResult.report_url || `/api/jobs/${currentResult.id}/report.pdf`;
  if (currentResult.id === "demo" && !currentResult.report_url) {
    setStatus("PDF available after a full Analyze run");
    return;
  }
  window.open(url, "_blank");
});

$("on-field").addEventListener("click", async () => {
  if (!currentResult) return;
  // Prefer the WHAM SMPL viewer (real mesh). Fall back to the old MediaPipe
  // field page only when no baked mesh exists yet.
  const clip = "footage";
  try {
    const head = await fetch(`/outputs/motion3d/${clip}/mesh_threejs.bin`, { method: "HEAD" });
    if (head.ok) {
      window.open(`/viewer3d?clip=${encodeURIComponent(clip)}`, "_blank");
      return;
    }
  } catch { /* fall through */ }
  const job = currentResult.id === "demo" ? "demo" : currentResult.id;
  window.open(`/field?job=${encodeURIComponent(job)}`, "_blank");
});

async function poll(id) {
  for (;;) {
    const res = await fetch(`/api/jobs/${id}`);
    const job = await res.json();
    const pct = Number.isFinite(job.percent) ? job.percent : null;
    setProgressUI(
      pct,
      job.progress || job.status || "Working…",
      job.stage || "",
      job.stages_done || []
    );
    if (job.status === "done") {
      render(job.result);
      setBusy(false, "");
      return;
    }
    if (job.status === "error") throw new Error(job.error || "Failed");
    await new Promise((r) => setTimeout(r, 700));
  }
}

/** Translate raw analysis into coach-facing language. */
function buildCoachBrief(r) {
  const flags = new Set(r.coach_language || []);
  const mods = r.modules || {};
  const fix = [];
  const keep = [];
  const play = r.play_type === "run" ? "run" : "pass";

  // Get-off / reaction
  const ms = r.reaction_time_ms;
  if (r.late_off_the_ball || flags.has("late_off_the_ball") || (ms != null && ms > 250)) {
    fix.push({
      title: "Get off the ball faster",
      detail:
        ms != null
          ? `First move came ~${Math.round(ms)} ms after snap — push the first step on the ball.`
          : "Looks late off the snap. Cue: move on the ball, not the defender.",
    });
  } else if (ms != null && ms <= 180) {
    keep.push({
      title: "Quick off the ball",
      detail: `First move in ~${Math.round(ms)} ms — that tempo is a win. Keep it.`,
    });
  } else if (ms != null) {
    keep.push({
      title: "Acceptable get-off",
      detail: `First move ~${Math.round(ms)} ms. Solid — chase a tick quicker next rep.`,
    });
  }

  if (r.initiated_by === "hip") {
    fix.push({
      title: "Lead with the feet, not the hips",
      detail: "Hips moved before the feet. Teach a clean first step so the body doesn’t leak early.",
    });
  } else if (r.initiated_by === "foot") {
    keep.push({
      title: "Foot-first start",
      detail: "Feet fired first — good sequence off the snap.",
    });
  }

  // Posture
  const posture = String(r.posture_classification || "");
  if (posture === "waist_bender" || flags.has("waist_bender")) {
    fix.push({
      title: "Stay out of the waist bend",
      detail: "Leaning at the waist. Cue: bend at the knees/ankles, keep the chest over the toes.",
    });
  } else if (posture.includes("balanced") || posture === "balanced") {
    keep.push({
      title: "Balanced posture",
      detail: "Pad level and torso look controlled through the set.",
    });
  } else if (posture && posture !== "unknown") {
    keep.push({
      title: `Posture: ${pretty(posture)}`,
      detail: "Hold this shape longer into contact.",
    });
  }

  // Footwork / set
  const foot = mods.footwork || {};
  if (foot.overset || flags.has("overset")) {
    fix.push({
      title: "Don’t overset",
      detail: "Set got too deep/wide. Shorten the second step — stay square to the rush lane.",
    });
  }
  if (flags.has("narrow_base") || (r.mean_base_width != null && r.mean_base_width < 0.35)) {
    fix.push({
      title: "Widen the base",
      detail: "Feet got tight. Cue: athletic base — feel pressure on the inside of both feet.",
    });
  } else if (r.mean_base_width != null && r.mean_base_width >= 0.45) {
    keep.push({
      title: "Good base width",
      detail: "Feet stayed under the body with room to redirect.",
    });
  }
  if (flags.has("skates") || flags.has("skating")) {
    fix.push({
      title: "Stop skating",
      detail: "Feet are sliding instead of planting. Cue: punch the ground, then redirect.",
    });
  }

  // Mirror / lateral
  const mirror = r.lateral_match;
  if (mirror != null && mirror < 0.35) {
    fix.push({
      title: "Mirror the rusher better",
      detail: "Lateral match to the defender was soft. Stay attached — shuffle with their hips.",
    });
  } else if (mirror != null && mirror >= 0.55) {
    keep.push({
      title: "Strong mirror",
      detail: "Moved with the rusher laterally — that keeps the pocket clean.",
    });
  }

  // Anchor
  const give = r.anchor_give;
  if (give != null && give > 0.18) {
    fix.push({
      title: "Anchor — stop giving ground",
      detail: "Hips slid back after contact. Cue: drop the hips, stay connected, don’t catch high.",
    });
  } else if (give != null && give <= 0.1) {
    keep.push({
      title: "Firm anchor",
      detail: "Held ground after contact — pocket stayed put.",
    });
  }

  // Hands / punch
  const punch = r.punch_ms;
  if (punch != null && punch > 400) {
    fix.push({
      title: "Get hands on sooner",
      detail: `Punch landed late (~${Math.round(punch)} ms). Strike on arrival — don’t let them into your chest.`,
    });
  } else if (punch != null && punch <= 280) {
    keep.push({
      title: "Quick hands",
      detail: `Hands got there in ~${Math.round(punch)} ms — keep striking on time.`,
    });
  }
  if (flags.has("early_hands") || flags.has("early_punch")) {
    fix.push({
      title: "Don’t punch air",
      detail: "Hands left early. Time the strike to contact — independent hands, not arm swimming.",
    });
  }

  // Sustain
  const engage = r.engagement_ms;
  if (flags.has("early_disengage") || (engage != null && engage < 400 && play === "pass")) {
    fix.push({
      title: "Finish the block longer",
      detail: "Came off too early. Stay attached through the whistle — drive feet after contact.",
    });
  } else if (engage != null && engage >= 700) {
    keep.push({
      title: "Sustained engagement",
      detail: "Stayed on the block — that’s how pockets hold up.",
    });
  }

  // Flag fallbacks into readable cues
  const FLAG_MAP = {
    late_off_the_ball: null, // already handled
    waist_bender: null,
    overset: null,
    narrow_base: null,
    skates: null,
    skating: null,
    early_disengage: null,
    early_hands: null,
    early_punch: null,
    loss_of_leverage: {
      title: "Regain leverage",
      detail: "Lost pad level / leverage. Sink the hips and get under their pads.",
      bucket: "fix",
    },
    high_pad_level: {
      title: "Get lower",
      detail: "Pad level climbed. Cue: sit in the stance and strike up through the breastplate.",
      bucket: "fix",
    },
    balance_issue: {
      title: "Clean up balance",
      detail: "Weight looked outside the base. Keep the center over the middle of the stance.",
      bucket: "fix",
    },
    occluded: {
      title: "Film was cluttered",
      detail: "Contact/occlusion made part of the read fuzzy — still trust the clear cues above.",
      bucket: "fix",
    },
  };
  for (const f of flags) {
    const mapped = FLAG_MAP[f];
    if (!mapped) continue;
    const list = mapped.bucket === "keep" ? keep : fix;
    if (!list.some((x) => x.title === mapped.title)) list.push({ title: mapped.title, detail: mapped.detail });
  }

  // Verdict
  let verdict = "Solid rep — chase one detail next";
  let summary = play === "run" ? "Run-block snapshot" : "Pass-pro snapshot";
  if (fix.length >= 3) {
    verdict = "Needs work — pick one cue and re-run";
    summary = `Top fix: ${fix[0].title.toLowerCase()}`;
  } else if (fix.length === 0 && keep.length > 0) {
    verdict = "Clean snap — build on what’s working";
    summary = keep[0].title;
  } else if (fix.length === 1) {
    verdict = "One clear coaching point";
    summary = fix[0].title;
  }

  // Cap lists so coaches aren’t buried
  return {
    verdict,
    summary,
    fix: fix.slice(0, 4),
    keep: keep.slice(0, 3),
  };
}

function renderInsightList(el, items, emptyText) {
  if (!items.length) {
    el.innerHTML = `<li class="empty-cue">${emptyText}</li>`;
    return;
  }
  el.innerHTML = items
    .map(
      (it) => `<li>
        <strong>${it.title}</strong>
        <span>${it.detail}</span>
      </li>`
    )
    .join("");
}

function metricRows(r) {
  const knee = r.mean_knee_flexion_deg;
  const minKnee = r.min_knee_flexion_deg;
  const torso = r.mean_torso_angle_deg;
  const hip = r.hip_height_at_lowest;
  const posture = r.posture_classification;
  const pconf = r.posture_confidence;

  const postureCls =
    posture === "waist_bender" ? "warn" : posture === "knee_bender" || posture === "balanced" ? "good" : "";

  return [
    {
      label: "Knee bend (avg)",
      value: knee != null ? `${Math.round(knee)}°` : "—",
      cls: postureCls,
    },
    {
      label: "Knee bend (deepest)",
      value: minKnee != null ? `${Math.round(minKnee)}°` : "—",
    },
    {
      label: "Torso lean",
      value: torso != null ? `${Math.round(torso)}°` : "—",
      cls: torso != null && torso >= 22 ? "warn" : "",
    },
    {
      label: "Pad level (hips)",
      value: hip != null ? fracPct(hip) : "—",
    },
    {
      label: "Posture",
      value:
        posture && posture !== "unknown"
          ? `${pretty(posture)}${pconf != null ? ` (${fracPct(pconf)} conf)` : ""}`
          : "Unclear",
      cls: postureCls,
    },
  ];
}

function renderMetrics(r) {
  const block = $("metrics-block");
  const grid = $("metrics-grid");
  const rows = metricRows(r);
  const any = rows.some((row) => row.value !== "—");
  block.hidden = !any;
  if (!any) return;
  grid.innerHTML = rows
    .map((row) => `<div class="m-label">${row.label}</div><div class="m-value ${row.cls || ""}">${row.value}</div>`)
    .join("");
}

const ATTR_ORDER = [
  ["stance", "Stance"],
  ["first_step", "First step"],
  ["feet", "Feet"],
  ["hands", "Hands"],
  ["hips", "Hips"],
];

const VERDICT_TEXT = { good: "Good", bad: "Bad", uncertain: "Not sure" };

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function fmtValue(v, unit) {
  if (v == null || !Number.isFinite(Number(v))) return "—";
  const n = Number(v);
  const u = String(unit || "").toLowerCase();
  if (u.startsWith("deg")) return `${Math.round(n)}°`;
  if (u === "ms") return `${Math.round(n)} ms`;
  if (u === "hz") return `${n.toFixed(2)} Hz`;
  if (Math.abs(n) >= 100) return String(Math.round(n));
  if (Math.abs(n) >= 10) return n.toFixed(1);
  return n.toFixed(2);
}

function attrSubline(a) {
  if (!a.n_votes) {
    return a.key === "hands"
      ? "Not enough labeled film on hands yet"
      : "Couldn't measure this on this clip";
  }
  return `${a.n_good_votes} of ${a.n_votes} signals look good`;
}

function renderScorecard(r) {
  const grades = r.grades;
  const grid = $("attr-grid");
  const tally = $("scorecard-tally");
  const title = $("scorecard-title");
  const foot = $("scorecard-foot");

  if (!grades || !grades.attributes?.length) {
    title.textContent = "Grading unavailable for this rep";
    tally.hidden = true;
    foot.textContent = grades?.error || "Build benchmarks/attribute_model_v1.json to enable grading.";
    grid.innerHTML = ATTR_ORDER.map(
      ([, label]) => `<div class="attr-card placeholder"><span class="attr-name">${label}</span><span class="attr-badge">—</span></div>`
    ).join("");
    return;
  }

  const byKey = Object.fromEntries(grades.attributes.map((a) => [a.key, a]));
  const attrs = ATTR_ORDER.map(([k, label]) => byKey[k] || { key: k, label, verdict: "uncertain", n_votes: 0, n_good_votes: 0, evidence: [] });
  const good = attrs.filter((a) => a.verdict === "good");
  const bad = attrs.filter((a) => a.verdict === "bad");
  const unsure = attrs.length - good.length - bad.length;

  if (bad.length) title.textContent = `Work on: ${bad.map((a) => a.label.toLowerCase()).join(", ")}`;
  else if (good.length) title.textContent = "Good on every graded trait";
  else title.textContent = "Not enough signal to grade this rep";

  tally.hidden = false;
  tally.innerHTML =
    `<span class="tally good"><b>${good.length}</b> good</span>` +
    `<span class="tally bad"><b>${bad.length}</b> bad</span>` +
    `<span class="tally unsure"><b>${unsure}</b> not sure</span>`;

  grid.innerHTML = attrs
    .map((a) => {
      const ev = (a.evidence || [])
        .map((e) => {
          const cmp = e.higher_is_good ? "≥" : "≤";
          return `<li class="${e.vote}">
            <span class="ev-mark">${e.vote === "good" ? "✓" : "✗"}</span>
            <span class="ev-name">${escapeHtml(e.label)}</span>
            <span class="ev-val">${fmtValue(e.value, e.unit)} <em>good ${cmp} ${fmtValue(e.threshold, e.unit)}</em></span>
          </li>`;
        })
        .join("");
      return `<div class="attr-card ${a.verdict}">
        <div class="attr-top">
          <span class="attr-name">${escapeHtml(a.label)}</span>
          <span class="attr-badge">${VERDICT_TEXT[a.verdict] || "Not sure"}</span>
        </div>
        <p class="attr-sub">${attrSubline(a)}</p>
        ${ev ? `<details><summary>Why</summary><ul class="ev-list">${ev}</ul></details>` : ""}
      </div>`;
    })
    .join("");

  const b = grades.benchmark || {};
  foot.textContent = b.n_reps
    ? `Grades come from expert-labeled high-school reps. Measurements compare to all ${b.n_reps} reference reps. Early model — more labeled film makes it sharper.`
    : "Grades come from expert-labeled high-school reps.";
}

const STATUS_TEXT = {
  within_reference_range: "In range",
  above_reference_range: "Above",
  below_reference_range: "Below",
};

function renderMeasures(r) {
  const section = $("measures");
  const list = $("measure-list");
  const rows = r.grades?.measurements || [];
  section.hidden = !rows.length;
  if (!rows.length) {
    list.innerHTML = "";
    return;
  }
  const sorted = [...rows].sort((x, y) => {
    const ox = x.status === "within_reference_range" ? 1 : 0;
    const oy = y.status === "within_reference_range" ? 1 : 0;
    if (ox !== oy) return ox - oy;
    return Math.abs(y.z ?? 0) - Math.abs(x.z ?? 0);
  });

  list.innerHTML = sorted
    .map((m) => {
      const { p10, p25, p75, p90 } = m.range;
      const vals = [p10, p25, p75, p90, m.value].filter((v) => v != null).map(Number);
      let lo = Math.min(...vals);
      let hi = Math.max(...vals);
      const pad = (hi - lo) * 0.12 || Math.abs(hi) * 0.1 || 1;
      lo -= pad;
      hi += pad;
      const pos = (v) => `${(((Number(v) - lo) / (hi - lo)) * 100).toFixed(1)}%`;
      const inRange = m.status === "within_reference_range";
      return `<div class="measure ${inRange ? "in" : "out"}">
        <div class="measure-label">
          <span>${escapeHtml(m.label)}</span>
          <small>${pretty(m.category)}</small>
        </div>
        <div class="measure-bar">
          <div class="mb-whisker" style="left:${pos(p10 ?? p25)};width:calc(${pos(p90 ?? p75)} - ${pos(p10 ?? p25)})"></div>
          <div class="mb-band" style="left:${pos(p25)};width:calc(${pos(p75)} - ${pos(p25)})"></div>
          <div class="mb-dot" style="left:${pos(m.value)}" title="${fmtValue(m.value, m.unit)}"></div>
        </div>
        <div class="measure-val">
          <b>${fmtValue(m.value, m.unit)}</b>
          <small>Reference avg ${fmtValue(m.mean, m.unit)}</small>
        </div>
        <span class="measure-status">${STATUS_TEXT[m.status] || "—"}</span>
      </div>`;
    })
    .join("");
}

function render(r) {
  currentResult = r;
  renderScorecard(r);
  renderMeasures(r);
  const brief = buildCoachBrief(r);
  const label =
    r.jersey != null && r.jersey !== ""
      ? `#${r.jersey}`
      : r.ol_lock?.method === "manual_pick_xy"
        ? "Locked OL"
        : "OL";

  $("title").textContent = label;
  const play = r.play_type === "run" ? "Run block" : "Pass pro";
  $("subtitle").textContent = `${play} · coach cues for this rep`;

  const trust = r.trust?.overall;
  if (trust) {
    const words = { high: "Clear", medium: "Okay", low: "Fuzzy" };
    $("signal-overall").textContent = words[trust.level] || trust.level;
    $("signal-pill").className = `signal-pill ${trust.level}`;
  } else {
    $("signal-overall").textContent = "—";
    $("signal-pill").className = "signal-pill";
  }

  $("verdict-v").textContent = brief.verdict;
  $("verdict-s").textContent = brief.summary;
  renderMetrics(r);
  renderInsightList($("fix-list"), brief.fix, "Nothing urgent — keep stacking good reps");
  renderInsightList($("keep-list"), brief.keep, "No clear wins tagged yet");

  // Keyframe thumbnails
  const kfs = r.keyframes || [];
  const kfBlock = $("keyframe-block");
  const kfRow = $("keyframe-row");
  if (kfs.length) {
    kfBlock.hidden = false;
    kfRow.innerHTML = kfs
      .map(
        (k) => `<figure>
          <img src="${k.url}" alt="${k.label}" loading="lazy" />
          <figcaption>${k.label}</figcaption>
        </figure>`
      )
      .join("");
  } else {
    kfBlock.hidden = true;
    kfRow.innerHTML = "";
  }

  $("export-pdf").disabled = !(r.report_url || (r.id && r.id !== "demo"));
  $("on-field").disabled = !r.id;

  renderMultiview(r);

  if (r.overlay_url) setPreview(r.overlay_url);

  renderCompare();
}

function setPreview(url) {
  const preview = $("preview");
  preview.onerror = () => {
    $("empty").classList.remove("hide");
    $("empty").textContent =
      "Preview codec not supported — install imageio-ffmpeg and re-analyze (OpenCV mp4v won’t play in browsers).";
    preview.classList.remove("on");
    setStatus("Film encoded but not browser-playable. Re-run Analyze after installing imageio-ffmpeg.");
  };
  preview.onloadeddata = () => {
    $("empty").textContent = "Film preview shows here";
  };
  preview.src = url;
  preview.classList.add("on");
  $("empty").classList.add("hide");
  preview.load();
}

function fracPct(x) {
  return `${Math.round((Number(x) || 0) * 100)}%`;
}

// Two-view fusion summary: alignment warnings, coverage recovery,
// per-view snap timing, and a toggle to switch each camera's overlay.
function renderMultiview(r) {
  const bar = $("mv-bar");
  const mv = r.multiview;
  if (!mv || !mv.coverage) {
    bar.hidden = true;
    return;
  }
  bar.hidden = false;

  let html = "";

  // Alignment warnings (different plays / low snap confidence / fps mismatch)
  const align = mv.alignment || {};
  const warns = align.warnings || [];
  if (warns.length) {
    const level = align.same_play_confidence === "low" ? "warn" : "note";
    html += `<div class="mv-align mv-align-${level}">`;
    for (const w of warns) {
      html += `<span class="mv-warn-line">${w}</span>`;
    }
    html += `</div>`;
  }

  // Per-view snap timing (shows how clips were aligned despite different starts)
  const viewTimings = (mv.views || [])
    .filter((v) => v.snap_at_s != null)
    .map((v) => `${pretty(v.role)} snap at ${v.snap_at_s}s (rep ${v.rep_duration_s}s)`)
    .join(" · ");
  if (viewTimings) {
    html += `<div class="mv-timing">${viewTimings}</div>`;
  }

  // Coverage stats
  const cov = mv.coverage;
  const gain = Math.max(0, (cov.combined_coverage || 0) - (cov.best_single_coverage || 0));
  html +=
    `<div class="mv-cov-row">` +
    `<strong>Occlusion coverage</strong>` +
    `<span class="mv-pill good">Both views: ${fracPct(cov.combined_coverage)}</span>` +
    `<span class="mv-pill">Best single: ${fracPct(cov.best_single_coverage)}</span>` +
    (gain > 0.001
      ? `<span class="mv-pill up">+${fracPct(gain)} recovered</span>`
      : `<span class="mv-pill">no extra gaps to fill</span>`) +
    `</div>`;

  $("mv-cov").innerHTML = html;

  // View toggle buttons
  const views = r.views || [];
  const toggle = $("view-toggle");
  const buttons = [];
  if (r.overlay_url) {
    buttons.push(
      `<button type="button" class="vt-btn active" data-url="${r.overlay_url}">Fused view</button>`
    );
  }
  for (const v of views) {
    if (!v.overlay_url) continue;
    buttons.push(
      `<button type="button" class="vt-btn" data-url="${v.overlay_url}">${pretty(v.role)} · ${fracPct(
        v.coverage
      )}</button>`
    );
  }
  toggle.innerHTML = buttons.join("");
  toggle.querySelectorAll(".vt-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      toggle.querySelectorAll(".vt-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      setPreview(btn.getAttribute("data-url"));
    });
  });
}

function briefLine(r) {
  if (!r) return "—";
  const b = buildCoachBrief(r);
  if (b.fix[0]) return `Fix: ${b.fix[0].title}`;
  if (b.keep[0]) return `Keep: ${b.keep[0].title}`;
  return b.verdict;
}

function renderCompare() {
  const grid = $("compare-grid");
  const a = pinnedResult;
  const b = currentResult;
  if (!a && !b) {
    grid.innerHTML = "";
    $("compare-hint").textContent = "Save a rep, run another, see what changed";
    return;
  }
  if (a && !b) $("compare-hint").textContent = "A saved — analyze another rep for B";
  else if (a && b && a.id === b.id) $("compare-hint").textContent = "Save a different rep to compare";
  else $("compare-hint").textContent = "A = saved · B = this rep";

  const rows = [
    ["Player", (r) => (r.jersey != null ? `#${r.jersey}` : "OL")],
    ["Play", (r) => (r.play_type === "run" ? "Run" : "Pass")],
    ["Verdict", (r) => buildCoachBrief(r).verdict],
    ...ATTR_ORDER.map(([key, label]) => [
      label,
      (r) => {
        const a = (r.grades?.attributes || []).find((x) => x.key === key);
        return a ? VERDICT_TEXT[a.verdict] || "Not sure" : "—";
      },
    ]),
    ["Top fix", (r) => buildCoachBrief(r).fix[0]?.title || "—"],
    ["Top strength", (r) => buildCoachBrief(r).keep[0]?.title || "—"],
    ["Knee bend (avg)", (r) => (r.mean_knee_flexion_deg != null ? `${Math.round(r.mean_knee_flexion_deg)}°` : "—")],
    ["Posture", (r) => pretty(r.posture_classification)],
    ["Get-off", (r) => {
      if (r.late_off_the_ball) return "Late";
      if (r.reaction_time_ms == null) return "—";
      if (r.reaction_time_ms <= 180) return "Quick";
      if (r.reaction_time_ms <= 250) return "Okay";
      return "Slow";
    }],
    ["Anchor", (r) => {
      if (r.anchor_give == null) return "—";
      if (r.anchor_give <= 0.1) return "Firm";
      if (r.anchor_give <= 0.18) return "Some give";
      return "Gave ground";
    }],
    ["Hands", (r) => {
      if (r.punch_ms == null) return "—";
      if (r.punch_ms <= 280) return "On time";
      if (r.punch_ms <= 400) return "Okay";
      return "Late";
    }],
  ];

  grid.innerHTML = `
    <div class="cmp-col head">Cue</div>
    <div class="cmp-col head">A ${a ? "" : "(empty)"}</div>
    <div class="cmp-col head">B ${b ? "" : "(empty)"}</div>
    ${rows
      .map(([name, fn]) => {
        const va = a ? fn(a) : "—";
        const vb = b ? fn(b) : "—";
        return `<div class="cmp-col">${name}</div>
          <div class="cmp-col">${va}</div>
          <div class="cmp-col">${vb}</div>`;
      })
      .join("")}
  `;
}

function pretty(v) {
  return v ? String(v).replaceAll("_", " ") : "—";
}

function setBusy(busy, msg, percent, stage) {
  $("run").disabled = busy;
  $("demo").disabled = busy;
  $("pin-compare").disabled = busy;
  $("export-pdf").disabled = busy || !currentResult;
  $("on-field").disabled = busy || !currentResult;
  const panel = $("progress-panel");
  const veil = $("busy-veil");
  if (busy) {
    panel.hidden = false;
    veil.hidden = false;
    setProgressUI(percent ?? 1, msg || "Working…", stage || "ingest", []);
  } else {
    panel.hidden = true;
    veil.hidden = true;
    setStatus(msg || "");
    if (currentResult) {
      $("export-pdf").disabled = !(currentResult.report_url || (currentResult.id && currentResult.id !== "demo"));
      $("on-field").disabled = !currentResult.id;
    }
  }
}

function setProgressUI(percent, message, stage, stagesDone) {
  const pct = Math.max(0, Math.min(100, Math.round(Number(percent) || 0)));
  const msg = message || "Working…";
  setStatus(msg);
  $("progress-label").textContent = msg;
  $("progress-pct").textContent = `${pct}%`;
  $("progress-fill").style.width = `${pct}%`;
  $("busy-msg").textContent = msg;
  $("busy-pct").textContent = `${pct}%`;
  $("busy-fill").style.width = `${pct}%`;

  const done = new Set(stagesDone || []);
  document.querySelectorAll("#progress-stages li").forEach((li) => {
    const s = li.getAttribute("data-stage");
    li.classList.toggle("done", done.has(s) || stage === "done");
    li.classList.toggle("active", stage === s && stage !== "done");
  });
}

function setStatus(msg) {
  $("status").textContent = msg || "";
}

renderCompare();
