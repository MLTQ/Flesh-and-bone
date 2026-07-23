const state = {
  activeJob: null,
  pollTimer: null,
  clockTimer: null,
  jobs: [],
};

const $ = (selector) => document.querySelector(selector);
const stageOrder = ["queued", "generating", "retargeting", "diagnostics", "rendering"];
const stageAliases = {
  checking_server: "queued",
  contact_ik: "retargeting",
  skinning: "rendering",
  complete: "complete",
  failed: "failed",
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({ error: response.statusText }));
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatElapsed(iso) {
  if (!iso) return "Idle";
  const seconds = Math.max(0, Math.floor((Date.now() - new Date(iso).getTime()) / 1000));
  const minutes = Math.floor(seconds / 60);
  return minutes ? `${minutes}m ${seconds % 60}s elapsed` : `${seconds}s elapsed`;
}

function formatMetric(metric) {
  if (metric.value === null || metric.value === undefined) return "n/a";
  if (metric.unit === "bool") return metric.value ? "yes" : "no";
  if (metric.unit === "deg") return `${metric.value.toFixed(1)}°`;
  if (metric.unit === "m") {
    const millimeters = metric.value * 1000;
    return `${millimeters < 1 ? millimeters.toFixed(2) : millimeters.toFixed(1)} mm`;
  }
  return `${metric.value.toFixed(2)} ${metric.unit}`;
}

async function checkHealth() {
  const dot = $("#service-dot");
  try {
    const health = await api("/api/health");
    dot.className = "service-dot online";
    $("#service-state").textContent = "Kimodo online";
    $("#service-model").textContent = `${health.kimodo.model || "model ready"} · ${health.kimodo.version || "unknown version"}`;
  } catch (error) {
    dot.className = "service-dot offline";
    $("#service-state").textContent = "Kimodo unavailable";
    $("#service-model").textContent = error.message;
  }
}

function estimatedProgress(job) {
  let progress = Number(job.progress || 0);
  if (job.stage === "generating") {
    const seconds = Math.max(0, (Date.now() - new Date(job.updated_at).getTime()) / 1000);
    const expected = Math.max(3, Number(job.request.diffusion_steps || 50) * Number(job.request.duration_s || 6) / 100);
    progress = Math.max(progress, Math.min(0.46, 0.10 + 0.34 * (1 - Math.exp(-seconds / expected))));
  }
  return progress;
}

function renderProgress(job) {
  const isActive = job && ["queued", "running"].includes(job.status);
  $("#generate-button").disabled = Boolean(isActive);
  if (!job) return;
  const progress = job.status === "failed" ? Number(job.progress || 0) : estimatedProgress(job);
  $("#progress-bar").style.width = `${Math.round(progress * 100)}%`;
  $("#progress-value").textContent = `${Math.round(progress * 100)}%`;
  $("#stage-title").textContent = job.status === "failed" ? "Pipeline failed" : job.stage.replaceAll("_", " ");
  $("#stage-message").textContent = job.error || job.message;
  $("#elapsed").textContent = formatElapsed(job.created_at);
  $("#stage-pulse").className = `stage-pulse ${job.status === "failed" ? "failed" : isActive ? "active" : ""}`;

  const normalized = stageAliases[job.stage] || job.stage;
  const currentIndex = normalized === "complete" ? stageOrder.length : stageOrder.indexOf(normalized);
  document.querySelectorAll("#stage-list li").forEach((item, index) => {
    item.classList.toggle("done", currentIndex > index);
    item.classList.toggle("current", currentIndex === index && job.status !== "failed");
  });
}

function artifactUrl(jobId, filename) {
  return `/artifacts/${encodeURIComponent(jobId)}/${encodeURIComponent(filename)}`;
}

function renderResult(job) {
  if (!job || job.status !== "complete" || !job.result) return;
  const result = job.result;
  const diagnostics = result.diagnostics;
  const decision = result.decision || { status: "unreviewed", note: "" };
  $("#result-panel").classList.remove("hidden");
  $("#animation-preview").src = `${artifactUrl(job.id, result.artifacts.animation)}?v=${encodeURIComponent(job.updated_at)}`;
  $("#sheet-link").href = artifactUrl(job.id, result.artifacts.contact_sheet);
  $("#anatomy-link").href = artifactUrl(job.id, result.artifacts.anatomy_frame);
  $("#raw-link").href = artifactUrl(job.id, result.artifacts.raw_motion);
  $("#raw-link").download = `${job.id}-raw-kimodo.npz`;
  $("#motion-link").href = artifactUrl(job.id, result.artifacts.retargeted_motion);
  $("#motion-link").download = `${job.id}-retargeted.npz`;
  $("#result-prompt").textContent = `“${result.request.prompt}”`;
  $("#clip-facts").innerHTML = [
    `seed ${result.request.seed}`,
    `${result.motion.frame_count} frames`,
    `${result.motion.fps} fps`,
    `${result.request.diffusion_steps} steps`,
    `${result.motion.mapped_role_count} mapped roles`,
    result.motion.contact_ik_applied ? "contact IK on" : "contact IK off",
  ].map((fact) => `<span>${escapeHtml(fact)}</span>`).join("");

  const badge = $("#verdict-badge");
  badge.className = `verdict ${diagnostics.verdict}`;
  badge.textContent = `${diagnostics.verdict} anatomy screen`;
  $("#metrics").innerHTML = diagnostics.metrics.map((metric) => `
    <article class="metric ${escapeHtml(metric.status)}" title="${escapeHtml(metric.detail)}">
      <strong>${escapeHtml(metric.label)}</strong>
      <span>${escapeHtml(formatMetric(metric))}</span>
      <small>${escapeHtml(metric.detail)}</small>
    </article>
  `).join("");

  const angles = Object.entries(diagnostics.angles).map(([name, values]) => `
    <div><span>${escapeHtml(name.replaceAll("_", " "))}</span><code>${values.minimum_deg.toFixed(0)}–${values.maximum_deg.toFixed(0)}°</code></div>
  `).join("");
  $("#detail-content").innerHTML = `
    <p>Worst pelvis frame: ${diagnostics.worst_frame + 1}. Root scale: ${result.motion.root_scale.toFixed(3)}. The diagnostic image uses stable left/right limb colors.</p>
    <div class="detail-grid">${angles}</div>
    <p>${Object.keys(result.retarget.mapped_pairs).length} destination joints are semantically driven. Full names and source pairs remain in manifest.json.</p>
  `;
  $("#decision-note").value = decision.note || "";
  $("#decision-state").textContent = decision.status === "unreviewed" ? "No manual decision recorded." : `${decision.status} · ${decision.updated_at ? new Date(decision.updated_at).toLocaleString() : "saved"}`;
}

function renderHistory() {
  const container = $("#history");
  if (!state.jobs.length) {
    container.innerHTML = '<p class="empty">No clips generated yet.</p>';
    return;
  }
  container.innerHTML = state.jobs.map((job) => {
    const decision = job.result?.decision?.status;
    const suffix = decision && decision !== "unreviewed" ? ` · ${decision}` : "";
    return `
      <button class="history-item" type="button" data-job-id="${escapeHtml(job.id)}">
        <span class="history-status ${escapeHtml(job.status)}"></span>
        <span class="history-copy"><strong>${escapeHtml(job.request.prompt)}</strong><small>${escapeHtml(job.status)}${escapeHtml(suffix)} · seed ${escapeHtml(job.request.seed)}</small></span>
        <code>${escapeHtml(job.id.slice(-8))}</code>
      </button>
    `;
  }).join("");
  container.querySelectorAll("[data-job-id]").forEach((button) => {
    button.addEventListener("click", () => selectJob(button.dataset.jobId));
  });
}

async function refreshJobs(selectNewest = false) {
  const payload = await api("/api/jobs");
  state.jobs = payload.jobs;
  renderHistory();
  if (selectNewest && state.jobs.length) await selectJob(state.jobs[0].id);
}

async function selectJob(jobId) {
  const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
  state.activeJob = job;
  renderProgress(job);
  renderResult(job);
  if (["queued", "running"].includes(job.status)) startPolling();
  else stopPolling();
  if (job.status === "complete") $("#result-panel").scrollIntoView({ behavior: "smooth", block: "start" });
}

function startPolling() {
  if (state.pollTimer) return;
  state.pollTimer = window.setInterval(async () => {
    if (!state.activeJob) return;
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(state.activeJob.id)}`);
      state.activeJob = job;
      renderProgress(job);
      if (!["queued", "running"].includes(job.status)) {
        renderResult(job);
        await refreshJobs();
        stopPolling();
      }
    } catch (error) {
      $("#stage-message").textContent = error.message;
    }
  }, 850);
  state.clockTimer = window.setInterval(() => {
    if (state.activeJob) renderProgress(state.activeJob);
  }, 250);
}

function stopPolling() {
  window.clearInterval(state.pollTimer);
  window.clearInterval(state.clockTimer);
  state.pollTimer = null;
  state.clockTimer = null;
}

async function submitGeneration(event) {
  event.preventDefault();
  $("#form-error").textContent = "";
  const seed = $("#seed").value.trim();
  const payload = {
    prompt: $("#prompt").value,
    duration_s: Number($("#duration").value),
    seed: seed === "" ? null : Number(seed),
    diffusion_steps: Number($("#steps").value),
    postprocess: $("#postprocess").checked,
    apply_contact_ik: $("#contact-ik").checked,
  };
  try {
    const job = await api("/api/jobs", { method: "POST", body: JSON.stringify(payload) });
    state.activeJob = job;
    renderProgress(job);
    await refreshJobs();
    startPolling();
  } catch (error) {
    $("#form-error").textContent = error.message;
  }
}

async function saveDecision(status) {
  if (!state.activeJob) return;
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(state.activeJob.id)}/decision`, {
      method: "POST",
      body: JSON.stringify({ status, note: $("#decision-note").value }),
    });
    state.activeJob = job;
    renderResult(job);
    await refreshJobs();
  } catch (error) {
    $("#decision-state").textContent = error.message;
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  $("#generation-form").addEventListener("submit", submitGeneration);
  $("#accept-button").addEventListener("click", () => saveDecision("accepted"));
  $("#reject-button").addEventListener("click", () => saveDecision("rejected"));
  $("#refresh-history").addEventListener("click", () => refreshJobs());
  document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => { $("#prompt").value = button.dataset.prompt; });
  });
  await Promise.allSettled([checkHealth(), refreshJobs(true)]);
});
