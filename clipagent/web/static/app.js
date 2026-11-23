// clipagent web UI - plain JS, no build step.
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtTime = (t) => {
  t = Math.max(0, t || 0);
  const m = Math.floor(t / 60), s = (t % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
};

const state = { jobs: [], job: null, defaults: null, status: null, transcript: null,
                editing: null, pollTimer: null, seen: new Set() };

async function api(path, opts = {}) {
  const res = await fetch(path, { credentials: "same-origin", ...opts,
    headers: opts.body && !(opts.body instanceof FormData)
      ? { "Content-Type": "application/json", ...(opts.headers || {}) } : opts.headers });
  if (res.status === 401) { showLogin(); throw new Error("login required"); }
  const data = res.headers.get("content-type")?.includes("json") ? await res.json() : null;
  if (!res.ok) throw new Error(data?.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : res.statusText);
  return data;
}

// ------------------------------------------------------------------ login
function showLogin() {
  const dlg = $("#login");
  if (!dlg.open) dlg.showModal();
}
$("#login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    await api("/api/login", { method: "POST", body: JSON.stringify({ password: e.target.password.value }) });
    $("#login").close();
    boot();
  } catch (err) { $("#login-error").textContent = err.message; }
});

// ------------------------------------------------------------------ status chips
async function loadStatus() {
  try {
    state.status = await api("/api/status");
  } catch { return; }
  const pick = (name) => state.status.checks.find((c) => c.name.startsWith(name));
  const chips = [
    ["AI", pick("AI provider")], ["GPU", pick("GPU")], ["B-roll", pick("B-roll")],
  ];
  const fails = state.status.checks.filter((c) => c.status === "fail");
  $("#status-chips").innerHTML =
    fails.map((c) => `<span class="chip fail" title="${esc(c.fix)}">${esc(c.name)}</span>`).join("") +
    chips.filter(([, c]) => c).map(([n, c]) =>
      `<span class="chip ${c.status}" title="${esc(c.detail)}">${esc(n)}</span>`).join("");
}

// ------------------------------------------------------------------ jobs list
async function loadJobs() {
  state.jobs = await api("/api/jobs");
  const current = location.hash.split("/")[2];
  $("#job-list").innerHTML = state.jobs.map((j) => `
    <li><a href="#/job/${j.id}" class="${j.id === current ? "active" : ""}">
      <span class="name">${esc(j.source_name)}</span>
      <span class="meta">${j.status === "running" ? `${esc(j.stage)} · ${Math.round(j.progress * 100)}%`
        : j.status === "done" ? `${j.clips} clips` : esc(j.status)}</span>
    </a></li>`).join("") || `<li class="muted small">No projects yet</li>`;
}

// ------------------------------------------------------------------ new project
function renderNew() {
  const main = $("#main");
  main.innerHTML = "";
  main.append($("#tpl-new").content.cloneNode(true));
  const form = $("#new-form"), d = state.defaults;
  for (const [k, v] of Object.entries(d)) {
    const el = form.elements[k];
    if (!el) continue;
    if (el.type === "checkbox") el.checked = !!v;
    else if (el.type === "color") el.value = "#" + v;
    else el.value = v ?? "";
  }
  if (!state.status?.checks.find((c) => c.name === "AI provider" && c.status === "ok")) {
    form.elements.use_llm.parentElement.title = "Add ANTHROPIC_API_KEY or GEMINI_API_KEY to enable";
  }
  const drop = $("#drop"), fileInput = form.elements.file;
  const showName = () => { $(".filename", drop).textContent = fileInput.files[0]?.name || ""; };
  fileInput.addEventListener("change", showName);
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault(); drop.classList.remove("over");
    fileInput.files = e.dataTransfer.files; showName();
  });
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = form.querySelector("button[type=submit]");
    const settings = {};
    for (const k of Object.keys(d)) {
      const el = form.elements[k];
      if (!el) continue;
      settings[k] = el.type === "checkbox" ? el.checked : el.type === "color" ? el.value.slice(1)
        : el.type === "number" ? Number(el.value) : el.value;
    }
    const fd = new FormData();
    if (fileInput.files[0]) fd.append("file", fileInput.files[0]);
    fd.append("url", form.elements.url.value);
    fd.append("settings", JSON.stringify(settings));
    btn.disabled = true; btn.textContent = fileInput.files[0] ? "Uploading…" : "Starting…";
    try {
      const job = await api("/api/jobs", { method: "POST", body: fd });
      location.hash = `#/job/${job.id}`;
    } catch (err) {
      $("#new-error").textContent = err.message;
      btn.disabled = false; btn.textContent = "Find clips";
    }
  });
}

// ------------------------------------------------------------------ job view
async function renderJob(id) {
  clearTimeout(state.pollTimer);
  let job;
  try { job = await api(`/api/jobs/${id}`); } catch (err) {
    $("#main").innerHTML = `<p class="empty">${esc(err.message)}</p>`; return;
  }
  state.job = job;
  const busy = job.status === "queued" || job.status === "running";
  const rendering = job.clips.some((c) => c.status === "queued" || c.status === "rendering");
  $("#main").innerHTML = `
    <div class="job-head">
      <div>
        <h1 class="font-serif job-title">${esc(job.source_name)}</h1>
        <div class="muted">${job.source_info?.duration ? `${fmtTime(job.source_info.duration)} source · ` : ""}
          ${job.clips.length} clips · ${esc(job.status)}</div>
      </div>
      <div class="row">
        ${job.status === "done" ? `<button class="pill pill-ghost" id="add-clip">+ Custom clip</button>` : ""}
        <button class="pill pill-danger" id="delete-job" ${busy ? "disabled" : ""}>Delete</button>
      </div>
    </div>
    ${busy ? `<div class="panel"><span class="stage-label">${esc(job.stage)}</span>
        <div class="progress"><div style="width:${Math.round(job.progress * 100)}%"></div></div>
        <div class="log">${esc(job.log.slice(-40).join("\n"))}</div></div>` : ""}
    ${job.status === "error" ? `<div class="panel"><p class="error">${esc(job.error)}</p>
        <div class="log">${esc(job.log.slice(-40).join("\n"))}</div></div>` : ""}
    <div class="clips">${job.clips.map(clipCard).join("")}</div>`;
  job.clips.forEach((c) => state.seen.add(c.id));

  $("#delete-job")?.addEventListener("click", async () => {
    if (!confirm("Delete this project and all its clips?")) return;
    await api(`/api/jobs/${id}`, { method: "DELETE" });
    location.hash = "#/"; loadJobs();
  });
  $("#add-clip")?.addEventListener("click", () => openEditor(null));
  for (const card of document.querySelectorAll(".card")) {
    const clip = job.clips.find((c) => c.id === card.dataset.id);
    card.querySelector("[data-act=edit]")?.addEventListener("click", () => openEditor(clip));
    card.querySelector("[data-act=delete]")?.addEventListener("click", async () => {
      if (!confirm("Delete this clip?")) return;
      await api(`/api/jobs/${id}/clips/${clip.id}`, { method: "DELETE" });
      renderJob(id);
    });
  }
  if (busy || rendering) state.pollTimer = setTimeout(() => { if (currentJobId() === id) { renderJob(id); loadJobs(); } }, 2000);
}

function clipCard(c) {
  const s = c.scores || {};
  const bars = ["hook", "flow", "value", "emotion", "pacing", "trend"].map((k) => `
    <span>${k}</span><div class="bar"><div style="width:${Math.round(s[k] || 0)}%"></div></div><span>${Math.round(s[k] || 0)}</span>`).join("");
  const working = c.status === "queued" || c.status === "rendering";
  const tags = [c.layout, c.broll?.length ? `${c.broll.length} b-roll` : "", c.emojis?.length ? "emoji" : "",
                c.speaker_switches ? `${c.speaker_switches} speaker cuts` : ""].filter(Boolean).join(" · ");
  return `
    <article class="card${state.seen.has(c.id) ? "" : " reveal"}" data-id="${esc(c.id)}">
      <div class="video">
        ${c.url ? `<video src="${esc(c.url)}" controls preload="metadata" playsinline></video>` : ""}
        <span class="badge ${c.virality >= 70 ? "hot" : ""}">${Math.round(c.virality)}</span>
        ${working ? `<div class="overlay-status">${c.status === "queued" ? "Queued…" : "Rendering…"}</div>` : ""}
        ${c.status === "error" ? `<div class="overlay-status error">Render failed</div>` : ""}
      </div>
      <div class="body">
        <div class="title">${esc(c.title)}</div>
        <div class="meta">${fmtTime(c.start)}–${fmtTime(c.end)} · ${Math.round(c.duration)}s${tags ? " · " + esc(tags) : ""}</div>
        ${c.reason ? `<div class="reason">${esc(c.reason)}</div>` : ""}
        <div class="bars">${bars}</div>
        <div class="actions">
          <button class="pill pill-white pill-sm" data-act="edit">Edit</button>
          ${c.url ? `<a class="pill pill-ghost pill-sm" href="${esc(c.url)}" download>Download</a>` : ""}
          <button class="icon-btn danger" data-act="delete" title="Delete clip" aria-label="Delete clip"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg></button>
        </div>
      </div>
    </article>`;
}

// ------------------------------------------------------------------ editor
async function openEditor(clip) {
  const job = state.job, dlg = $("#editor");
  state.editing = clip;
  try { state.transcript = await api(`/api/jobs/${job.id}/transcript`); } catch { state.transcript = { words: [] }; }
  const opts = clip?.options || {}, d = job.settings || {}, def = state.defaults;
  const pick = (k) => (k in opts ? opts[k] : k in d ? d[k] : def[k]);
  const video = $("#ed-source");
  video.src = job.source_url;
  $("#ed-start").value = (clip?.start ?? 0).toFixed(2);
  $("#ed-end").value = (clip?.end ?? Math.min(30, job.source_info?.duration || 30)).toFixed(2);
  $("#ed-title").value = clip?.title || "";
  $("#ed-description").value = clip?.description || "";
  $("#ed-hashtags").value = (clip?.hashtags || []).join(", ");
  $("#ed-layout").value = opts.layout || "";
  $("#ed-highlight").value = "#" + pick("caption_highlight");
  for (const k of ["captions", "show_title", "emoji", "broll"]) $(`#ed-${k}`).checked = !!pick(k);
  $("#ed-save").hidden = !clip;
  $("#ed-save-new").textContent = clip ? "Save as new clip" : "Create clip";
  $("#ed-error").textContent = "";

  const words = state.transcript.words;
  $("#ed-transcript").innerHTML = words.map((w, i) => `<span data-i="${i}">${esc(w.text)}</span>`).join(" ");
  paintRange();
  video.addEventListener("loadedmetadata", () => { video.currentTime = Number($("#ed-start").value); }, { once: true });
  dlg.showModal();
}

function paintRange() {
  const a = Number($("#ed-start").value), b = Number($("#ed-end").value);
  const words = state.transcript?.words || [];
  for (const el of $("#ed-transcript").children) {
    const w = words[el.dataset.i];
    el.classList.toggle("in", w.start >= a - 1e-3 && w.end <= b + 1e-3);
  }
  $("#ed-length").textContent = b > a ? `${(b - a).toFixed(1)}s` : "end must be after start";
}

$("#ed-transcript").addEventListener("click", (e) => {
  const i = e.target.dataset?.i;
  if (i === undefined) return;
  const w = state.transcript.words[i], video = $("#ed-source");
  if (e.shiftKey) {
    // Shift-click moves whichever edge of the range is closer to the word.
    const a = Number($("#ed-start").value), b = Number($("#ed-end").value);
    if (w.start < (a + b) / 2) $("#ed-start").value = w.start.toFixed(2);
    else $("#ed-end").value = w.end.toFixed(2);
    paintRange();
  } else {
    video.currentTime = w.start;
  }
});
for (const id of ["#ed-start", "#ed-end"]) $(id).addEventListener("input", paintRange);
$("#ed-set-start").addEventListener("click", () => { $("#ed-start").value = $("#ed-source").currentTime.toFixed(2); paintRange(); });
$("#ed-set-end").addEventListener("click", () => { $("#ed-end").value = $("#ed-source").currentTime.toFixed(2); paintRange(); });
$("#ed-preview").addEventListener("click", () => {
  const v = $("#ed-source"), end = Number($("#ed-end").value);
  v.currentTime = Number($("#ed-start").value); v.play();
  const stop = () => { if (v.currentTime >= end) { v.pause(); v.removeEventListener("timeupdate", stop); } };
  v.addEventListener("timeupdate", stop);
});
$("#ed-source").addEventListener("timeupdate", () => {
  const t = $("#ed-source").currentTime, words = state.transcript?.words || [];
  for (const el of $("#ed-transcript").children) {
    const w = words[el.dataset.i];
    el.classList.toggle("now", t >= w.start && t < w.end);
  }
});

function editorPayload() {
  const job = state.job, d = { ...state.defaults, ...(job.settings || {}) };
  const options = {};
  const layout = $("#ed-layout").value;
  if (layout) options.layout = layout;
  const hl = $("#ed-highlight").value.slice(1).toUpperCase();
  if (hl !== String(d.caption_highlight).toUpperCase()) options.caption_highlight = hl;
  for (const k of ["captions", "show_title", "emoji", "broll"]) {
    const v = $(`#ed-${k}`).checked;
    if (v !== !!d[k]) options[k] = v;
  }
  return {
    start: Number($("#ed-start").value), end: Number($("#ed-end").value),
    title: $("#ed-title").value.trim(), description: $("#ed-description").value.trim(),
    hashtags: $("#ed-hashtags").value.split(",").map((s) => s.trim()).filter(Boolean), options,
  };
}

async function saveEditor(asNew) {
  const job = state.job, body = editorPayload();
  try {
    if (asNew || !state.editing) {
      const clip = await api(`/api/jobs/${job.id}/clips`, { method: "POST",
        body: JSON.stringify({ start: body.start, end: body.end, title: body.title || "Custom clip" }) });
      await api(`/api/jobs/${job.id}/clips/${clip.id}?render=false`, { method: "PATCH", body: JSON.stringify(body) });
    } else {
      await api(`/api/jobs/${job.id}/clips/${state.editing.id}`, { method: "PATCH", body: JSON.stringify(body) });
    }
    $("#ed-source").pause();
    $("#editor").close();
    renderJob(job.id);
  } catch (err) { $("#ed-error").textContent = err.message; }
}
$("#ed-save").addEventListener("click", () => saveEditor(false));
$("#ed-save-new").addEventListener("click", () => saveEditor(true));
$("#editor").addEventListener("close", () => $("#ed-source").pause());

// ------------------------------------------------------------------ motion
const revealObserver = new IntersectionObserver((entries) => {
  for (const e of entries) if (e.isIntersecting) { e.target.classList.add("active"); revealObserver.unobserve(e.target); }
}, { threshold: 0.1, rootMargin: "0px 0px -40px 0px" });
function observeReveals() {
  document.querySelectorAll(".reveal:not(.active)").forEach((el) => revealObserver.observe(el));
}
new MutationObserver(observeReveals).observe($("#main"), { childList: true, subtree: true });
window.addEventListener("scroll", () => $("#main-nav").classList.toggle("scrolled", window.scrollY > 20), { passive: true });

// ------------------------------------------------------------------ routing
function currentJobId() { const p = location.hash.split("/"); return p[1] === "job" ? p[2] : null; }
async function route() {
  clearTimeout(state.pollTimer);
  const id = currentJobId();
  await loadJobs();
  if (id) renderJob(id);
  else renderNew();
}
$("#new-project-btn").addEventListener("click", () => { location.hash = "#/"; });
window.addEventListener("hashchange", route);

async function boot() {
  try {
    state.defaults = await api("/api/defaults");
  } catch { return; }
  loadStatus().then(() => { if (!currentJobId()) renderNew(); });
  route();
  setInterval(loadJobs, 5000);
}
boot();
