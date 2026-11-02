const questionEl = document.querySelector("#question");
const runButton = document.querySelector("#run");
const hintEl = document.querySelector("#hint");
const chipsEl = document.querySelector("#chips");
const bannerEl = document.querySelector("#banner");
const providerEl = document.querySelector("#provider");
const workspaceEl = document.querySelector("#workspace");
const timelineEl = document.querySelector("#timeline");
const sheetEl = document.querySelector("#sheet");
const recentWrap = document.querySelector("#recent-wrap");
const recentEl = document.querySelector("#recent");

const eventsByJob = new Map();
let currentId = null;
let source = null;
let pollTimer = null;

questionEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    submit();
  }
});
runButton.addEventListener("click", submit);

boot();

async function boot() {
  const [meta, examples] = await Promise.all([
    fetchJson("/api/meta"),
    fetchJson("/api/examples"),
  ]);
  if (meta) {
    providerEl.textContent = `${meta.llm_provider} · ${meta.llm_model} · ${meta.embedding_provider}`;
    if (meta.llm_provider === "mock") {
      bannerEl.hidden = false;
      bannerEl.textContent =
        "Extractive mock writer is on, so claims quote the sources. Set LLM_PROVIDER to groq, gemini, or ollama for a written synthesis. Search still uses the live free APIs.";
    }
  }
  for (const example of examples?.examples || []) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = example;
    button.addEventListener("click", () => {
      questionEl.value = example;
      questionEl.focus();
    });
    chipsEl.appendChild(button);
  }
  await loadRecent();
}

async function submit() {
  const question = questionEl.value.trim();
  hintEl.classList.remove("error");
  if (question.length < 12) {
    hintEl.textContent = "Use at least 12 characters so the planner has something to split.";
    hintEl.classList.add("error");
    return;
  }
  runButton.disabled = true;
  runButton.textContent = "Queuing";
  hintEl.textContent = "Sending the question to the worker.";
  try {
    const job = await fetchJson("/api/jobs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!job?.id) {
      throw new Error("The API did not return a job id.");
    }
    openJob(job);
  } catch (error) {
    hintEl.textContent = error.message || "Could not start the job.";
    hintEl.classList.add("error");
    runButton.disabled = false;
    runButton.textContent = "Run research";
  }
}

function openJob(job) {
  currentId = job.id;
  workspaceEl.hidden = false;
  eventsByJob.set(job.id, new Map());
  renderTimeline(job.id);
  renderSheetLoading(job);
  follow(job.id);
  loadRecent();
}

function follow(jobId) {
  stopFollow();
  source = new EventSource(`/api/jobs/${jobId}/events`);
  source.onmessage = (message) => {
    const event = JSON.parse(message.data);
    addEvent(jobId, event);
    if (event.kind === "job.completed" || event.kind === "job.failed") {
      stopFollow();
      refreshJob(jobId);
    }
  };
  source.onerror = () => {
    if (source) source.close();
    source = null;
    poll(jobId);
  };
}

function poll(jobId) {
  if (pollTimer) return;
  const tick = () => {
    sync(jobId).catch(() => {});
  };
  pollTimer = setInterval(tick, 2000);
  tick();
}

async function sync(jobId) {
  const [job, log] = await Promise.all([
    fetchJson(`/api/jobs/${jobId}`),
    fetchJson(`/api/jobs/${jobId}/log`),
  ]);
  if (Array.isArray(log)) {
    for (const event of log) addEvent(jobId, event);
  }
  if (job && (job.status === "completed" || job.status === "failed")) {
    stopFollow();
    await refreshJob(jobId);
  }
}

function stopFollow() {
  if (source) {
    source.close();
    source = null;
  }
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function addEvent(jobId, event) {
  if (!event || event.kind === "keepalive" || !event.seq) return;
  const bucket = eventsByJob.get(jobId) || new Map();
  bucket.set(event.seq, event);
  eventsByJob.set(jobId, bucket);
  if (jobId === currentId) renderTimeline(jobId);
}

function renderTimeline(jobId) {
  const bucket = eventsByJob.get(jobId) || new Map();
  const events = [...bucket.values()].sort((a, b) => a.seq - b.seq);
  timelineEl.replaceChildren();
  if (!events.length) {
    const item = document.createElement("li");
    item.innerHTML = `<span class="agent">queue</span><p>Waiting for the worker.</p>`;
    timelineEl.appendChild(item);
    return;
  }
  for (const event of events) {
    const item = document.createElement("li");
    if (String(event.kind).includes("failed")) item.classList.add("failed");
    const agent = document.createElement("span");
    agent.className = "agent";
    agent.textContent = event.agent || "job";
    const text = document.createElement("p");
    text.textContent = event.message || event.kind;
    item.append(agent, text);
    const extra = extraLines(event.payload);
    if (extra) item.append(extra);
    timelineEl.appendChild(item);
  }
}

function extraLines(payload) {
  if (!payload) return null;
  const list = document.createElement("ul");
  list.className = "sub";
  const questions = payload.sub_questions || [];
  const notes = payload.notes || [];
  for (const line of [...questions, ...notes]) {
    const item = document.createElement("li");
    item.textContent = line;
    list.appendChild(item);
  }
  return list.childElementCount ? list : null;
}

function renderSheetLoading(job) {
  sheetEl.replaceChildren();
  const p = document.createElement("p");
  p.className = "waiting";
  p.textContent = `Working on “${job.question}”. Agent notes will collect on the left.`;
  sheetEl.appendChild(p);
}

async function refreshJob(jobId) {
  const job = await fetchJson(`/api/jobs/${jobId}`);
  if (!job || jobId !== currentId) return;
  runButton.disabled = false;
  runButton.textContent = "Run research";
  if (job.status === "failed") {
    hintEl.textContent = job.error || "The job failed.";
    hintEl.classList.add("error");
    sheetEl.replaceChildren();
    const p = document.createElement("p");
    p.className = "waiting";
    p.textContent = job.error || "The job failed before a report was written.";
    sheetEl.appendChild(p);
    loadRecent();
    return;
  }
  hintEl.classList.remove("error");
  hintEl.textContent = "Report ready. Downloads are markdown, HTML, and PDF.";
  renderReport(job);
  loadRecent();
}

function renderReport(job) {
  sheetEl.replaceChildren();
  const downloads = document.createElement("div");
  downloads.className = "downloads";
  for (const [label, format] of [["Markdown", "md"], ["HTML", "html"], ["PDF", "pdf"]]) {
    const link = document.createElement("a");
    link.href = `/api/jobs/${job.id}/report?format=${format}`;
    link.textContent = label;
    downloads.appendChild(link);
  }
  sheetEl.appendChild(downloads);
  const score = document.createElement("p");
  score.className = "score";
  const stats = job.stats || {};
  if (stats.claim_count) {
    score.textContent = `${stats.supported_count}/${stats.claim_count} claims supported · ${stats.source_count || 0} sources · ${stats.revisions || 0} revisions · ${stats.elapsed_ms || 0} ms`;
  } else if (job.support_score === null || job.support_score === undefined) {
    score.textContent = "No factual claims were drafted.";
  } else {
    score.textContent = `Support score ${Number(job.support_score).toFixed(2)}`;
  }
  sheetEl.appendChild(score);
  const body = document.createElement("div");
  body.innerHTML = renderMarkdown(job.report_md || "");
  sheetEl.appendChild(body);
}

async function loadRecent() {
  const jobs = await fetchJson("/api/jobs?limit=8");
  if (!Array.isArray(jobs) || !jobs.length) {
    recentWrap.hidden = true;
    return;
  }
  recentWrap.hidden = false;
  recentEl.replaceChildren();
  for (const job of jobs) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    const status = document.createElement("span");
    status.className = "status";
    status.textContent = job.status;
    button.append(status, document.createTextNode(job.question));
    button.addEventListener("click", () => restore(job.id));
    item.appendChild(button);
    recentEl.appendChild(item);
  }
}

async function restore(jobId) {
  const job = await fetchJson(`/api/jobs/${jobId}`);
  if (!job) return;
  questionEl.value = job.question;
  currentId = job.id;
  workspaceEl.hidden = false;
  if (!eventsByJob.has(job.id)) eventsByJob.set(job.id, new Map());
  const log = await fetchJson(`/api/jobs/${jobId}/log`);
  if (Array.isArray(log)) {
    for (const event of log) addEvent(job.id, event);
  }
  if (job.status === "completed" || job.status === "failed") {
    stopFollow();
    renderTimeline(job.id);
    if (job.status === "completed") renderReport(job);
    else {
      sheetEl.replaceChildren();
      const note = document.createElement("p");
      note.className = "waiting";
      note.textContent = job.error || "The job failed before a report was written.";
      sheetEl.appendChild(note);
    }
    return;
  }
  openJob(job);
}

function renderMarkdown(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  let html = "";
  let list = null;
  let inRefs = false;
  const closeList = () => {
    if (list) {
      html += list === "ol" ? "</ol>" : "</ul>";
      list = null;
    }
  };
  for (const line of lines) {
    if (!line.trim()) {
      closeList();
      continue;
    }
    if (line.startsWith("### ")) {
      closeList();
      html += `<h3>${inline(line.slice(4))}</h3>`;
      continue;
    }
    if (line.startsWith("## ")) {
      closeList();
      inRefs = line.slice(3).trim().toLowerCase() === "references";
      html += `<h2>${inline(line.slice(3))}</h2>`;
      continue;
    }
    if (line.startsWith("# ")) {
      closeList();
      html += `<h1>${inline(line.slice(2))}</h1>`;
      continue;
    }
    if (line.startsWith("> ")) {
      closeList();
      html += `<blockquote><p>${inline(line.slice(2))}</p></blockquote>`;
      continue;
    }
    const ordered = line.match(/^(\d+)\.\s+(.*)$/);
    if (ordered) {
      if (list !== "ol") {
        closeList();
        html += "<ol>";
        list = "ol";
      }
      const id = inRefs ? ` id="ref-${ordered[1]}"` : "";
      html += `<li${id}>${inline(ordered[2])}</li>`;
      continue;
    }
    if (line.startsWith("- ")) {
      if (list !== "ul") {
        closeList();
        html += "<ul>";
        list = "ul";
      }
      html += `<li>${inline(line.slice(2))}</li>`;
      continue;
    }
    if (line.startsWith("  - ")) {
      html += `<li>${inline(line.slice(4))}</li>`;
      continue;
    }
    closeList();
    html += `<p>${inline(line)}</p>`;
  }
  closeList();
  return html;
}

function inline(text) {
  let safe = escapeHtml(text);
  safe = safe.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  safe = safe.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  safe = safe.replace(/\[(\d+)\]/g, '<a class="cite" href="#ref-$1">[$1]</a>');
  return safe;
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[char]));
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const text = await response.text();
  let body = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch (_error) {
      body = null;
    }
  }
  if (!response.ok) {
    const detail = body?.detail;
    const message = typeof detail === "string" ? detail : `Request failed (${response.status})`;
    throw new Error(message);
  }
  return body;
}
