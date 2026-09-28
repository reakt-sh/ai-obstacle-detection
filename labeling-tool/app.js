//99% made by Codex
"use strict";

const state = {
  manifest: null,
  candidates: [],
  filtered: [],
  currentId: null,
  annotation: null,
  image: null,
  selectedId: null,
  scale: 1,
  offsetX: 0,
  offsetY: 0,
  draggingVertex: null,
  panning: null,
  drawing: null,
  mergeSourceId: null,
  mergePending: false,
  maskClipboard: null,
  dirty: false,
  history: [],
  future: [],
};

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const context = canvas.getContext("2d");

function clone(value) { return JSON.parse(JSON.stringify(value)); }
function selectedInstance() {
  return state.annotation?.instances.find((item) => item.id === state.selectedId) || null;
}
function colorFor(instance) {
  if (instance.class_name === "rail") return "#35ef7d";
  const hue = (instance.class_id * 47 + 205) % 360;
  return `hsl(${hue} 90% 62%)`;
}
function imagePoint(event) {
  const rect = canvas.getBoundingClientRect();
  return [(event.clientX - rect.left - state.offsetX) / state.scale, (event.clientY - rect.top - state.offsetY) / state.scale];
}
function screenPoint(point) { return [point[0] * state.scale + state.offsetX, point[1] * state.scale + state.offsetY]; }

function resizeCanvas() {
  const rect = $("canvasShell").getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * ratio));
  canvas.height = Math.max(1, Math.round(rect.height * ratio));
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  draw();
}

function fitView() {
  if (!state.image) return;
  const rect = canvas.getBoundingClientRect();
  state.scale = Math.min(rect.width / state.image.width, rect.height / state.image.height) * 0.98;
  state.offsetX = (rect.width - state.image.width * state.scale) / 2;
  state.offsetY = (rect.height - state.image.height * state.scale) / 2;
  draw();
}

function polygonPath(points) {
  if (!points.length) return;
  const first = screenPoint(points[0]);
  context.beginPath();
  context.moveTo(first[0], first[1]);
  for (const point of points.slice(1)) {
    const screen = screenPoint(point);
    context.lineTo(screen[0], screen[1]);
  }
  context.closePath();
}

function draw() {
  const rect = canvas.getBoundingClientRect();
  context.clearRect(0, 0, rect.width, rect.height);
  if (!state.image || !state.annotation) return;
  context.drawImage(
    state.image,
    state.offsetX,
    state.offsetY,
    state.image.width * state.scale,
    state.image.height * state.scale,
  );
  const opacity = Number($("opacityInput").value);
  const orderedInstances = [...state.annotation.instances].sort((first, second) => {
    const firstActive = first.id === state.selectedId || first.id === state.mergeSourceId;
    const secondActive = second.id === state.selectedId || second.id === state.mergeSourceId;
    return Number(firstActive) - Number(secondActive);
  });
  for (const instance of orderedInstances) {
    const selected = instance.id === state.selectedId;
    const mergeSource = instance.id === state.mergeSourceId;
    const color = colorFor(instance);
    polygonPath(instance.points);
    context.globalAlpha = opacity;
    context.fillStyle = color;
    context.fill();
    context.globalAlpha = 1;
    context.strokeStyle = mergeSource ? "#ffe55d" : selected ? "#ffffff" : color;
    context.lineWidth = mergeSource || selected ? 3 : 1.6;
    context.setLineDash(mergeSource ? [9, 5] : []);
    context.stroke();
    context.setLineDash([]);
    if ($("labelsInput").checked && instance.points.length) {
      const anchor = screenPoint(instance.points.reduce((best, point) => point[1] < best[1] ? point : best));
      const label = `${instance.class_name}${instance.confidence == null ? "" : ` ${(instance.confidence * 100).toFixed(0)}%`}`;
      context.font = "600 13px system-ui";
      const width = context.measureText(label).width + 10;
      context.fillStyle = "rgba(0,0,0,.78)";
      context.fillRect(anchor[0], anchor[1] - 20, width, 20);
      context.fillStyle = "#fff";
      context.fillText(label, anchor[0] + 5, anchor[1] - 5);
    }
    if (selected && $("verticesInput").checked) {
      for (const point of instance.points) {
        const screen = screenPoint(point);
        context.beginPath();
        context.arc(screen[0], screen[1], 4.5, 0, Math.PI * 2);
        context.fillStyle = "#fff";
        context.fill();
        context.strokeStyle = color;
        context.lineWidth = 2;
        context.stroke();
      }
    }
  }
  if (state.drawing?.points.length) {
    context.beginPath();
    state.drawing.points.forEach((point, index) => {
      const screen = screenPoint(point);
      if (index === 0) context.moveTo(screen[0], screen[1]);
      else context.lineTo(screen[0], screen[1]);
    });
    context.strokeStyle = "#ffe55d";
    context.lineWidth = 2;
    context.stroke();
    for (const point of state.drawing.points) {
      const screen = screenPoint(point);
      context.fillStyle = "#ffe55d";
      context.beginPath();
      context.arc(screen[0], screen[1], 4, 0, Math.PI * 2);
      context.fill();
    }
  }
}

function pointInPolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0], yi = polygon[i][1];
    const xj = polygon[j][0], yj = polygon[j][1];
    if (((yi > point[1]) !== (yj > point[1])) &&
        (point[0] < (xj - xi) * (point[1] - yi) / ((yj - yi) || 1e-9) + xi)) inside = !inside;
  }
  return inside;
}

function nearestVertex(instance, event, radius = 11) {
  const rect = canvas.getBoundingClientRect();
  const mouse = [event.clientX - rect.left, event.clientY - rect.top];
  let best = null;
  instance.points.forEach((point, index) => {
    const screen = screenPoint(point);
    const distance = Math.hypot(mouse[0] - screen[0], mouse[1] - screen[1]);
    if (distance <= radius && (!best || distance < best.distance)) best = { index, distance };
  });
  return best;
}

function distanceToSegment(point, start, end) {
  const dx = end[0] - start[0], dy = end[1] - start[1];
  const denominator = dx * dx + dy * dy;
  const t = denominator ? Math.max(0, Math.min(1, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator)) : 0;
  const projected = [start[0] + t * dx, start[1] + t * dy];
  return { distance: Math.hypot(point[0] - projected[0], point[1] - projected[1]), point: projected };
}

function snapshot() {
  if (!state.annotation) return;
  state.history.push(clone(state.annotation));
  if (state.history.length > 60) state.history.shift();
  state.future = [];
  updateButtons();
}
function markDirty() { state.dirty = true; $("saveButton").textContent = "Speichern •"; }
function restore(annotation) {
  state.annotation = clone(annotation);
  if (!selectedInstance()) state.selectedId = null;
  markDirty();
  renderInspector(); draw(); updateButtons();
}
function undo() {
  if (!state.history.length) return;
  state.future.push(clone(state.annotation));
  restore(state.history.pop());
}
function redo() {
  if (!state.future.length) return;
  state.history.push(clone(state.annotation));
  restore(state.future.pop());
}

async function loadManifest(keepId = null) {
  const response = await fetch("/api/manifest", { cache: "no-store" });
  if (!response.ok) throw new Error("Manifest konnte nicht geladen werden");
  state.manifest = await response.json();
  state.candidates = state.manifest.candidates;
  applyFilter();
  updateProgress();
  if (keepId && state.candidates.some((item) => item.id === keepId)) state.currentId = keepId;
}

function applyFilter() {
  const value = $("filterSelect").value;
  state.filtered = state.candidates.filter((item) => {
    if (value === "deferred") return Boolean(item.deferred);
    if (item.deferred) return false;
    if (value === "all") return true;
    if (value === "unreviewed") return !item.reviewed;
    if (value === "reviewed") return item.reviewed && !item.excluded;
    if (value === "excluded") return item.excluded;
    return item.reasons.some((reason) => reason.type === value);
  });
  renderCandidateList();
  updateProgress();
}

function updateProgress() {
  const showDeferred = $("filterSelect").value === "deferred";
  const scope = state.candidates.filter((item) => Boolean(item.deferred) === showDeferred);
  const total = scope.length;
  const reviewed = scope.filter((item) => item.reviewed).length;
  const accepted = scope.filter((item) => item.reviewed && !item.excluded).length;
  const deferred = state.candidates.filter((item) => item.deferred).length;
  const datasetName = state.manifest?.name ? `${state.manifest.name} · ` : "";
  const deferredText = deferred ? ` · ${deferred} zurückgestellt` : "";
  $("progressText").textContent = `${datasetName}${reviewed} von ${total} geprüft · ${accepted} übernommen${deferredText}`;
}

function renderCandidateList() {
  const list = $("candidateList");
  list.replaceChildren();
  for (const candidate of state.filtered) {
    const button = document.createElement("button");
    button.className = `candidate${candidate.id === state.currentId ? " active" : ""}`;
    const reasons = [...new Set(candidate.reasons.map((item) => item.label))].join(" · ");
    button.innerHTML = `<span class="time">${candidate.source_seconds.toFixed(2)} s</span><span class="status-dot ${candidate.excluded ? "excluded" : candidate.reviewed ? "reviewed" : ""}"></span><span class="small">${candidate.id}</span><span></span><span class="small">${escapeHtml(reasons)}</span>`;
    button.addEventListener("click", () => openCandidate(candidate.id));
    list.appendChild(button);
  }
  const active = list.querySelector(".candidate.active");
  if (active) active.scrollIntoView({ block: "nearest" });
}

function escapeHtml(text) {
  return String(text).replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[character]);
}

async function openCandidate(id) {
  if (id === state.currentId && state.annotation) return;
  if (state.dirty) await saveAnnotation(false);
  const response = await fetch(`/api/annotation/${encodeURIComponent(id)}`, { cache: "no-store" });
  if (!response.ok) throw new Error("Annotation konnte nicht geladen werden");
  const annotation = await response.json();
  const image = new Image();
  const datasetKey = `${state.manifest.video}|${state.manifest.created_at}`;
  image.src = `/data/${annotation.image}?dataset=${encodeURIComponent(datasetKey)}`;
  await image.decode();
  state.currentId = id;
  state.annotation = annotation;
  state.image = image;
  state.selectedId = null;
  state.history = [];
  state.future = [];
  state.drawing = null;
  state.mergeSourceId = null;
  state.mergePending = false;
  state.dirty = false;
  $("saveButton").textContent = "Speichern";
  $("notesInput").value = annotation.notes || "";
  fitView();
  renderCandidateList();
  renderInspector();
  updateButtons();
}

async function saveAnnotation(showMessage = true) {
  if (!state.annotation) return;
  state.annotation.notes = $("notesInput").value;
  const response = await fetch(`/api/annotation/${encodeURIComponent(state.annotation.id)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state.annotation),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Speichern fehlgeschlagen");
  state.dirty = false;
  $("saveButton").textContent = "Speichern";
  const candidate = state.candidates.find((item) => item.id === state.currentId);
  if (candidate) {
    candidate.reviewed = state.annotation.reviewed;
    candidate.excluded = state.annotation.excluded;
    candidate.instance_count = state.annotation.instances.length;
  }
  applyFilter(); updateProgress();
  if (showMessage) toast("Gespeichert");
}

function renderInspector() {
  if (!state.annotation) return;
  const instance = selectedInstance();
  $("noSelection").classList.toggle("hidden", Boolean(instance));
  $("selectionEditor").classList.toggle("hidden", !instance);
  if (instance) {
    $("classSelect").value = String(instance.class_id);
    $("instanceSource").textContent = instance.source || "manuell";
    $("instanceConfidence").textContent = instance.confidence == null ? "–" : `${(instance.confidence * 100).toFixed(1)} %`;
    $("instanceVertices").textContent = String(instance.points.length);
  }
  $("reasonList").replaceChildren(...state.annotation.reasons.map((reason) => {
    const node = document.createElement("div");
    node.className = `reason ${reason.type}`;
    node.textContent = reason.label;
    return node;
  }));
  $("frameMeta").textContent = `${state.annotation.id} · Frame ${state.annotation.source_frame} · ${state.annotation.source_seconds.toFixed(3)} s · ${state.annotation.instances.length} Masken`;
  $("acceptButton").textContent = state.annotation.reviewed && !state.annotation.excluded ? "✓ Übernommen" : "Geprüft & übernehmen";
  $("excludeButton").textContent = state.annotation.excluded ? "✕ Verworfen" : "Frame verwerfen";
  renderInstanceList();
}

function renderInstanceList() {
  const list = $("instanceList");
  list.replaceChildren();
  state.annotation.instances.forEach((instance, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `instance-item${instance.id === state.selectedId ? " selected" : ""}`;
    button.title = instance.id;

    const swatch = document.createElement("span");
    swatch.className = "instance-item__swatch";
    swatch.style.background = colorFor(instance);
    const name = document.createElement("span");
    name.className = "instance-item__name";
    name.textContent = `#${index + 1} · ${instance.class_name}`;
    const meta = document.createElement("span");
    meta.className = "instance-item__meta";
    const source = instance.source === "manual-copy" ? "Kopie" : instance.source || "manuell";
    meta.textContent = `${source} · ${instance.points.length} Punkte`;
    button.append(swatch, name, meta);
    button.addEventListener("click", () => {
      if (state.mergeSourceId) cancelMergeMode();
      state.selectedId = instance.id;
      renderInspector(); draw(); updateButtons();
    });
    list.appendChild(button);
  });
}

function updateButtons() {
  const instance = selectedInstance();
  $("deleteInstanceButton").disabled = !instance;
  $("copyMaskButton").disabled = !instance;
  $("pasteMaskButton").disabled = !state.maskClipboard || !state.annotation;
  $("selectionLockInput").disabled = !instance;
  $("mergeMasksButton").disabled = !instance || state.annotation.instances.length < 2 || state.mergePending;
  $("mergeMasksButton").textContent = state.mergeSourceId ? "Verbinden abbrechen" : "Masken verbinden";
  $("mergeMasksButton").classList.toggle("merge-active", Boolean(state.mergeSourceId));
  $("undoButton").disabled = !state.history.length;
  $("redoButton").disabled = !state.future.length;
  const index = state.filtered.findIndex((item) => item.id === state.currentId);
  $("previousButton").disabled = index <= 0;
  $("nextButton").disabled = index < 0 || index >= state.filtered.length - 1;
}

function navigate(offset) {
  const index = state.filtered.findIndex((item) => item.id === state.currentId);
  const target = state.filtered[index + offset];
  if (target) openCandidate(target.id).catch(showError);
}

function nextUnreviewed() {
  const current = state.candidates.findIndex((item) => item.id === state.currentId);
  const ordered = [...state.candidates.slice(current + 1), ...state.candidates.slice(0, current + 1)];
  const showDeferred = $("filterSelect").value === "deferred";
  const target = ordered.find((item) => !item.reviewed && Boolean(item.deferred) === showDeferred);
  if (target) openCandidate(target.id).catch(showError);
  else toast("Alle Frames dieses Stapels wurden geprüft");
}

function beginNewPolygon() {
  cancelMergeMode();
  state.drawing = { points: [] };
  state.selectedId = null;
  $("canvasHint").textContent = "Punkte anklicken · Enter beendet · Escape bricht ab";
  $("canvasHint").classList.remove("hidden");
  draw(); renderInspector();
}

function finishPolygon() {
  if (!state.drawing) return;
  if (state.drawing.points.length < 3) return toast("Mindestens drei Punkte setzen", true);
  snapshot();
  const fallbackClass = state.manifest.classes.find((item) => item.name === "rail")
    || state.manifest.classes[0];
  const classId = Number($("classSelect").value || fallbackClass?.id || 0);
  const classItem = state.manifest.classes.find((item) => item.id === classId);
  const instance = {
    id: `manual-${Date.now()}-${Math.random().toString(16).slice(2, 7)}`,
    class_id: classId,
    class_name: classItem?.name || String(classId),
    confidence: null,
    source: "manual",
    geometry: "polygon",
    points: state.drawing.points,
  };
  state.annotation.instances.push(instance);
  state.selectedId = instance.id;
  state.drawing = null;
  $("canvasHint").classList.add("hidden");
  markDirty(); renderInspector(); draw(); updateButtons();
}

function deleteSelected() {
  if (!selectedInstance()) return;
  snapshot();
  state.annotation.instances = state.annotation.instances.filter((item) => item.id !== state.selectedId);
  if (state.mergeSourceId === state.selectedId) state.mergeSourceId = null;
  state.selectedId = null;
  markDirty(); renderInspector(); draw(); updateButtons();
}

function copySelected() {
  const instance = selectedInstance();
  if (!instance) return toast("Zuerst eine Maske auswählen", true);
  state.maskClipboard = {
    instance: clone(instance),
    width: state.annotation.width,
    height: state.annotation.height,
  };
  updateButtons();
  toast(`${instance.class_name}-Maske kopiert`);
}

function pasteCopied() {
  if (!state.annotation || !state.maskClipboard) return toast("Keine kopierte Maske vorhanden", true);
  cancelMergeMode();
  snapshot();
  const copied = clone(state.maskClipboard.instance);
  const scaleX = state.annotation.width / state.maskClipboard.width;
  const scaleY = state.annotation.height / state.maskClipboard.height;
  copied.id = `copy-${Date.now()}-${Math.random().toString(16).slice(2, 7)}`;
  copied.points = copied.points.map(([x, y]) => [
    Math.max(0, Math.min(state.annotation.width, x * scaleX)),
    Math.max(0, Math.min(state.annotation.height, y * scaleY)),
  ]);
  copied.confidence = null;
  copied.source = "manual-copy";
  copied.geometry = "polygon";
  delete copied.bbox;
  state.annotation.instances.push(copied);
  state.selectedId = copied.id;
  markDirty(); renderInspector(); draw(); updateButtons();
  toast(`${copied.class_name}-Maske an gleicher Position eingefügt`);
}

function cancelMergeMode(showMessage = false) {
  if (!state.mergeSourceId) return;
  state.mergeSourceId = null;
  state.mergePending = false;
  $("canvasHint").classList.add("hidden");
  if (showMessage) toast("Zusammenführen abgebrochen");
  draw(); updateButtons();
}

function toggleMergeMode() {
  if (state.mergeSourceId) return cancelMergeMode(true);
  const source = selectedInstance();
  if (!source) return toast("Zuerst die erste Maske auswählen", true);
  const compatible = state.annotation.instances.some(
    (item) => item.id !== source.id && item.class_id === source.class_id,
  );
  if (!compatible) return toast("Keine zweite Maske derselben Klasse vorhanden", true);
  state.mergeSourceId = source.id;
  $("canvasHint").textContent = `Erste ${source.class_name}-Maske markiert · jetzt zweite Maske anklicken · Escape bricht ab`;
  $("canvasHint").classList.remove("hidden");
  draw(); updateButtons();
}

async function mergeWith(instanceId) {
  if (state.mergePending) return;
  const source = state.annotation.instances.find((item) => item.id === state.mergeSourceId);
  const target = state.annotation.instances.find((item) => item.id === instanceId);
  if (!source || !target || source.id === target.id) return;
  if (source.class_id !== target.class_id) {
    return toast("Nur Masken derselben Klasse können verbunden werden", true);
  }

  state.mergePending = true;
  updateButtons();
  try {
    const response = await fetch("/api/merge-polygons", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        width: state.annotation.width,
        height: state.annotation.height,
        polygons: [source.points, target.points],
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Masken konnten nicht verbunden werden");

    snapshot();
    const merged = clone(source);
    merged.points = result.points;
    merged.confidence = null;
    merged.source = "manual-merged";
    merged.geometry = "polygon";
    delete merged.bbox;
    state.annotation.instances = state.annotation.instances
      .filter((item) => item.id !== target.id)
      .map((item) => item.id === source.id ? merged : item);
    state.selectedId = source.id;
    state.mergeSourceId = null;
    markDirty(); renderInspector(); draw();
    toast(result.snapped ? "Masken verbunden; kleiner Spalt wurde geschlossen" : "Masken verbunden");
  } finally {
    state.mergePending = false;
    if (!state.mergeSourceId) $("canvasHint").classList.add("hidden");
    updateButtons();
  }
}

canvas.addEventListener("pointerdown", (event) => {
  if (!state.annotation) return;
  if (event.button === 1 || event.shiftKey) {
    state.panning = { x: event.clientX, y: event.clientY, offsetX: state.offsetX, offsetY: state.offsetY };
    $("canvasShell").classList.add("panning");
    return;
  }
  if (event.button !== 0) return;
  const point = imagePoint(event);
  if (state.mergeSourceId) {
    const hit = [...state.annotation.instances].reverse().find(
      (instance) => instance.id !== state.mergeSourceId && pointInPolygon(point, instance.points),
    );
    if (!hit) return toast("Bitte die zweite Maske anklicken", true);
    mergeWith(hit.id).catch(showError);
    return;
  }
  if (state.drawing) {
    state.drawing.points.push(point);
    draw();
    return;
  }
  const current = selectedInstance();
  if (current) {
    const vertex = nearestVertex(current, event);
    if (vertex) {
      snapshot();
      state.draggingVertex = { instanceId: current.id, index: vertex.index };
      canvas.setPointerCapture(event.pointerId);
      return;
    }
    if ($("selectionLockInput").checked || pointInPolygon(point, current.points)) {
      draw();
      return;
    }
  }
  const hit = [...state.annotation.instances].reverse().find((instance) => pointInPolygon(point, instance.points));
  state.selectedId = hit?.id || null;
  renderInspector(); draw(); updateButtons();
});

canvas.addEventListener("pointermove", (event) => {
  if (state.panning) {
    state.offsetX = state.panning.offsetX + event.clientX - state.panning.x;
    state.offsetY = state.panning.offsetY + event.clientY - state.panning.y;
    draw(); return;
  }
  if (!state.draggingVertex) return;
  const instance = state.annotation.instances.find((item) => item.id === state.draggingVertex.instanceId);
  if (!instance) return;
  const point = imagePoint(event);
  instance.points[state.draggingVertex.index] = [
    Math.max(0, Math.min(state.annotation.width, point[0])),
    Math.max(0, Math.min(state.annotation.height, point[1])),
  ];
  markDirty(); renderInspector(); draw();
});

function endPointer() {
  state.draggingVertex = null;
  state.panning = null;
  $("canvasShell").classList.remove("panning");
}
canvas.addEventListener("pointerup", endPointer);
canvas.addEventListener("pointercancel", endPointer);

canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  if (!state.image) return;
  const rect = canvas.getBoundingClientRect();
  const mouse = [event.clientX - rect.left, event.clientY - rect.top];
  const before = [(mouse[0] - state.offsetX) / state.scale, (mouse[1] - state.offsetY) / state.scale];
  const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
  state.scale = Math.max(0.05, Math.min(8, state.scale * factor));
  state.offsetX = mouse[0] - before[0] * state.scale;
  state.offsetY = mouse[1] - before[1] * state.scale;
  draw();
}, { passive: false });

canvas.addEventListener("dblclick", (event) => {
  const instance = selectedInstance();
  if (!instance || state.drawing) return;
  const point = imagePoint(event);
  let best = null;
  instance.points.forEach((start, index) => {
    const end = instance.points[(index + 1) % instance.points.length];
    const result = distanceToSegment(point, start, end);
    if (!best || result.distance < best.distance) best = { ...result, index };
  });
  if (best && best.distance * state.scale < 16) {
    snapshot();
    instance.points.splice(best.index + 1, 0, best.point);
    markDirty(); renderInspector(); draw(); updateButtons();
  }
});

canvas.addEventListener("contextmenu", (event) => {
  event.preventDefault();
  const instance = selectedInstance();
  if (!instance || instance.points.length <= 3) return;
  const vertex = nearestVertex(instance, event, 13);
  if (!vertex) return;
  snapshot();
  instance.points.splice(vertex.index, 1);
  markDirty(); renderInspector(); draw(); updateButtons();
});

$("classSelect").addEventListener("change", () => {
  const instance = selectedInstance();
  if (!instance) return;
  snapshot();
  const classId = Number($("classSelect").value);
  const item = state.manifest.classes.find((entry) => entry.id === classId);
  instance.class_id = classId;
  instance.class_name = item.name;
  instance.confidence = null;
  instance.source = "manual";
  markDirty(); renderInspector(); draw(); updateButtons();
});
$("notesInput").addEventListener("input", markDirty);
$("opacityInput").addEventListener("input", draw);
$("verticesInput").addEventListener("change", draw);
$("labelsInput").addEventListener("change", draw);
$("selectionLockInput").addEventListener("change", () => {
  toast($("selectionLockInput").checked ? "Auswahl im Bild gesperrt" : "Auswahlsperre aufgehoben");
});
$("filterSelect").addEventListener("change", () => {
  applyFilter();
  if (state.filtered.length && !state.filtered.some((item) => item.id === state.currentId)) {
    openCandidate(state.filtered[0].id).catch(showError);
  }
});
$("previousButton").addEventListener("click", () => navigate(-1));
$("nextButton").addEventListener("click", () => navigate(1));
$("nextUnreviewedButton").addEventListener("click", nextUnreviewed);
$("fitButton").addEventListener("click", fitView);
$("newPolygonButton").addEventListener("click", beginNewPolygon);
$("mergeMasksButton").addEventListener("click", toggleMergeMode);
$("copyMaskButton").addEventListener("click", copySelected);
$("pasteMaskButton").addEventListener("click", pasteCopied);
$("deleteInstanceButton").addEventListener("click", deleteSelected);
$("undoButton").addEventListener("click", undo);
$("redoButton").addEventListener("click", redo);
$("saveButton").addEventListener("click", () => saveAnnotation().catch(showError));
$("acceptButton").addEventListener("click", async () => {
  snapshot(); state.annotation.reviewed = true; state.annotation.excluded = false; markDirty(); renderInspector();
  await saveAnnotation(); nextUnreviewed();
});
$("excludeButton").addEventListener("click", async () => {
  snapshot(); state.annotation.reviewed = true; state.annotation.excluded = true; markDirty(); renderInspector();
  await saveAnnotation(); nextUnreviewed();
});

window.addEventListener("keydown", (event) => {
  const key = event.key.toLowerCase();
  const modifier = event.ctrlKey || event.metaKey;
  const formControlFocused = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
  if (formControlFocused && !(modifier && key === "s")) return;
  if (modifier && key === "s") { event.preventDefault(); saveAnnotation().catch(showError); }
  else if (modifier && key === "c") { event.preventDefault(); copySelected(); }
  else if (modifier && key === "v") { event.preventDefault(); pasteCopied(); }
  else if (modifier && key === "z") { event.preventDefault(); event.shiftKey ? redo() : undo(); }
  else if (modifier && key === "y") { event.preventDefault(); redo(); }
  else if (event.key === "ArrowLeft") navigate(-1);
  else if (event.key === "ArrowRight") navigate(1);
  else if (key === "n") beginNewPolygon();
  else if (key === "m") toggleMergeMode();
  else if (key === "f") fitView();
  else if (event.key === "Delete") deleteSelected();
  else if (event.key === "Enter" && state.drawing) finishPolygon();
  else if (event.key === "Escape" && state.drawing) { state.drawing = null; $("canvasHint").classList.add("hidden"); draw(); }
  else if (event.key === "Escape" && state.mergeSourceId) cancelMergeMode(true);
  else if (key === "r" && state.annotation) $("acceptButton").click();
  else if (key === "x" && state.annotation) $("excludeButton").click();
});

window.addEventListener("resize", resizeCanvas);
window.addEventListener("beforeunload", (event) => { if (state.dirty) { event.preventDefault(); event.returnValue = ""; } });

function toast(message, error = false) {
  const node = $("toast");
  node.textContent = message;
  node.classList.toggle("error", error);
  node.classList.remove("hidden");
  clearTimeout(toast.timeout);
  toast.timeout = setTimeout(() => node.classList.add("hidden"), 2600);
}
function showError(error) { console.error(error); toast(error.message || String(error), true); }

async function start() {
  await loadManifest();
  for (const item of state.manifest.classes) {
    const option = document.createElement("option");
    option.value = String(item.id);
    option.textContent = `${item.id}: ${item.name}`;
    $("classSelect").appendChild(option);
  }
  const defaultClass = state.manifest.classes.find((item) => item.name === "rail")
    || state.manifest.classes[0];
  if (defaultClass) $("classSelect").value = String(defaultClass.id);
  resizeCanvas();
  const first = state.filtered[0];
  if (first) await openCandidate(first.id);
}

start().catch(showError);
