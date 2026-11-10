"use strict";
const $ = (id) => document.getElementById(id);
const hints = {
  WARMUP: "Preparing models and inference.",
  INITIALIZED: "Ready. Click Detect to find the object.",
  "Looking for object": "Searching. Tracking starts automatically.",
  TRACKING: "Object found. Pose tracking is active.",
  RECOVERING: "目标暂时失效，正在尝试恢复",
  ERROR: "The session needs attention. See the error below.",
  STOPPED: "The session has stopped."
};
let currentStatus = "WARMUP";
let lastState = null;
let lastVersion = -1;
let imageUrl = null;
let online = false;
let submitting = false;
let changingMode = false;
let selecting = false;
let frozen = null;
let bbox = null;
let anchor = null;
let selectionEpoch = 0;
let localError = null;

function showError(error) {
  localError = error.message;
  $("error").hidden = false;
  $("error").textContent = localError;
}

function clearSelection() {
  selectionEpoch++;
  frozen = bbox = anchor = null;
  $("selection").hidden = true;
  lastVersion = -1;
}

function hideFrame(message) {
  $("frame").hidden = true;
  $("selection").hidden = true;
  $("placeholder").hidden = false;
  $("placeholder-text").textContent = message;
}

async function api(path, body) {
  const response = await fetch(path, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {}), signal: AbortSignal.timeout(3000)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Session changed. Please try again.");
  return result;
}

async function displayImage(response, accept = () => true) {
  const nextUrl = URL.createObjectURL(await response.blob());
  try {
    const decoded = new Image();
    decoded.src = nextUrl;
    await decoded.decode();
    if (!accept()) {
      URL.revokeObjectURL(nextUrl);
      return false;
    }
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    imageUrl = nextUrl;
    $("frame").src = nextUrl;
    $("frame").hidden = false;
    $("placeholder").hidden = true;
    return true;
  } catch (error) {
    URL.revokeObjectURL(nextUrl);
    throw error;
  }
}

function drawBox() {
  if (!frozen) return;
  const canvas = $("selection");
  const area = canvas.parentElement.getBoundingClientRect();
  const scale = Math.min((canvas.parentElement.clientWidth || area.width) / frozen.width,
    (canvas.parentElement.clientHeight || area.height) / frozen.height);
  canvas.width = frozen.width;
  canvas.height = frozen.height;
  canvas.style.width = `${frozen.width * scale}px`;
  canvas.style.height = `${frozen.height * scale}px`;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (bbox) {
    context.strokeStyle = "#b7e8a5";
    context.fillStyle = "#b7e8a520";
    context.lineWidth = 2 / scale;
    context.fillRect(...bbox);
    context.strokeRect(...bbox);
  }
}

function updateControls() {
  const ready = online && lastState && lastState.can_detect && !submitting && !changingMode && !selecting;
  $("detect").disabled = !ready || (lastState.manual && !bbox);
  $("manual").disabled = !online || !lastState || !lastState.can_change_mode || submitting || changingMode || selecting;
  $("draw").hidden = !lastState || !lastState.manual;
  $("draw").disabled = !ready;
}

async function freezeFrame() {
  if (selecting || !online || !lastState?.manual || !lastState.can_detect) return;
  clearSelection();
  const epoch = selectionEpoch;
  selecting = true;
  localError = null;
  updateControls();
  try {
    const selection = await api("/api/selection");
    const response = await fetch(`/api/selection.jpg?token=${selection.token}`, {
      cache: "no-store", signal: AbortSignal.timeout(3000)
    });
    if (!response.ok) throw new Error("Frame expired. Please draw again.");
    if (await displayImage(response, () => epoch === selectionEpoch && online && lastState?.manual)) {
      frozen = selection;
      $("selection").hidden = false;
      drawBox();
    }
  } catch (error) {
    showError(error);
  } finally {
    selecting = false;
    updateControls();
  }
}

function render(state) {
  const changedMode = lastState && lastState.manual !== state.manual;
  const newNotice = state.notice && state.notice !== lastState?.notice;
  lastState = state;
  if (changedMode || newNotice || !state.manual || !state.can_detect || (frozen && state.selection_token > frozen.token)) {
    clearSelection();
  }
  $("status").textContent = state.status;
  $("status-card").dataset.status = state.status;
  $("hint").textContent = state.notice || (state.manual && state.status === "INITIALIZED"
    ? "在画面上拖动框选目标，然后点击 Detect。" : hints[state.status] || "");
  if (!changingMode) $("manual").checked = state.manual;
  $("demo").hidden = !state.demo;
  $("error").hidden = !state.error && !localError;
  $("error").textContent = state.error || localError || "";
  updateControls();
  if (state.status !== currentStatus) {
    $("warmup-block").open = state.status === "WARMUP";
    currentStatus = state.status;
  }
  $("warmup").textContent = state.timings.WARMUP.join("\n") || "Waiting for timing…";
  $("search").textContent = state.timings.INIT.join("\n");
  $("search-title").textContent = state.search_part;
  $("search-block").hidden = !state.timings.INIT.length;
  $("tracking").textContent = state.timings.TRACKING.join("\n");
  $("tracking-block").hidden = !state.timings.TRACKING.length;
  if (["WARMUP", "ERROR", "STOPPED"].includes(state.status)) {
    hideFrame(state.status === "WARMUP" ? "Waiting for warmup" : "No preview available");
  }
}

async function poll() {
  try {
    const response = await fetch("/api/status", {cache: "no-store", signal: AbortSignal.timeout(3000)});
    if (!response.ok) throw new Error("Status unavailable");
    const state = await response.json();
    online = true;
    $("connection").textContent = "Connected";
    render(state);
    if (state.manual && state.status === "INITIALIZED" && state.has_frame && !frozen && !selecting && !submitting && !changingMode) {
      await freezeFrame();
    } else if (!["WARMUP", "ERROR", "STOPPED"].includes(state.status) && !frozen && !selecting && state.image_version !== lastVersion) {
      const epoch = selectionEpoch;
      const image = await fetch(`/api/frame.jpg?v=${state.image_version}`, {
        cache: "no-store", signal: AbortSignal.timeout(3000)
      });
      if (image.status === 200) {
        if (await displayImage(image, () => epoch === selectionEpoch && !selecting && !frozen)) {
          lastVersion = state.image_version;
        }
      } else if (image.status === 204) {
        hideFrame("Waiting for preview");
      } else {
        throw new Error("Preview unavailable");
      }
    }
  } catch (error) {
    online = false;
    clearSelection();
    updateControls();
    $("connection").textContent = "Disconnected";
    $("hint").textContent = "Connection lost. Reconnecting…";
    hideFrame("Connection lost");
  } finally {
    window.setTimeout(poll, 200);
  }
}

$("manual").addEventListener("change", async () => {
  changingMode = true;
  localError = null;
  clearSelection();
  updateControls();
  try {
    await api("/api/mode", {manual: $("manual").checked});
  } catch (error) {
    $("manual").checked = lastState.manual;
    showError(error);
  } finally {
    changingMode = false;
  }
});

$("draw").addEventListener("click", freezeFrame);
$("detect").addEventListener("click", async () => {
  if (!online || submitting || $("detect").disabled) return;
  submitting = true;
  localError = null;
  updateControls();
  try {
    await api("/api/detect", lastState.manual ? {bbox, token: frozen.token} : {});
    clearSelection();
  } catch (error) {
    showError(error);
  } finally {
    submitting = false;
  }
});

function pixelPoint(event) {
  const rect = $("selection").getBoundingClientRect();
  return [Math.max(0, Math.min(frozen.width, Math.round((event.clientX - rect.left) * frozen.width / rect.width))),
    Math.max(0, Math.min(frozen.height, Math.round((event.clientY - rect.top) * frozen.height / rect.height)))];
}

$("selection").addEventListener("pointerdown", (event) => {
  if (!frozen || !online || submitting || !lastState.can_detect || event.button !== 0) return;
  anchor = pixelPoint(event);
  localError = null;
  $("error").hidden = true;
  bbox = null;
  $("selection").setPointerCapture(event.pointerId);
  updateControls();
  drawBox();
});
$("selection").addEventListener("pointermove", (event) => {
  if (!anchor || !frozen) return;
  const point = pixelPoint(event);
  bbox = [Math.min(anchor[0], point[0]), Math.min(anchor[1], point[1]),
    Math.abs(anchor[0] - point[0]), Math.abs(anchor[1] - point[1])];
  drawBox();
});
$("selection").addEventListener("pointerup", (event) => {
  if (!anchor || !frozen) return;
  const point = pixelPoint(event);
  bbox = [Math.min(anchor[0], point[0]), Math.min(anchor[1], point[1]),
    Math.abs(anchor[0] - point[0]), Math.abs(anchor[1] - point[1])];
  if (bbox[2] < 2 || bbox[3] < 2) bbox = null;
  anchor = null;
  drawBox();
  updateControls();
});
$("selection").addEventListener("pointercancel", () => {
  anchor = bbox = null;
  drawBox();
  updateControls();
});
window.addEventListener("resize", drawBox);
poll();
