// Exercise the browser UI contract without downloading browser dependencies.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

async function main() {
  const root = path.join(__dirname, "../wapr_realsense/static");
  const elements = {};
  for (const match of fs.readFileSync(path.join(root, "index.html"), "utf8").matchAll(/<[^>]*\bid="([^"]+)"[^>]*>/g)) {
    elements[match[1]] = {hidden: /\bhidden\b/.test(match[0]), disabled: /\bdisabled\b/.test(match[0]),
      dataset: {}, style: {}, textContent: "", addEventListener(event, callback) { this[event] = callback; },
      parentElement: {getBoundingClientRect: () => ({width: 320, height: 240})},
      getBoundingClientRect: () => ({left: 10, top: 20, width: 320, height: 240}),
      getContext: () => ({clearRect() {}, fillRect() {}, strokeRect() {}}), setPointerCapture() {}};
  }
  let state;
  let nextPoll;
  let imageRequests = 0;
  let detectRequests = 0;
  let disconnected = false;
  let selectionToken = 0;
  let lastDetect = null;
  const context = {
    AbortSignal,
    document: {getElementById: (id) => elements[id]},
    window: {setTimeout: (callback) => { nextPoll = callback; }, addEventListener() {}},
    URL: {createObjectURL: () => "blob:preview", revokeObjectURL() {}},
    Image: class { async decode() {} },
    fetch: async (url, options) => {
      if (disconnected) throw new Error("offline");
      if (url === "/api/detect") {
        assert.equal(options.method, "POST");
        detectRequests++;
        lastDetect = JSON.parse(options.body);
        return {ok: true, status: 202, json: async () => ({accepted: true})};
      }
      if (url === "/api/mode") {
        state.manual = JSON.parse(options.body).manual;
        state.status = "INITIALIZED";
        return {ok: true, json: async () => ({accepted: true})};
      }
      if (url === "/api/selection") {
        state.selection_token = ++selectionToken;
        return {ok: true, json: async () => ({token: selectionToken, width: 640, height: 480})};
      }
      if (url === "/api/status") return {ok: true, json: async () => state};
      imageRequests++;
      return {status: 200, ok: true, blob: async () => ({})};
    }
  };
  function setState(status, version) {
    const ready = ["INITIALIZED", "TRACKING"].includes(status);
    state = {status, image_version: version, can_detect: ready, can_change_mode: ready,
      manual: false, selection_token: null, has_frame: version > 0,
      demo: true, timings: {WARMUP: ["Startup total=12 ms"], INIT: [], TRACKING: []}, search_part: "INIT"};
  }
  async function settle() { await new Promise(setImmediate); }
  setState("WARMUP", 0);
  vm.runInNewContext(fs.readFileSync(path.join(root, "app.js"), "utf8"), context);
  await settle();
  assert.equal(elements.detect.disabled, true);
  assert.equal(elements.frame.hidden, true);
  assert.equal(imageRequests, 0);
  assert.equal(elements.status.textContent, "WARMUP");
  setState("INITIALIZED", 1);
  await nextPoll();
  assert.equal(elements.detect.disabled, false);
  assert.equal(elements.frame.hidden, false);
  await elements.detect.click();
  assert.equal(detectRequests, 1);
  assert.equal(elements.detect.disabled, true);
  setState("Looking for object", 2);
  await nextPoll();
  assert.equal(elements.status.textContent, "Looking for object");
  assert.equal(elements.detect.disabled, true);
  setState("TRACKING", 3);
  state.timings.TRACKING = ["WAPR avg=20 ms | frames=10"];
  await nextPoll();
  assert.equal(elements.status.textContent, "TRACKING");
  assert.equal(elements.detect.disabled, false);
  assert.equal(elements["tracking-block"].hidden, false);
  elements.manual.checked = true;
  await elements.manual.change();
  await nextPoll();
  assert.equal(elements.selection.hidden, false);
  assert.equal(elements.detect.disabled, true);
  elements.selection.pointerdown({button: 0, pointerId: 1, clientX: 110, clientY: 120});
  elements.selection.pointerup({clientX: 60, clientY: 70});
  assert.equal(elements.detect.disabled, false);
  await elements.detect.click();
  assert.deepEqual(lastDetect.bbox, [100, 100, 100, 100]);
  assert.equal(lastDetect.token, 1);
  setState("TRACKING", 4);
  state.manual = true;
  await nextPoll();
  setState("INITIALIZED", 5);
  state.manual = true;
  state.notice = "目标丢失，请重新手动框选";
  await nextPoll();
  assert.equal(elements.hint.textContent, "目标丢失，请重新手动框选");
  assert.equal(elements.detect.disabled, true);
  assert.equal(elements.selection.hidden, false);
  assert.equal(selectionToken, 2);
  elements.selection.pointerdown({button: 0, pointerId: 2, clientX: 50, clientY: 50});
  elements.selection.pointerup({clientX: 50, clientY: 50});
  assert.equal(elements.detect.disabled, true);
  elements.manual.checked = false;
  await elements.manual.change();
  await nextPoll();
  assert.equal(elements.selection.hidden, true);
  assert.equal(elements.detect.disabled, false);
  setState("Looking for object", 4);
  await nextPoll();
  assert.equal(elements.detect.disabled, true);
  disconnected = true;
  await nextPoll();
  assert.equal(elements.connection.textContent, "Disconnected");
  assert.equal(elements.frame.hidden, true);
  assert.equal(elements.detect.disabled, true);
  disconnected = false;
  setState("WARMUP", 0);
  await nextPoll();
  assert.equal(elements.frame.hidden, true);
  assert.equal(elements["warmup-block"].open, true);
  setState("ERROR", 1);
  state.error = "Camera disconnected";
  await nextPoll();
  assert.equal(elements.frame.hidden, true);
  assert.equal(elements.error.textContent, "Camera disconnected");
  console.log("Web UI lifecycle, Detect, timing and disconnection checks passed.");
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
