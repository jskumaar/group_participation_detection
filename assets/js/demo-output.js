/* Proto explorer, queue simulator, and label scoring. */
(function () {
  "use strict";
  var el = UI.el, M = window.Model;

  /* ---------------------------------------------------- proto explorer */
  var FIELDS = [
    { n: "frame_index", ex: "1843", d: "OpenCV's frame position after reading this frame. It starts at 2, and it should rise by exactly 1 each update. The evaluator uses that to detect dropped frames and seeks.", r: "publish_call" },
    { n: "playback_timestamp_ns", ex: "61494000000", d: "Position in the video, in nanoseconds, from the video file. This is the time that labels are matched against.", r: "video_ts" },
    { n: "gazes[].person_id", ex: "1", d: "Stable display ID, from 1 to num_people.", r: "sort_update" },
    { n: "gazes[].origin", ex: "(-47.7, -20.5, -94.7)", d: "Head position in centimetres, with the camera at the origin.", r: "add_gaze" },
    { n: "gazes[].direction", ex: "(0.25, -0.28, 0.93)", d: "Unit gaze direction in world coordinates.", r: "global_gaze" },
    { n: "interactions[].from_person_id", ex: "1", d: "The observer.", r: "bayes_infer" },
    { n: "interactions[].to_person_id", ex: "2", d: "The candidate target. There is no edge for the null target: its probability is whatever is left over.", r: "bayes_infer" },
    { n: "interactions[].angle_deg", ex: "11.8", d: "Angle between the gaze ray and the line to the target. Useful for debugging. It does not feed into is_looking.", r: "bayes_infer" },
    { n: "interactions[].is_looking", ex: "true", d: "True if this is the observer's most likely target. At most one per observer, and none if no one is most likely.", r: "bayes_infer" },
    { n: "interactions[].probability", ex: "0.74", d: "The Bayesian posterior that the observer is looking at this target.", r: "proto_edge" }
  ];

  function buildProto() {
    var root = document.getElementById("demo-proto");
    root.appendChild(el("div", { class: "demo-head" }, [
      el("h4", { text: "One PipelineUpdate, field by field" }),
      el("p", { text: "Click a field to see what it means and where it is set." })
    ]));
    var body = el("div", { class: "demo-body cols even" });
    var list = el("div", { class: "tree", role: "group", "aria-label": "Message fields", style: "max-height:none" });
    var detail = el("div");
    var buttons = FIELDS.map(function (f, i) {
      var b = el("button", { type: "button", text: f.n + ": " + f.ex });
      b.addEventListener("click", function () { show(i); });
      list.appendChild(b);
      return b;
    });
    function show(i) {
      buttons.forEach(function (b, k) { b.setAttribute("aria-current", String(k === i)); });
      var f = FIELDS[i];
      detail.textContent = "";
      detail.appendChild(el("h4", { text: f.n, style: "margin-top:0;overflow-wrap:anywhere" }));
      detail.appendChild(el("p", { text: f.d, style: "font-size:.92rem" }));
      detail.appendChild(el("p", { class: "note" }, [document.createTextNode("See "), UI.refLink(f.r)]));
    }
    body.appendChild(list); body.appendChild(detail);
    root.appendChild(body);
    show(9);
  }

  /* ---------------------------------------------------- queue simulator */
  function buildQueue() {
    var root = document.getElementById("demo-queue");
    root.appendChild(el("div", { class: "demo-head" }, [
      el("h4", { text: "Queue simulator" }),
      el("p", { text: "The rules match RapportStreamImpl: publish adds a frame, and if the queue is full the oldest frame is dropped first. The queue here is small so that you can see it fill." })
    ]));
    var body = el("div", { class: "demo-body cols" });
    var left = el("div"), right = el("div");
    var canvas = el("canvas", { role: "img", "aria-label": "Queue fill level" });
    left.appendChild(el("p", { class: "fig-label", text: "Queue, oldest frame on the left" }));
    left.appendChild(el("div", { class: "canvas-frame" }, [canvas]));
    var stats = el("div", { class: "stats", style: "margin-top:10px" });
    left.appendChild(stats);
    var warning = el("div", { class: "callout", style: "margin-bottom:0" });
    left.appendChild(warning);
    var row = el("div", { class: "btn-row" });
    var run = el("button", { class: "btn primary", type: "button", text: "Start" });
    var stall = el("button", { class: "btn", type: "button", text: "Stall client 3 s" });
    var reset = el("button", { class: "btn", type: "button", text: "Reset" });
    [run, stall, reset].forEach(function (b) { row.appendChild(b); });
    right.appendChild(row);
    var pub = UI.slider(right, { label: "Tracker publishes", min: 2, max: 40, value: 10, format: function (v) { return v + " frames/s"; } });
    var con = UI.slider(right, { label: "Client reads", min: 1, max: 80, value: 40, format: function (v) { return v + " frames/s"; } });
    var cap = UI.slider(right, { label: "RAPPORT_GRPC_MAX_QUEUED", min: 4, max: 64, value: 24 });
    var connected = el("input", { type: "checkbox", checked: "checked" });
    right.appendChild(el("label", { class: "toggle" }, [connected, el("span", { text: "Client is connected" })]));
    body.appendChild(left); body.appendChild(right);
    root.appendChild(body);

    var s, timer = null;
    function clear() {
      s = { next: 2, queue: [], delivered: 0, dropped: 0, gaps: 0, last: null, pubAcc: 0, conAcc: 0, stallUntil: 0, time: 0 };
      draw();
    }
    function tick() {
      var dt = 0.05;
      s.time += dt;
      s.pubAcc += pub.get() * dt;
      while (s.pubAcc >= 1) {
        s.pubAcc -= 1;
        if (s.queue.length >= cap.get()) { s.queue.shift(); s.dropped++; }
        s.queue.push(s.next++);
      }
      while (s.queue.length > cap.get()) { s.queue.shift(); s.dropped++; }
      if (connected.checked && s.time >= s.stallUntil) {
        s.conAcc += con.get() * dt;
        while (s.conAcc >= 1 && s.queue.length) {
          s.conAcc -= 1;
          var f = s.queue.shift();
          if (s.last !== null && f !== s.last + 1) s.gaps++;
          s.last = f;
          s.delivered++;
        }
        s.conAcc = Math.min(s.conAcc, 1);
      }
      draw();
    }
    function draw() {
      var W = 520, H = 58, ctx = UI.hidpi(canvas, W, H);
      ctx.fillStyle = UI.css("--surface"); ctx.fillRect(0, 0, W, H);
      var n = cap.get(), w = (W - 16) / n;
      for (var i = 0; i < n; i++) {
        ctx.fillStyle = i < s.queue.length ? UI.css("--accent") : UI.css("--surface-2");
        ctx.fillRect(8 + i * w, 12, Math.max(1, w - 2), 34);
      }
      stats.textContent = "";
      [["Published", s.next - 2], ["Delivered", s.delivered], ["Dropped", s.dropped], ["frame_index jumps", s.gaps]].forEach(function (x) {
        stats.appendChild(el("div", { class: "stat" }, [el("b", { text: String(x[1]) }), el("span", { text: x[0] })]));
      });
      warning.className = "callout " + (s.gaps ? "warn" : s.dropped ? "warn" : "good");
      warning.textContent = "";
      warning.appendChild(el("p", {
        text: s.gaps
          ? "The final report will warn: " + s.gaps + " jumps in frame_index. Those frames were never scored."
          : s.dropped
            ? "Frames are being dropped, and the client has not noticed yet. It will see the jump on the next read."
            : "Every frame reaches the client."
      }));
    }
    function stop() { if (timer) { clearInterval(timer); timer = null; run.textContent = "Start"; } }
    run.addEventListener("click", function () {
      if (timer) { stop(); return; }
      run.textContent = "Pause";
      timer = setInterval(tick, 50);
    });
    stall.addEventListener("click", function () { s.stallUntil = s.time + 3; if (!timer) run.click(); });
    reset.addEventListener("click", function () { stop(); clear(); });
    cap.input.addEventListener("input", draw);
    UI.onTheme(draw);
    clear();
  }

  /* ---------------------------------------------------- scoring timeline */
  var LABELS = [
    { start: 0, end: 5, src: 1, target: null },
    { start: 5, end: 10, src: 1, target: 2 },
    { start: 11, end: 15, src: 1, target: 3 },
    { start: 14, end: 16, src: 1, target: null },
    { start: 16, end: 20, src: 1, target: 2 }
  ];
  var SECONDS = 24;

  function buildEval() {
    var root = document.getElementById("demo-eval");
    root.appendChild(el("div", { class: "demo-head" }, [
      el("h4", { text: "Score P1's predictions" }),
      el("p", { text: "Each cell is one second of a made-up timeline. Click a prediction to cycle it through P2, P3 and no one." })
    ]));
    var body = el("div", { class: "demo-body" });
    var scroller = el("div", { class: "scroll-x" });
    var grid = el("div", { class: "tl" });
    scroller.appendChild(grid);
    body.appendChild(scroller);
    var pred = [null, null, null, 2, null, 2, 2, 2, 3, 2, 2, 2, 3, 3, 3, null, 2, 2, 2, 3, 2, 2, 3, 3];

    function row(name) {
      var cells = el("div", { class: "tl-cells" });
      grid.appendChild(el("div", { class: "tl-row" }, [el("span", { text: name }), cells]));
      return cells;
    }
    var axis = row("second");
    var labelRow = row("label");
    var predRow = row("prediction");
    var verdictRow = row("result");
    for (var a = 0; a < SECONDS; a++) axis.appendChild(el("div", { style: "text-align:center;font:.66rem var(--mono);color:var(--ink-3)", text: String(a) }));

    function short(t) { return t === null ? "–" : "P" + t; }
    for (var i = 0; i < SECONDS; i++) {
      var set = M.acceptedTargets(LABELS, i + 0.5).get(1);
      var targets = set ? Array.from(set) : null;
      var cell;
      if (!targets) cell = el("div", { class: "cell gap", text: "", title: "No label: not scored" });
      else if (targets.length > 1) cell = el("div", { class: "cell split", text: "P3|–", title: "Overlap: P3 or no one both count" });
      else cell = el("div", { class: "cell " + UI.personClass(targets[0]), text: short(targets[0]), title: targets[0] === null ? "Robot label: no one" : "" });
      labelRow.appendChild(cell);
    }
    var predButtons = pred.map(function (p, i) {
      var b = el("button", { class: "cell", type: "button", "aria-label": "Second " + i + " prediction" });
      b.addEventListener("click", function () {
        pred[i] = pred[i] === 2 ? 3 : pred[i] === 3 ? null : 2;
        update();
      });
      predRow.appendChild(b);
      return b;
    });
    var verdicts = pred.map(function () {
      var d = el("div", { class: "cell" });
      verdictRow.appendChild(d);
      return d;
    });
    var stats = el("div", { class: "stats", style: "margin-top:12px" });
    var logBox = el("div", { style: "margin-top:12px" });
    logBox.appendChild(el("p", { class: "fig-label", text: "report_N.jsonl (fields shortened)" }));
    var log = el("div", { class: "log", tabindex: 0 });
    logBox.appendChild(log);
    body.appendChild(stats);
    body.appendChild(logBox);
    root.appendChild(body);
    root.appendChild(el("div", { class: "demo-foot", html: "Key: hatched means unlabeled, – means no one. Uses ports of <code>accepted_targets</code> and <code>compare_update</code>. Runs of mismatched frames merge into one event, and a real report has one event per pair." }));

    function edgesAt(i) {
      return [2, 3].map(function (to) {
        return { from: 1, to: to, isLooking: pred[i] === to, probability: pred[i] === to ? 0.7 : 0.1 };
      });
    }
    function update() {
      var scored = 0, correct = 0, skipped = 0, active = new Map(), lines = [];
      function flush(key) {
        var a = active.get(key);
        lines.push(JSON.stringify({ from: 1, to: Number(key), start_s: a.start, end_s: a.last, kinds: Array.from(a.kinds).sort() }));
        active.delete(key);
      }
      pred.forEach(function (p, i) {
        var t = i + 0.5;
        var b = predButtons[i];
        b.className = "cell " + UI.personClass(p);
        b.textContent = short(p);
        var set = M.acceptedTargets(LABELS, t).get(1);
        var v = verdicts[i];
        if (!set) { skipped++; v.className = "cell skip"; v.textContent = "skip"; }
        else {
          var ok = set.has(p);
          scored++;
          if (ok) correct++;
          v.className = "cell " + (ok ? "ok" : "no");
          v.textContent = ok ? "✓" : "✗";
        }
        var found = M.compareEdges(LABELS, t, edgesAt(i));
        var now = new Set(found.map(function (m) { return String(m.to); }));
        Array.from(active.keys()).forEach(function (k) { if (!now.has(k)) flush(k); });
        found.forEach(function (m) {
          var k = String(m.to);
          if (!active.has(k)) active.set(k, { start: t, last: t, kinds: new Set([m.kind]) });
          else { var a = active.get(k); a.last = t; a.kinds.add(m.kind); }
        });
      });
      Array.from(active.keys()).forEach(flush);
      stats.textContent = "";
      [["Accuracy", scored ? (correct / scored).toFixed(3) : "n/a"], ["Scored seconds", scored], ["Correct", correct], ["Not scored", skipped]].forEach(function (s) {
        stats.appendChild(el("div", { class: "stat" }, [el("b", { text: String(s[1]) }), el("span", { text: s[0] })]));
      });
      log.textContent = lines.length ? lines.join("\n") : "No mismatch events.";
    }
    update();
  }

  buildProto();
  buildQueue();
  buildEval();
})();
