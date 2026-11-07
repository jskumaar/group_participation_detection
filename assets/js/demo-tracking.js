/* SORT and display ID demo. The ID logic follows Sort::update. A simple
 * constant-velocity predictor stands in for the 7-state Kalman filter. */
(function () {
  "use strict";
  var el = UI.el, M = window.Model;
  var W = 900, H = 170, SIZE = 58;

  var SCENARIOS = {
    normal: { name: "Everyone visible", hide: function () { return false; } },
    blink: { name: "P2 lost for 6 frames", hide: function (id, f) { return id === 2 && f >= 8 && f < 14; } },
    long: { name: "P2 lost, moves, returns", hide: function (id, f) { return id === 2 && f >= 8 && f < 26; } },
    both: { name: "P1 and P3 lost, then swap sides", hide: function (id, f) { return (id === 1 || id === 3) && f >= 8 && f < 26; } }
  };

  function truePos(id, f, scenario) {
    var base = { 1: 130, 2: 420, 3: 700 }[id];
    var drift = { 1: 3.2, 2: -1.5, 3: -4.4 }[id];
    var extra = scenario === "both" && id === 1 ? (f > 8 ? (f - 8) * 12 : 0) : 0;
    var extra3 = scenario === "both" && id === 3 ? (f > 8 ? (f - 8) * -11 : 0) : 0;
    var jump = scenario === "long" && id === 2 && f >= 8 ? (f - 8) * 7.5 : 0;
    return { x: base + drift * f + extra + extra3 + jump, y: 60 + 8 * Math.sin((f + id * 2) / 4), width: SIZE, height: SIZE };
  }

  function permutations(n, m) {
    // All ways to assign up to n trackers to m detections; small enough to brute force.
    var out = [];
    (function rec(i, used, acc) {
      if (i === n) { out.push(acc.slice()); return; }
      acc.push(-1); rec(i + 1, used, acc); acc.pop();
      for (var j = 0; j < m; j++) {
        if (!used[j]) { used[j] = true; acc.push(j); rec(i + 1, used, acc); acc.pop(); used[j] = false; }
      }
    })(0, [], []);
    return out;
  }

  function Tracker(id, box) {
    this.id = id; this.box = box; this.vx = 0; this.vy = 0; this.since = 0; this.pred = box;
  }

  function step(state, iouThreshold, decay) {
    var f = state.frame;
    var dets = [];
    [1, 2, 3].forEach(function (id) {
      if (!SCENARIOS[state.scenario].hide(id, f)) dets.push({ truth: id, box: truePos(id, f, state.scenario) });
    });
    // Two returning people get free IDs in arrival order, which nothing guarantees.
    if (state.scenario === "both" && f >= 26) dets.reverse();

    state.trackers.forEach(function (t) {
      if (t.since > 0) { t.vx *= decay; t.vy *= decay; }
      t.pred = { x: t.box.x + t.vx, y: t.box.y + t.vy, width: t.box.width, height: t.box.height };
      if (t.since > 0) t.box = t.pred;
      t.since += 1;
    });

    // Hungarian equivalent: the assignment with the lowest total cost.
    var best = null, bestCost = Infinity;
    permutations(state.trackers.length, dets.length).forEach(function (a) {
      if (a.filter(function (v) { return v >= 0; }).length !== Math.min(state.trackers.length, dets.length)) return;
      var cost = 0;
      a.forEach(function (j, i) { if (j >= 0) cost += 1 - M.iou(state.trackers[i].pred, dets[j].box); });
      if (cost < bestCost) { bestCost = cost; best = a; }
    });

    var matched = new Set(), results = [], events = [];
    if (best) {
      best.forEach(function (j, i) {
        if (j < 0) return;
        if (M.iou(state.trackers[i].pred, dets[j].box) < iouThreshold) return;
        var t = state.trackers[i], d = dets[j].box;
        t.vx = d.x - t.box.x; t.vy = d.y - t.box.y;
        t.box = d; t.since = 0;
        matched.add(j);
        results.push({ internal: t.id + 1, personID: t.id + 1, det: dets[j] });
      });
    }
    dets.forEach(function (d, j) {
      if (matched.has(j)) return;
      var t = new Tracker(state.next++, d.box);
      state.trackers.push(t);
      results.push({ internal: t.id + 1, personID: t.id + 1, det: d });
      if (f > 0) events.push("No match for a detection, so a new tracker was created with internal ID " + (t.id + 1) + ".");
    });

    // Display IDs: same logic as the expected_num_people_ block.
    var n = 3, active = new Set(), need = [];
    results.forEach(function (r) { if (r.personID <= n) active.add(r.personID); });
    results.forEach(function (r) {
      if (r.personID <= n) return;
      if (state.map.has(r.internal)) {
        var mapped = state.map.get(r.internal);
        if (active.has(mapped)) { state.map.delete(r.internal); need.push(r); }
        else { r.personID = mapped; active.add(mapped); }
      } else need.push(r);
    });
    need.forEach(function (r) {
      for (var i = 1; i <= n; i++) {
        if (!active.has(i)) {
          state.map.set(r.internal, i);
          events.push("Internal ID " + r.internal + " was given free display ID " + i + (r.det.truth === i ? ", the correct person." : ". That is really P" + r.det.truth + ", so the IDs have swapped."));
          r.personID = i;
          active.add(i);
          break;
        }
      }
    });
    state.results = results;
    state.events = events.length ? events : state.events;
    state.frame += 1;
  }

  function build() {
    var root = document.getElementById("demo-sort");
    root.appendChild(el("div", { class: "demo-head" }, [
      el("h4", { text: "Stepping the tracker" }),
      el("p", { text: "Coloured boxes are detections, coloured by the true person. The number is the ID the code assigns. Dashed boxes are lost trackers still predicting." })
    ]));
    var body = el("div", { class: "demo-body" });
    var row = el("div", { class: "btn-row", role: "group", "aria-label": "Scenario" });
    var buttons = {};
    Object.keys(SCENARIOS).forEach(function (k) {
      var b = el("button", { class: "btn", type: "button", text: SCENARIOS[k].name });
      b.addEventListener("click", function () { scenario = k; reset(); });
      row.appendChild(b);
      buttons[k] = b;
    });
    body.appendChild(row);
    var canvas = el("canvas", { role: "img", "aria-label": "Tracked boxes across a strip of panorama", style: "margin-top:11px" });
    body.appendChild(el("div", { class: "canvas-frame" }, [canvas]));
    var ctrl = el("div", { class: "btn-row", style: "margin-top:10px" });
    var stepBtn = el("button", { class: "btn primary", type: "button", text: "Next frame" });
    var playBtn = el("button", { class: "btn", type: "button", text: "Play" });
    var resetBtn = el("button", { class: "btn", type: "button", text: "Restart" });
    var frameLabel = el("span", { class: "note" });
    [stepBtn, playBtn, resetBtn, frameLabel].forEach(function (b) { ctrl.appendChild(b); });
    body.appendChild(ctrl);
    var cols = el("div", { class: "cols", style: "margin-top:10px" });
    var tableBox = el("div", { class: "table-scroll" });
    var side = el("div");
    var iouS = UI.slider(side, { label: "iou_threshold", min: 0.02, max: 0.6, step: 0.01, value: 0.15, format: function (v) { return v.toFixed(2); }, onInput: reset });
    var decayS = UI.slider(side, { label: "velocity_decay", min: 0, max: 1, step: 0.05, value: 0.2, format: function (v) { return v.toFixed(2); }, onInput: reset });
    cols.appendChild(tableBox); cols.appendChild(side);
    body.appendChild(cols);
    var event = el("p", { class: "note", style: "min-height:1.6em" });
    body.appendChild(event);
    root.appendChild(body);
    root.appendChild(el("div", { class: "demo-foot", text: "ID logic follows Sort::update. A simple constant-velocity predictor replaces the real Kalman filter, so motion here is approximate." }));

    var scenario = "blink", state, timer = null;
    function reset() {
      stop();
      state = { frame: 0, scenario: scenario, trackers: [], next: 0, map: new Map(), results: [], events: [] };
      step(state, iouS.get(), decayS.get());
      Object.keys(buttons).forEach(function (k) { buttons[k].setAttribute("aria-pressed", String(k === scenario)); });
      draw();
    }
    function stop() {
      if (timer) { clearInterval(timer); timer = null; playBtn.textContent = "Play"; }
    }
    function advance() {
      if (state.frame >= 40) { stop(); return; }
      step(state, iouS.get(), decayS.get());
      draw();
    }
    function draw() {
      var ctx = UI.hidpi(canvas, W, H);
      ctx.fillStyle = UI.css("--surface"); ctx.fillRect(0, 0, W, H);
      state.trackers.forEach(function (t) {
        if (t.since === 0) return;
        ctx.strokeStyle = UI.css("--ink-3"); ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
        ctx.strokeRect(t.pred.x, t.pred.y, t.pred.width, t.pred.height); ctx.setLineDash([]);
      });
      state.results.forEach(function (r) {
        var b = r.det.box;
        ctx.fillStyle = UI.personColor(r.det.truth);
        ctx.globalAlpha = 0.2; ctx.fillRect(b.x, b.y, b.width, b.height); ctx.globalAlpha = 1;
        ctx.strokeStyle = UI.personColor(r.det.truth); ctx.lineWidth = 3; ctx.strokeRect(b.x, b.y, b.width, b.height);
        ctx.fillStyle = UI.css("--ink"); ctx.font = "700 14px " + UI.css("--mono"); ctx.textAlign = "center";
        ctx.fillText("ID " + r.personID, b.x + b.width / 2, b.y - 7);
        ctx.fillStyle = UI.css("--ink-2"); ctx.font = "11px " + UI.css("--mono");
        ctx.fillText("P" + r.det.truth, b.x + b.width / 2, b.y + b.height / 2 + 4);
      });
      frameLabel.textContent = "Frame " + (state.frame - 1) + " of 39";
      event.textContent = state.events.length ? state.events.join(" ") : "Every detection matched its tracker.";
      tableBox.textContent = "";
      var table = el("table");
      table.appendChild(el("tr", {}, ["Internal ID", "Frames since match", "Shown as"].map(function (t) { return el("th", { text: t }); })));
      state.trackers.forEach(function (t) {
        var internal = t.id + 1;
        var res = state.results.filter(function (r) { return r.internal === internal; })[0];
        var display = res ? "ID " + res.personID : "lost, still predicting";
        table.appendChild(el("tr", {}, [el("td", { class: "num", text: String(internal) }), el("td", { class: "num", text: String(t.since) }), el("td", { text: display })]));
      });
      tableBox.appendChild(table);
    }
    stepBtn.addEventListener("click", function () { stop(); advance(); });
    playBtn.addEventListener("click", function () {
      if (timer) { stop(); return; }
      playBtn.textContent = "Pause";
      timer = setInterval(advance, 260);
    });
    resetBtn.addEventListener("click", reset);
    UI.onTheme(draw);
    reset();
  }

  build();
})();
