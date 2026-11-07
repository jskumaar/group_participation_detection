/* The Bayesian gaze mapper. All numbers come from Model.inferGaze, a verified port. */
(function () {
  "use strict";
  var el = UI.el, M = window.Model, D = window.DemoDraw;
  var W = 520, H = 460, SCALE = 1.25;

  function build() {
    var root = document.getElementById("demo-bayes");
    root.appendChild(el("div", { class: "demo-head" }, [
      el("h4", { text: "Bayesian gaze mapper" }),
      el("p", { text: "View from above with the camera at the centre. Drag a person to move them, and drag the round handle to aim the observer. Click a person to make them the observer. The shaded fan is one and two standard deviations of the likelihood." })
    ]));
    var body = el("div", { class: "demo-body" });
    var cols = el("div", { class: "cols" });
    var canvas = el("canvas", { role: "img", "aria-label": "Top view of people and the selected observer's gaze" });
    var left = el("div", {}, [el("div", { class: "canvas-frame" }, [canvas])]);
    var right = el("div");
    cols.appendChild(left); cols.appendChild(right);
    body.appendChild(cols);

    var observerRow = el("div", { class: "btn-row", role: "group", "aria-label": "Observer" });
    observerRow.appendChild(el("span", { class: "note", text: "Observer" }));
    var obsBtns = [1, 2, 3].map(function (id) {
      var b = el("button", { class: "btn", type: "button", text: "P" + id });
      b.addEventListener("click", function () { state.observer = id; draw(); });
      observerRow.appendChild(b);
      return b;
    });
    right.appendChild(observerRow);

    var charts = el("div", { style: "margin-top:9px" });
    right.appendChild(charts);
    var verdict = el("div", { class: "callout", style: "margin:.7rem 0 0" });
    right.appendChild(verdict);

    var controls = el("div", { class: "cols three", style: "margin-top:12px" });
    var c1 = el("div"), c2 = el("div"), c3 = el("div");
    controls.appendChild(c1); controls.appendChild(c2); controls.appendChild(c3);
    body.appendChild(controls);

    var state = {
      observer: 1,
      people: {
        1: { pos: [-47, -95], yaw: 2, pitch: -6, motion: 0 },
        2: { pos: [-58, 112], yaw: 112, pitch: -6, motion: 0 },
        3: { pos: [118, -5], yaw: -55, pitch: -6, motion: 0 }
      },
      cfg: Object.assign({}, M.CONFIG_INI)
    };

    function makeCfgSlider(parent, key, label, min, max, step) {
      return UI.slider(parent, {
        label: label, min: min, max: max, step: step, value: state.cfg[key],
        format: function (v) { return Number(v).toFixed(step < 1 ? 2 : 0); },
        onInput: function (v) { state.cfg[key] = v; markPreset(); draw(); }
      });
    }
    var presetRow = el("div", { class: "btn-row" });
    var iniBtn = el("button", { class: "btn", type: "button", text: "tracker_settings.ini" });
    var defBtn = el("button", { class: "btn", type: "button", text: "Code defaults" });
    presetRow.appendChild(iniBtn); presetRow.appendChild(defBtn);
    c1.appendChild(el("p", { class: "fig-label", text: "Likelihood" }));
    var sliders = {
      sigma_x_deg: makeCfgSlider(c1, "sigma_x_deg", "sigma_x (degrees)", 3, 45, 1),
      sigma_y_deg: makeCfgSlider(c1, "sigma_y_deg", "sigma_y (degrees)", 3, 30, 1)
    };
    var pitchS = UI.slider(c1, { label: "Observer pitch", min: -45, max: 45, value: -6, format: function (v) { return v + "°"; }, onInput: function (v) { state.people[state.observer].pitch = v; draw(); } });
    c2.appendChild(el("p", { class: "fig-label", text: "Prior" }));
    sliders.omega_base = makeCfgSlider(c2, "omega_base", "omega_base", -2, 3, 0.1);
    sliders.omega_motion = makeCfgSlider(c2, "omega_motion", "omega_motion", 0, 5, 0.1);
    sliders.omega_null = makeCfgSlider(c2, "omega_null", "omega_null", -2, 3, 0.1);
    sliders.likelihood_null_baseline = makeCfgSlider(c2, "likelihood_null_baseline", "likelihood_null_baseline", 0.01, 0.6, 0.01);
    c3.appendChild(el("p", { class: "fig-label", text: "Motion this frame" }));
    [1, 2, 3].forEach(function (id) {
      UI.slider(c3, { label: "P" + id + " motion", min: 0, max: 1, step: 0.01, value: 0, format: function (v) { return v.toFixed(2); }, onInput: function (v) { state.people[id].motion = v; draw(); } });
    });
    c3.appendChild(presetRow);

    var edgeBox = el("div", { class: "table-scroll", style: "margin-top:12px" });
    body.appendChild(edgeBox);
    root.appendChild(body);
    root.appendChild(el("div", { class: "demo-foot", text: "The gaze ray is drawn top-down, so pitch only shows up in the numbers. All head heights are set to the same value here. Motion is normally measured from head boxes, and is a slider here so that you can explore it." }));

    function setPreset(cfg) {
      state.cfg = Object.assign({}, cfg);
      Object.keys(sliders).forEach(function (k) { sliders[k].set(cfg[k]); });
      markPreset();
      draw();
    }
    function markPreset() {
      var same = function (c) { return Object.keys(c).every(function (k) { return Math.abs(c[k] - state.cfg[k]) < 1e-6; }); };
      iniBtn.setAttribute("aria-pressed", String(same(M.CONFIG_INI)));
      defBtn.setAttribute("aria-pressed", String(same(M.CONFIG_DEFAULT)));
    }
    iniBtn.addEventListener("click", function () { setPreset(M.CONFIG_INI); });
    defBtn.addEventListener("click", function () { setPreset(M.CONFIG_DEFAULT); });

    function gazes() {
      return [1, 2, 3].map(function (id) {
        var p = state.people[id];
        return { id: id, start: [p.pos[0], -25, p.pos[1]], direction: M.dirFromYawPitch(p.yaw * M.DEG, p.pitch * M.DEG), box: { x: 0, y: 0, width: 0, height: 0 }, motion: p.motion };
      });
    }

    var sx = function (x) { return W / 2 + x * SCALE; };
    var sy = function (z) { return H / 2 - z * SCALE; };
    function fan(ctx, p, yawDeg, sigma, n, color) {
      var r = 2000;
      ctx.fillStyle = color;
      ctx.globalAlpha = n === 1 ? 0.2 : 0.12;
      ctx.beginPath();
      ctx.moveTo(sx(p[0]), sy(p[1]));
      for (var a = -n * sigma; a <= n * sigma + 0.01; a += sigma / 6) {
        var t = (yawDeg + a) * M.DEG;
        ctx.lineTo(sx(p[0] + Math.sin(t) * r), sy(p[1] + Math.cos(t) * r));
      }
      ctx.closePath(); ctx.fill();
      ctx.globalAlpha = 1;
    }
    function handlePos(p) {
      var t = p.yaw * M.DEG, r = 62;
      return [sx(p.pos[0]) + Math.sin(t) * r * 1.25, sy(p.pos[1]) - Math.cos(t) * r * 1.25];
    }

    function draw() {
      var result = M.inferGaze(gazes(), state.cfg);
      var obs = result.observers.filter(function (o) { return o.id === state.observer; })[0];
      var ctx = UI.hidpi(canvas, W, H);
      ctx.fillStyle = UI.css("--surface"); ctx.fillRect(0, 0, W, H);

      var active = state.people[state.observer];
      fan(ctx, active.pos, active.yaw, state.cfg.sigma_x_deg, 2, UI.personColor(state.observer));
      fan(ctx, active.pos, active.yaw, state.cfg.sigma_x_deg, 1, UI.personColor(state.observer));

      D.cameraIcon(ctx, W / 2, H / 2);

      [1, 2, 3].forEach(function (id) {
        var p = state.people[id], x = sx(p.pos[0]), y = sy(p.pos[1]);
        var h = handlePos(p);
        if (id === state.observer) {
          D.arrow(ctx, x, y, h[0], h[1], UI.css("--ink"), 2.6, false);
          ctx.fillStyle = UI.css("--surface"); ctx.strokeStyle = UI.css("--ink"); ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(h[0], h[1], 8, 0, 7); ctx.fill(); ctx.stroke();
        }
        if (p.motion > 0) {
          ctx.strokeStyle = UI.personColor(id); ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(x, y, 19 + p.motion * 16, 0, 7); ctx.stroke();
        }
        ctx.fillStyle = UI.personColor(id);
        ctx.strokeStyle = UI.css("--surface"); ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.arc(x, y, 15, 0, 7); ctx.fill(); ctx.stroke();
        ctx.fillStyle = "#fff"; ctx.font = "700 12px " + UI.css("--mono"); ctx.textAlign = "center";
        ctx.fillText("P" + id, x, y + 4);
      });

      obsBtns.forEach(function (b, i) { b.setAttribute("aria-pressed", String(state.observer === i + 1)); });
      pitchS.set(active.pitch);

      charts.textContent = "";
      var sorted = obs.targets.slice().sort(function (a, b) { return (a.id === -1) - (b.id === -1) || a.id - b.id; });
      function section(titleText, key, max, fmt) {
        charts.appendChild(el("p", { class: "fig-label", text: titleText, style: "margin:8px 0 3px" }));
        var box = el("div");
        UI.bars(box, sorted.map(function (t) {
          return {
            label: t.id === -1 ? "no one" : "P" + t.id, value: t[key], cls: UI.personClass(t.id),
            best: key === "posterior" && t.id === obs.best, text: fmt ? fmt(t) : undefined
          };
        }), max);
        charts.appendChild(box);
      }
      section("Prior: who is worth looking at", "prior", 1);
      section("Likelihood: does the ray point there", "likelihood", 1);
      section("Posterior: the answer", "posterior", 1);

      var bestT = obs.targets.filter(function (t) { return t.id === obs.best; })[0];
      verdict.textContent = "";
      verdict.appendChild(el("p", { html: "<strong>" + "P" + state.observer + " is looking at " + UI.personName(obs.best) + "</strong>, with probability " + bestT.posterior.toFixed(3) + "." }));

      edgeBox.textContent = "";
      var table = el("table");
      table.appendChild(el("tr", {}, ["from", "to", "angle_deg", "is_looking", "probability", "horizontal offset", "vertical offset"].map(function (t, i) {
        return el("th", { text: t, class: i > 1 ? "num" : "" });
      })));
      result.interactions.forEach(function (e) {
        var o = result.observers.filter(function (x) { return x.id === e.from; })[0];
        var t = o.targets.filter(function (x) { return x.id === e.to; })[0];
        table.appendChild(el("tr", { style: e.from === state.observer ? "font-weight:600" : "" }, [
          el("td", { text: String(e.from) }), el("td", { text: String(e.to) }),
          el("td", { class: "num", text: e.angle.toFixed(1) }), el("td", { class: "num", text: String(e.isLooking) }),
          el("td", { class: "num", text: e.probability.toFixed(3) }),
          el("td", { class: "num", text: t.behind ? "behind" : t.thetaX.toFixed(1) + "°" }),
          el("td", { class: "num", text: t.behind ? "behind" : t.thetaY.toFixed(1) + "°" })
        ]));
      });
      edgeBox.appendChild(el("p", { class: "fig-label", text: "What this frame sends on the wire (the last two columns are for reference)" }));
      edgeBox.appendChild(table);
    }

    var drag = null;
    function pick(p) {
      var obs = state.people[state.observer], h = handlePos(obs);
      if (Math.hypot(p[0] - h[0], p[1] - h[1]) < 16) return { type: "aim" };
      for (var id = 1; id <= 3; id++) {
        var q = state.people[id];
        if (Math.hypot(p[0] - sx(q.pos[0]), p[1] - sy(q.pos[1])) < 20) return { type: "move", id: id };
      }
      return null;
    }
    canvas.addEventListener("pointerdown", function (e) {
      var p = UI.canvasPoint(canvas, e, W, H);
      drag = pick(p);
      if (!drag) return;
      drag.start = p;
      drag.moved = false;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", function (e) {
      var p = UI.canvasPoint(canvas, e, W, H);
      if (!drag) { canvas.style.cursor = pick(p) ? "grab" : "default"; return; }
      if (Math.hypot(p[0] - drag.start[0], p[1] - drag.start[1]) > 4) drag.moved = true;
      if (!drag.moved) return;
      if (drag.type === "aim") {
        var o = state.people[state.observer];
        o.yaw = Math.atan2(p[0] - sx(o.pos[0]), sy(o.pos[1]) - p[1]) / M.DEG;
      } else {
        state.people[drag.id].pos = [M.clamp((p[0] - W / 2) / SCALE, -190, 190), M.clamp((H / 2 - p[1]) / SCALE, -170, 170)];
      }
      draw();
    });
    canvas.addEventListener("pointerup", function () {
      if (drag && drag.type === "move" && !drag.moved) { state.observer = drag.id; draw(); }
      drag = null;
    });
    canvas.addEventListener("pointercancel", function () { drag = null; });
    UI.onTheme(draw);
    markPreset();
    draw();
  }

  build();
})();
