/* Panorama, L2CS, head and eye blending, world space, and ROI demos. */
(function () {
  "use strict";
  var el = UI.el, M = window.Model;

  /* ---------- synthetic room, ray traced to a real equirectangular image ---------- */
  // Positions are in centimetres, in the project's world frame (camera at the origin).
  var SCENE = [
    { id: 1, head: [-47, 8, -95], gaze: [-11, 0, 207], rgb: [42, 120, 214] },
    { id: 2, head: [-58, 3, 112], gaze: [176, 0, -117], rgb: [235, 104, 52] },
    { id: 3, head: [118, 12, -5], gaze: [-176, 0, 117], rgb: [27, 175, 122] }
  ];
  var PW = 1440, PH = 720;

  function traceScene() {
    var data = new Uint8ClampedArray(PW * PH * 4);
    var ROOM = 330, FLOOR = -125, CEIL = 175, HEAD_R = 13, BODY_R = 24;
    SCENE.forEach(function (p) { p.face = M.unit(p.gaze); });
    function shade(i, r, g, b, k) {
      data[i] = r * k; data[i + 1] = g * k; data[i + 2] = b * k; data[i + 3] = 255;
    }
    for (var v = 0; v < PH; v++) {
      var lat = (0.5 - (v + 0.5) / PH) * Math.PI, cl = Math.cos(lat), sl = Math.sin(lat);
      for (var u = 0; u < PW; u++) {
        var lon = ((u + 0.5) / PW - 0.5) * 2 * Math.PI;
        var dx = Math.sin(lon) * cl, dy = sl, dz = Math.cos(lon) * cl;
        var i = (v * PW + u) * 4;
        var best = Infinity, hit = null;

        // room
        var tx = (dx > 0 ? ROOM : -ROOM) / (dx || 1e-9), tz = (dz > 0 ? ROOM : -ROOM) / (dz || 1e-9);
        var ty = (dy > 0 ? CEIL : FLOOR) / (dy || 1e-9);
        best = Math.min(tx, tz, ty);
        hit = best === ty ? (dy > 0 ? "ceil" : "floor") : "wall";

        // table around the camera
        if (dy < 0) {
          var tt = -42 / dy, tablex = dx * tt, tablez = dz * tt;
          if (tablex * tablex + tablez * tablez < 72 * 72 && tt < best) { best = tt; hit = "table"; }
        }

        // people: a sphere and a cylinder
        for (var k = 0; k < SCENE.length; k++) {
          var p = SCENE[k], cx = p.head[0], cy = p.head[1], cz = p.head[2];
          var b = dx * cx + dy * cy + dz * cz, c = cx * cx + cy * cy + cz * cz - HEAD_R * HEAD_R;
          var disc = b * b - c;
          if (disc > 0) {
            var t = b - Math.sqrt(disc);
            if (t > 0 && t < best) { best = t; hit = p; }
          }
          var A = dx * dx + dz * dz, B = dx * cx + dz * cz, C = cx * cx + cz * cz - BODY_R * BODY_R;
          var d2 = B * B - A * C;
          if (d2 > 0 && A > 1e-9) {
            var t2 = (B - Math.sqrt(d2)) / A, yy = dy * t2;
            if (t2 > 0 && t2 < best && yy < cy - 17 && yy > FLOOR) { best = t2; hit = { body: p }; }
          }
        }

        if (hit === "wall") {
          var wx = dx * best, wy = dy * best, wz = dz * best;
          var along = Math.abs(Math.abs(wx) - ROOM) < 0.5 ? wz : wx;
          var line = Math.abs(((along % 80) + 80) % 80 - 40) > 38.4 || Math.abs(((wy % 80) + 80) % 80 - 40) > 38.8;
          var window1 = wy > 20 && wy < 125 && Math.abs(((along % 220) + 220) % 220 - 110) < 62;
          if (window1) shade(i, 186, 214, 232, 1);
          else if (line) shade(i, 176, 170, 156, 1);
          else shade(i, 222, 216, 202, 0.96);
        } else if (hit === "floor") {
          var fx = dx * best, fz = dz * best;
          var check = (Math.floor(fx / 60) + Math.floor(fz / 60)) & 1;
          shade(i, 156, 138, 112, check ? 1 : 0.86);
        } else if (hit === "ceil") {
          shade(i, 236, 232, 222, 1);
        } else if (hit === "table") {
          shade(i, 118, 92, 68, 1);
        } else if (hit && hit.body) {
          shade(i, hit.body.rgb[0], hit.body.rgb[1], hit.body.rgb[2], 0.62);
        } else if (hit) {
          var nx = (dx * best - hit.head[0]) / HEAD_R, ny = (dy * best - hit.head[1]) / HEAD_R, nz = (dz * best - hit.head[2]) / HEAD_R;
          var f = nx * hit.face[0] + ny * hit.face[1] + nz * hit.face[2];
          if (f > 0.55) shade(i, 244, 224, 200, 0.75 + 0.25 * f);
          else shade(i, hit.rgb[0], hit.rgb[1], hit.rgb[2], 0.72 + 0.28 * Math.max(0, ny));
        }
      }
    }
    return data;
  }

  function personBox(p, panoW, panoH) {
    var dist = Math.hypot(p.head[0], p.head[2]);
    var lon = Math.atan2(p.head[0], p.head[2]);
    var top = Math.atan2(p.head[1] + 13, dist), bottom = Math.atan2(-125, dist);
    var half = Math.atan2(26, dist);
    var cx = (lon / (2 * Math.PI) + 0.5) * panoW, w = (2 * half / (2 * Math.PI)) * panoW;
    var y0 = (0.5 - top / Math.PI) * panoH, y1 = (0.5 - bottom / Math.PI) * panoH;
    return { x: cx - w / 2, y: y0, width: w, height: y1 - y0 };
  }

  function buildPano() {
    var root = document.getElementById("demo-pano");
    root.appendChild(el("div", { class: "demo-head" }, [
      el("h4", { text: "A virtual camera inside a panorama" }),
      el("p", { text: "A synthetic room, ray traced. Faces are the pale patches. Drag on the panorama to aim. Notice how the straight walls curve there and come out straight on the right." })
    ]));
    var body = el("div", { class: "demo-body" });
    var cols = el("div", { class: "cols" });
    var left = el("div");
    var right = el("div");
    var panoCanvas = el("canvas", { role: "img", "aria-label": "Equirectangular panorama with the camera footprint outlined" });
    var viewCanvas = el("canvas", { role: "img", "aria-label": "Perspective view rendered from the panorama" });
    left.appendChild(el("p", { class: "fig-label", text: "Equirectangular frame (what the video stores)" }));
    left.appendChild(el("div", { class: "canvas-frame" }, [panoCanvas]));
    right.appendChild(el("p", { class: "fig-label", text: "Perspective crop (what the networks see)" }));
    right.appendChild(el("div", { class: "canvas-frame" }, [viewCanvas]));
    cols.appendChild(left); cols.appendChild(right);
    body.appendChild(cols);

    var controls = el("div", { class: "cols", style: "margin-top:10px" });
    var c1 = el("div"), c2 = el("div");
    controls.appendChild(c1); controls.appendChild(c2);
    var row = el("div", { class: "btn-row", role: "group", "aria-label": "Aim at a person" });
    var manual = false, selected = 1;
    var btns = [1, 2, 3].map(function (id) {
      var b = el("button", { class: "btn", type: "button", text: "Aim at P" + id, "aria-pressed": "false" });
      b.addEventListener("click", function () { selected = id; manual = false; aim(); });
      row.appendChild(b);
      return b;
    });
    c1.appendChild(row);
    var yaw = UI.slider(c1, { label: "Yaw", min: -180, max: 180, step: 0.5, value: 0, format: function (v) { return v.toFixed(1) + "°"; }, onInput: function () { manual = true; render(); } });
    var pitch = UI.slider(c1, { label: "Pitch", min: -60, max: 60, step: 0.5, value: 0, format: function (v) { return v.toFixed(1) + "°"; }, onInput: function () { manual = true; render(); } });
    var fov = UI.slider(c1, { label: "FOV", min: 20, max: 110, step: 0.5, value: 60, format: function (v) { return v.toFixed(1) + "°"; }, onInput: function () { manual = true; render(); } });
    var ratio = UI.slider(c2, { label: "head_height_ratio: head size in the crop", min: 0.05, max: 0.4, step: 0.01, value: 0.18, format: function (v) { return v.toFixed(2); }, onInput: function () { manual = false; aim(); } });
    var info = el("div", { class: "stats" });
    c2.appendChild(info);
    var hint = el("p", { class: "note" });
    c2.appendChild(hint);
    body.appendChild(controls);
    root.appendChild(body);
    root.appendChild(el("div", { class: "demo-foot", html: "Uses the project's <code>viewerForBox</code> and remap formulas at 3840×1920 coordinates. The right pane is rendered at 320×240 here, half the real 640×480." }));

    var scene = traceScene();
    var panoCtx = UI.hidpi(panoCanvas, 960, 480);
    var offscreen = document.createElement("canvas");
    offscreen.width = PW; offscreen.height = PH;
    offscreen.getContext("2d").putImageData(new ImageData(scene, PW, PH), 0, 0);
    viewCanvas.width = 320; viewCanvas.height = 240;
    viewCanvas.style.aspectRatio = "4 / 3";
    var viewCtx = viewCanvas.getContext("2d");
    var viewImage = viewCtx.createImageData(320, 240);

    function currentView() { return { yaw: yaw.get(), pitch: pitch.get(), fov: fov.get() }; }

    function aim() {
      var box = personBox(SCENE[selected - 1], 3840, 1920);
      var v = M.viewerForBox(box, 1920, 3840, ratio.get());
      yaw.set(Math.max(-180, Math.min(180, v.yaw)));
      pitch.set(Math.max(-60, Math.min(60, v.pitch)));
      fov.set(Math.max(20, Math.min(110, v.fov)));
      render();
    }

    function drawFootprint(view) {
      var pts = [], steps = 40, i;
      for (i = 0; i <= steps; i++) pts.push([640 * i / steps, 0]);
      for (i = 0; i <= steps; i++) pts.push([640, 480 * i / steps]);
      for (i = 0; i <= steps; i++) pts.push([640 * (1 - i / steps), 480]);
      for (i = 0; i <= steps; i++) pts.push([0, 480 * (1 - i / steps)]);
      panoCtx.lineWidth = 2.4;
      panoCtx.strokeStyle = "#111";
      panoCtx.setLineDash([]);
      for (var pass = 0; pass < 2; pass++) {
        panoCtx.strokeStyle = pass ? "#ffffff" : "#111111";
        panoCtx.lineWidth = pass ? 1.3 : 3.2;
        panoCtx.beginPath();
        var last = null;
        pts.forEach(function (q) {
          var p = M.perspectiveToPano(q[0], q[1], view, 960, 480);
          if (last && Math.abs(p[0] - last[0]) < 480) panoCtx.lineTo(p[0], p[1]);
          else panoCtx.moveTo(p[0], p[1]);
          last = p;
        });
        panoCtx.stroke();
      }
    }

    function render() {
      var view = currentView();
      panoCtx.imageSmoothingQuality = "high";
      panoCtx.drawImage(offscreen, 0, 0, 960, 480);
      if (!manual) {
        SCENE.forEach(function (p) {
          var b = personBox(p, 960, 480);
          panoCtx.strokeStyle = "#111";
          panoCtx.lineWidth = 1;
          panoCtx.setLineDash([4, 3]);
          panoCtx.strokeRect(b.x, b.y, b.width, b.height);
          panoCtx.setLineDash([]);
        });
      }
      drawFootprint(view);

      var out = viewImage.data;
      for (var y = 0; y < 240; y++) {
        for (var x = 0; x < 320; x++) {
          var p = M.perspectiveToPano(x * 2 + 0.5, y * 2 + 0.5, view, PW, PH);
          var px = p[0] - 0.5, py = p[1] - 0.5;
          var x0 = Math.floor(px), y0 = Math.max(0, Math.floor(py));
          var fx = px - x0, fy = Math.max(0, py - y0);
          var x1 = (x0 + 1 + PW) % PW, y1 = Math.min(PH - 1, y0 + 1);
          x0 = (x0 + PW) % PW;
          var a = (y0 * PW + x0) * 4, b = (y0 * PW + x1) * 4, c = (y1 * PW + x0) * 4, d = (y1 * PW + x1) * 4;
          var di = (y * 320 + x) * 4;
          for (var ch = 0; ch < 3; ch++) {
            out[di + ch] = (scene[a + ch] * (1 - fx) + scene[b + ch] * fx) * (1 - fy) + (scene[c + ch] * (1 - fx) + scene[d + ch] * fx) * fy;
          }
          out[di + 3] = 255;
        }
      }
      viewCtx.putImageData(viewImage, 0, 0);

      btns.forEach(function (b, i) { b.setAttribute("aria-pressed", String(!manual && selected === i + 1)); });
      info.textContent = "";
      var center = M.perspectiveToPano(319.5, 239.5, view, 3840, 1920);
      [["Yaw", view.yaw.toFixed(1) + "°"], ["Pitch", view.pitch.toFixed(1) + "°"], ["FOV", view.fov.toFixed(1) + "°"], ["Crop centre", Math.round(center[0]) + ", " + Math.round(center[1])]].forEach(function (s) {
        info.appendChild(el("div", { class: "stat" }, [el("b", { text: s[1] }), el("span", { text: s[0] })]));
      });
      hint.textContent = manual ? "Manual aim. In the app, Sort::inject sets these from the YOLO box." : "Aimed by Sort::inject from the dashed person box.";
    }

    var dragging = false;
    function dragTo(e) {
      var p = UI.canvasPoint(panoCanvas, e, 960, 480);
      manual = true;
      yaw.set(M.clamp((p[0] / 960 - 0.5) * 360, -180, 180));
      pitch.set(M.clamp((0.5 - p[1] / 480) * 180, -60, 60));
      render();
    }
    panoCanvas.addEventListener("pointerdown", function (e) { dragging = true; panoCanvas.setPointerCapture(e.pointerId); dragTo(e); });
    panoCanvas.addEventListener("pointermove", function (e) { if (dragging) dragTo(e); });
    panoCanvas.addEventListener("pointerup", function () { dragging = false; });
    aim();
  }

  /* ---------- L2CS bins ---------- */
  function buildL2cs() {
    var root = document.getElementById("demo-l2cs");
    root.appendChild(el("div", { class: "demo-head" }, [
      el("h4", { text: "From 90 bin scores to one angle" }),
      el("p", { text: "Move the peak and change how sure the network is. Hover the chart to read a bin." })
    ]));
    var body = el("div", { class: "demo-body cols" });
    var left = el("div"), right = el("div");
    var W = 560, H = 190;
    var canvas = el("canvas", { role: "img", "aria-label": "Bar chart of probability per angle bin" });
    left.appendChild(el("div", { class: "canvas-frame" }, [canvas]));
    var state = { peak: 24, spread: 4.5 };
    var last;
    function logits() {
      var out = [];
      for (var i = 0; i < 90; i++) {
        var a = i * 4 - 180;
        // A small second bump shows why the expected value differs from the top bin.
        out.push(Math.log(Math.exp(-Math.pow(a - state.peak, 2) / (2 * state.spread * state.spread)) + 0.12 * Math.exp(-Math.pow(a - state.peak - 40, 2) / 450) + 1e-4));
      }
      return out;
    }
    UI.slider(right, { label: "Where the network's score peaks", min: -60, max: 60, value: 24, format: function (v) { return v + "°"; }, onInput: function (v) { state.peak = v; draw(); } });
    UI.slider(right, { label: "How spread out the scores are", min: 2.5, max: 30, step: 0.5, value: 4.5, format: function (v) { return v.toFixed(1) + "°"; }, onInput: function (v) { state.spread = v; draw(); } });
    var stats = el("div", { class: "stats" });
    right.appendChild(stats);
    var note = el("p", { class: "note" });
    right.appendChild(note);
    body.appendChild(left); body.appendChild(right);
    root.appendChild(body);

    var ctx;
    function draw() {
      ctx = UI.hidpi(canvas, W, H);
      var r = M.decodeL2cs(logits());
      last = r;
      var max = Math.max.apply(null, r.probs);
      var x0 = 12, base = H - 22, plot = W - 24, bw = plot / 90;
      ctx.fillStyle = UI.css("--surface"); ctx.fillRect(0, 0, W, H);
      ctx.strokeStyle = UI.css("--rule-strong"); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x0, base + 0.5); ctx.lineTo(x0 + plot, base + 0.5); ctx.stroke();
      ctx.fillStyle = UI.css("--accent");
      r.probs.forEach(function (p, i) {
        var h = (p / max) * (base - 28);
        ctx.fillRect(x0 + i * bw + 0.5, base - h, Math.max(1, bw - 1), h);
      });
      var markX = x0 + ((r.angle + 180) / 4 + 0.5) * bw;
      ctx.strokeStyle = UI.css("--ink"); ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(markX, 16); ctx.lineTo(markX, base); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = UI.css("--ink"); ctx.font = "600 11px " + UI.css("--mono");
      ctx.textAlign = r.angle > 80 ? "right" : "left";
      ctx.fillText("expected " + r.angle.toFixed(1) + "°", markX + (r.angle > 80 ? -5 : 5), 13);
      ctx.fillStyle = UI.css("--ink-3"); ctx.font = "11px " + UI.css("--mono"); ctx.textAlign = "center";
      [-180, -90, 0, 90, 176].forEach(function (a) { ctx.fillText(a + "°", x0 + ((a + 180) / 4 + 0.5) * bw, H - 6); });

      stats.textContent = "";
      var w = M.eyeWeight(r.confidence);
      [["Decoded angle", r.angle.toFixed(1) + "°"], ["Confidence", (r.confidence * 100).toFixed(1) + "%"], ["Eye weight", w.toFixed(2)]].forEach(function (s) {
        stats.appendChild(el("div", { class: "stat" }, [el("b", { text: s[1] }), el("span", { text: s[0] })]));
      });
      note.textContent = w === 0 ? "Below 20% confidence the eyes are ignored completely." : w === 1 ? "Above 40% confidence the eye correction is used in full." : "Between 20% and 40% the eye correction is partly applied.";
    }
    canvas.addEventListener("pointermove", function (e) {
      var p = UI.canvasPoint(canvas, e, W, H);
      var i = Math.floor((p[0] - 12) / ((W - 24) / 90));
      if (i < 0 || i > 89 || !last) { UI.hideTip(); return; }
      UI.showTip(e, [el("b", { text: (last.probs[i] * 100).toFixed(1) + "%" }), document.createTextNode("  bin " + i + ", " + (i * 4 - 180) + "°")]);
    });
    canvas.addEventListener("pointerleave", UI.hideTip);
    UI.onTheme(draw);
    draw();
  }

  /* ---------- shared drawing ---------- */
  function arrow(ctx, x0, y0, x1, y1, color, width, dashed) {
    var a = Math.atan2(y1 - y0, x1 - x0), head = 7 + width;
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
    ctx.setLineDash(dashed ? [5, 4] : []);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - Math.cos(a) * head * 0.7, y1 - Math.sin(a) * head * 0.7); ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - head * Math.cos(a - 0.42), y1 - head * Math.sin(a - 0.42));
    ctx.lineTo(x1 - head * Math.cos(a + 0.42), y1 - head * Math.sin(a + 0.42));
    ctx.closePath(); ctx.fill();
  }
  function label(ctx, text, x, y, align, color) {
    ctx.font = "600 12px " + UI.css("--sans");
    ctx.textAlign = align || "left";
    ctx.fillStyle = color || UI.css("--ink");
    ctx.fillText(text, x, y);
  }
  function cameraIcon(ctx, x, y) {
    ctx.fillStyle = UI.css("--ink");
    ctx.beginPath(); ctx.arc(x, y, 9, 0, 7); ctx.fill();
    ctx.fillStyle = UI.css("--surface");
    ctx.beginPath(); ctx.arc(x, y, 3.5, 0, 7); ctx.fill();
  }
  window.DemoDraw = { arrow: arrow, label: label, cameraIcon: cameraIcon };

  /* ---------- head and eye blend ---------- */
  function buildBlend() {
    var root = document.getElementById("demo-blend");
    root.appendChild(el("div", { class: "demo-head" }, [
      el("h4", { text: "Blending head and eyes" }),
      el("p", { text: "View from above, with a person facing the camera. Yaw 0 points at the camera. Pitch is held at 0 here." })
    ]));
    var body = el("div", { class: "demo-body cols" });
    var W = 460, H = 330;
    var canvas = el("canvas", { role: "img", "aria-label": "Top view of head, eye and final gaze directions" });
    var left = el("div", {}, [el("div", { class: "canvas-frame" }, [canvas])]);
    var right = el("div");
    var fmt = function (v) { return v + "°"; };
    var head = UI.slider(right, { label: "Head yaw (pose network)", min: -70, max: 70, value: 15, format: fmt, onInput: draw });
    var gaze = UI.slider(right, { label: "Gaze yaw (L2CS)", min: -80, max: 80, value: 60, format: fmt, onInput: draw });
    var conf = UI.slider(right, { label: "L2CS confidence", min: 0, max: 60, value: 32, format: function (v) { return v + "%"; }, onInput: draw });
    var stats = el("div", { class: "stats" });
    right.appendChild(stats);
    body.appendChild(left); body.appendChild(right);
    root.appendChild(body);
    root.appendChild(el("div", { class: "demo-foot", text: "Dashed: raw inputs. Solid: the output. At 20% confidence the output falls back to the head direction. At 40% and above it can pass the L2CS value, because only 40% of the head yaw is removed before the two are added." }));

    function draw() {
      var ctx = UI.hidpi(canvas, W, H);
      ctx.fillStyle = UI.css("--surface"); ctx.fillRect(0, 0, W, H);
      var cx = W / 2, cy = 115, len = 95;
      var toScreen = function (yawDeg) {
        var d = M.globalGazeFromPanorama(yawDeg * M.DEG, 0, [0, 0, 1]);
        return [cx + d[0] * len, cy - d[2] * len];
      };
      cameraIcon(ctx, cx, H - 28);
      label(ctx, "camera", cx + 15, H - 24, "left", UI.css("--ink-3"));
      ctx.strokeStyle = UI.css("--rule-strong"); ctx.setLineDash([2, 4]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx, H - 40); ctx.stroke(); ctx.setLineDash([]);

      var r = M.combineGaze(head.get(), 0, gaze.get(), 0, conf.get() / 100);
      var h = toScreen(head.get()), g = toScreen(gaze.get()), o = toScreen(r.yaw);
      arrow(ctx, cx, cy, h[0], h[1], UI.css("--none"), 2, true);
      arrow(ctx, cx, cy, g[0], g[1], UI.css("--p2"), 2, true);
      arrow(ctx, cx, cy, o[0], o[1], UI.css("--accent"), 3.2, false);
      ctx.fillStyle = UI.css("--surface-2"); ctx.strokeStyle = UI.css("--ink-2"); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(cx, cy, 20, 0, 7); ctx.fill(); ctx.stroke();

      var side = function (p) { return p[0] < cx ? "right" : "left"; };
      var pad = function (p) { return p[0] < cx ? -7 : 7; };
      label(ctx, "head", h[0] + pad(h), h[1] + 4, side(h), UI.css("--ink-2"));
      label(ctx, "L2CS", g[0] + pad(g), g[1] - 8, side(g), UI.css("--ink-2"));
      label(ctx, "output", o[0] + pad(o), o[1] + 16, side(o));

      stats.textContent = "";
      [["Eye weight", r.eyeWeight.toFixed(2)], ["Eye yaw after −0.4 × head", r.compYaw.toFixed(1) + "°"], ["Combined yaw", r.yaw.toFixed(2) + "°"], ["Stored in Pose (int)", r.yawInt + "°"]].forEach(function (s) {
        stats.appendChild(el("div", { class: "stat" }, [el("b", { text: s[1] }), el("span", { text: s[0] })]));
      });
    }
    UI.onTheme(draw);
    draw();
  }

  /* ---------- crop to world ---------- */
  function buildWorld() {
    var root = document.getElementById("demo-world");
    root.appendChild(el("div", { class: "demo-head" }, [
      el("h4", { text: "Why yaw 0 means looking at the camera" }),
      el("p", { text: "View from above: +Z is up the screen, +X to the right. Drag the person around the camera." })
    ]));
    var body = el("div", { class: "demo-body cols" });
    var W = 460, H = 360, SCALE = 0.75;
    var canvas = el("canvas", { role: "img", "aria-label": "Top view of the camera, one person and their gaze ray in the world" });
    var left = el("div", {}, [el("div", { class: "canvas-frame" }, [canvas])]);
    var right = el("div");
    var state = { camYaw: 40 };
    var yawS = UI.slider(right, { label: "Pose yaw (from detect)", min: -90, max: 90, value: 0, format: function (v) { return v + "°"; }, onInput: draw });
    var camS = UI.slider(right, { label: "Camera yaw for this person", min: -180, max: 180, value: 40, format: function (v) { return v + "°"; }, onInput: function (v) { state.camYaw = v; draw(); } });
    var distS = UI.slider(right, { label: "Distance from camera", min: 60, max: 200, value: 150, format: function (v) { return v + " cm"; }, onInput: draw });
    var stats = el("div", { class: "stats" });
    right.appendChild(stats);
    body.appendChild(left); body.appendChild(right);
    root.appendChild(body);
    root.appendChild(el("div", { class: "demo-foot", text: "The same pose yaw gives a different world direction for each person, because it is measured relative to the line back to the camera." }));

    function draw() {
      var ctx = UI.hidpi(canvas, W, H);
      ctx.fillStyle = UI.css("--surface"); ctx.fillRect(0, 0, W, H);
      var cx = W / 2, cy = H / 2;
      var r = M.addGaze(state.camYaw, 0, yawS.get(), 0, [0, 0, distS.get()]);
      var px = cx + r.start[0] * SCALE, py = cy - r.start[2] * SCALE;
      ctx.strokeStyle = UI.css("--rule"); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(W, cy); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
      label(ctx, "+Z (yaw 0)", cx + 6, 15, "left", UI.css("--ink-3"));
      label(ctx, "+X", W - 8, cy - 7, "right", UI.css("--ink-3"));
      ctx.strokeStyle = UI.css("--rule-strong"); ctx.setLineDash([3, 4]);
      ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(cx, cy); ctx.stroke(); ctx.setLineDash([]);
      arrow(ctx, px, py, px + r.direction[0] * 100, py - r.direction[2] * 100, UI.css("--accent"), 3, false);
      cameraIcon(ctx, cx, cy);
      ctx.fillStyle = UI.css("--p1");
      ctx.strokeStyle = UI.css("--surface"); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(px, py, 13, 0, 7); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#fff"; ctx.font = "700 11px " + UI.css("--mono"); ctx.textAlign = "center";
      ctx.fillText("P1", px, py + 4);
      stats.textContent = "";
      [["start (x, z) cm", r.start[0].toFixed(0) + ", " + r.start[2].toFixed(0)], ["direction (x, z)", r.direction[0].toFixed(2) + ", " + r.direction[2].toFixed(2)]].forEach(function (s) {
        stats.appendChild(el("div", { class: "stat" }, [el("b", { text: s[1] }), el("span", { text: s[0] })]));
      });
    }
    var dragging = false;
    function drag(e) {
      var p = UI.canvasPoint(canvas, e, W, H);
      state.camYaw = Math.round(Math.atan2(p[0] - W / 2, H / 2 - p[1]) / M.DEG);
      camS.set(state.camYaw);
      distS.set(M.clamp(Math.round(Math.hypot(p[0] - W / 2, p[1] - H / 2) / SCALE), 60, 200));
      draw();
    }
    canvas.addEventListener("pointerdown", function (e) { dragging = true; canvas.setPointerCapture(e.pointerId); drag(e); });
    canvas.addEventListener("pointermove", function (e) { if (dragging) drag(e); });
    canvas.addEventListener("pointerup", function () { dragging = false; });
    UI.onTheme(draw);
    draw();
  }

  /* ---------- ROI state machine ---------- */
  function buildRoi() {
    var root = document.getElementById("demo-roi");
    root.appendChild(el("div", { class: "demo-head" }, [
      el("h4", { text: "Per-person ROI versus one shared ROI" }),
      el("p", { text: "Each panel is one person's crop. The coloured box is the true face and the black box is the ROI handed to the pose network. Faces are static here and positions are illustrative." })
    ]));
    var FACES = [
      { x: 240, y: 110, width: 110, height: 120 },
      { x: 345, y: 200, width: 80, height: 88 },
      { x: 150, y: 80, width: 140, height: 150 }
    ];
    var body = el("div", { class: "demo-body" });
    var modeRow = el("div", { class: "btn-row", role: "group", "aria-label": "ROI mode" });
    var perBtn = el("button", { class: "btn", type: "button", text: "Per person (current code)" });
    var sharedBtn = el("button", { class: "btn", type: "button", text: "Shared (the old bug)" });
    var stepBtn = el("button", { class: "btn primary", type: "button", text: "Process next person" });
    var runBtn = el("button", { class: "btn", type: "button", text: "Run 30 frames" });
    var resetBtn = el("button", { class: "btn", type: "button", text: "Reset" });
    [perBtn, sharedBtn, stepBtn, runBtn, resetBtn].forEach(function (b) { modeRow.appendChild(b); });
    body.appendChild(modeRow);
    var grid = el("div", { class: "cols three", style: "margin-top:12px" });
    var canvases = FACES.map(function (f, i) {
      var c = el("canvas", { role: "img", "aria-label": "Crop for person " + (i + 1) });
      grid.appendChild(el("div", {}, [el("p", { class: "fig-label", text: "P" + (i + 1) + " crop" }), el("div", { class: "canvas-frame" }, [c])]));
      return c;
    });
    body.appendChild(grid);
    var stats = el("div", { class: "stats", style: "margin-top:12px" });
    body.appendChild(stats);
    var log = el("p", { class: "note", style: "min-height:2.8em" });
    body.appendChild(log);
    root.appendChild(body);
    root.appendChild(el("div", { class: "demo-foot", text: "Run 30 frames in both modes. The shared version runs the localizer less often, so it can look like a speed-up, while the ROI overlap collapses. Overlap here is a stand-in for how well framed the patch is." }));

    var shared, states, person, frame, overlaps, runs, shown;
    function reset() {
      states = [{}, {}, {}];
      person = 0; frame = 0; overlaps = [[], [], []]; runs = 0; shown = [null, null, null];
      log.textContent = "Press Process next person to run detect() once.";
      draw();
    }
    function ewa(last, cur, a) {
      var lc = [last.x + last.width / 2, last.y + last.height / 2], cc = [cur.x + cur.width / 2, cur.y + cur.height / 2];
      var w = last.width + a * (cur.width - last.width), h = last.height + a * (cur.height - last.height);
      var nx = lc[0] + a * (cc[0] - lc[0]), ny = lc[1] + a * (cc[1] - lc[1]);
      return { x: nx - w / 2, y: ny - h / 2, width: w, height: h };
    }
    function expand(b, s) {
      var cx = b.x + b.width / 2, cy = b.y + b.height / 2, w = b.width * s, h = b.height * s;
      return { x: cx - w / 2, y: cy - h / 2, width: w, height: h };
    }
    function detect(p) {
      var st = shared ? states[0] : states[p];
      var face = FACES[p], ranLocalizer = false;
      if (!st.loc || !st.roi || M.iou(st.loc, st.roi) < 0.25) {
        ranLocalizer = true; runs++;
        var rect = expand(face, 1.25);
        if (st.roi && M.iou(rect, st.roi) >= 0.25) st.loc = rect;
        else { st.loc = rect; st.roi = rect; }
      }
      var used = st.roi, overlap = M.iou(used, face);
      overlaps[p].push(overlap);
      shown[p] = used;
      st.roi = ewa(used, expand(face, 1.1), 0.5);
      return { ran: ranLocalizer, overlap: overlap };
    }
    function step() {
      var r = detect(person);
      var name = "P" + (person + 1);
      log.textContent = name + ": " + (r.ran ? "the localizer ran. " : "the localizer was skipped, reusing the stored ROI. ") +
        "The ROI overlapped the real face by " + (r.overlap * 100).toFixed(0) + "%. " +
        (shared && !r.ran ? "That ROI came from the previous person's crop, so the pose network got a badly framed patch." : r.overlap > 0.6 ? "A well framed patch." : "");
      person = (person + 1) % 3;
      if (person === 0) frame++;
    }
    function draw() {
      perBtn.setAttribute("aria-pressed", String(!shared));
      sharedBtn.setAttribute("aria-pressed", String(!!shared));
      canvases.forEach(function (c, i) {
        var W = 320, H = 240, ctx = UI.hidpi(c, W, H);
        ctx.fillStyle = UI.css("--surface-2"); ctx.fillRect(0, 0, W, H);
        ctx.save(); ctx.scale(0.5, 0.5);
        var f = FACES[i];
        ctx.fillStyle = UI.personColor(i + 1);
        ctx.globalAlpha = 0.22; ctx.fillRect(f.x, f.y, f.width, f.height); ctx.globalAlpha = 1;
        ctx.strokeStyle = UI.personColor(i + 1); ctx.lineWidth = 4; ctx.strokeRect(f.x, f.y, f.width, f.height);
        if (shown[i]) {
          ctx.strokeStyle = UI.css("--ink"); ctx.lineWidth = 3; ctx.setLineDash([12, 7]);
          ctx.strokeRect(shown[i].x, shown[i].y, shown[i].width, shown[i].height); ctx.setLineDash([]);
        }
        ctx.restore();
        if (i === person) {
          ctx.fillStyle = UI.css("--ink-3"); ctx.font = "600 11px " + UI.css("--mono"); ctx.textAlign = "left";
          ctx.fillText("next", 7, 15);
        }
      });
      var all = overlaps[0].concat(overlaps[1], overlaps[2]);
      var mean = all.length ? all.reduce(function (a, b) { return a + b; }, 0) / all.length : null;
      var recent = all.length ? overlaps.map(function (o) { return o.length ? o[o.length - 1] : 0; }) : null;
      stats.textContent = "";
      [["Frames done", String(frame)], ["Localizer runs", String(runs)], ["Mean ROI overlap", mean === null ? "n/a" : (mean * 100).toFixed(0) + "%"],
        ["Latest overlap, P1 / P2 / P3", recent ? recent.map(function (v) { return (v * 100).toFixed(0); }).join(" / ") : "n/a"]].forEach(function (s) {
        stats.appendChild(el("div", { class: "stat" }, [el("b", { text: s[1] }), el("span", { text: s[0] })]));
      });
    }
    perBtn.addEventListener("click", function () { shared = false; reset(); });
    sharedBtn.addEventListener("click", function () { shared = true; reset(); });
    stepBtn.addEventListener("click", function () { step(); draw(); });
    runBtn.addEventListener("click", function () { for (var i = 0; i < 90; i++) step(); draw(); });
    resetBtn.addEventListener("click", reset);
    UI.onTheme(draw);
    shared = false;
    reset();
  }

  buildPano();
  buildL2cs();
  buildBlend();
  buildWorld();
  buildRoi();
})();
