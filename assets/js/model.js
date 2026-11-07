/* Pure JavaScript ports of the maths in src/c_impl and src/python_client.
 * Every demo on this site calls these functions, so what you see is what the
 * code does. tools/ holds the check that compares them with the real C++. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.Model = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var PI = 3.14159265359;
  var DEG = PI / 180;

  /* --- vectors --------------------------------------------------------- */
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function cross(a, b) {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  }
  function norm(a) { return Math.sqrt(dot(a, a)); }
  function unit(a) {
    var n = norm(a);
    return n > 1e-6 ? [a[0] / n, a[1] / n, a[2] / n] : [0, 0, 0];
  }
  function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  /* --- 360_image_process.cpp ------------------------------------------- */
  function dirFromYawPitch(yaw, pitch) {
    var cp = Math.cos(pitch), sp = Math.sin(pitch), cy = Math.cos(yaw), sy = Math.sin(yaw);
    return unit([sy * cp, sp, cy * cp]);
  }

  // Yaw 0 and pitch 0 mean "looking straight back at the camera".
  function globalGazeFromPanorama(yaw, pitch, camToPerson) {
    var worldUp = [0, 1, 0];
    var forward = unit([-camToPerson[0], -camToPerson[1], -camToPerson[2]]);
    if (Math.abs(dot(forward, worldUp)) > 0.999) {
      worldUp = [0, 0, 1];
      if (Math.abs(dot(forward, worldUp)) > 0.999) worldUp = [1, 0, 0];
    }
    var right = unit(cross(worldUp, forward));
    var up = unit(cross(forward, right));
    var l = dirFromYawPitch(yaw, pitch);
    return unit([
      right[0] * l[0] + up[0] * l[1] + forward[0] * l[2],
      right[1] * l[0] + up[1] * l[1] + forward[1] * l[2],
      right[2] * l[0] + up[2] * l[1] + forward[2] * l[2]
    ]);
  }

  function addGaze(camYawDeg, camPitchDeg, poseYawDeg, posePitchDeg, position) {
    var cy = Math.cos(camYawDeg * DEG), sy = Math.sin(camYawDeg * DEG);
    var cp = Math.cos(camPitchDeg * DEG), sp = Math.sin(camPitchDeg * DEG);
    var x = position[0], y = position[1], z = position[2];
    var v1y = cp * y - sp * z, v1z = sp * y + cp * z;
    var start = [cy * x + sy * v1z, v1y, -sy * x + cy * v1z];
    var camDir = dirFromYawPitch(camYawDeg * DEG, camPitchDeg * DEG);
    return { start: start, direction: globalGazeFromPanorama(poseYawDeg * DEG, posePitchDeg * DEG, camDir) };
  }

  // One perspective pixel to one panorama pixel (buildRemapTables and
  // convertPerspectiveRectToEquirectangular share this mapping).
  function perspectiveToPano(px, py, view, panoW, panoH) {
    var W = view.outW || 640, H = view.outH || 480;
    var fovFactor = 1 / Math.tan(view.fov * DEG * 0.5);
    var cy = Math.cos(view.yaw * DEG), sy = Math.sin(view.yaw * DEG);
    var cp = Math.cos(view.pitch * DEG), sp = Math.sin(view.pitch * DEG);
    var screenY = 1 - 2 * (py + 0.5) / H;
    var screenX = (2 * (px + 0.5) / W - 1) * (W / H);
    var vx = screenX, vy = screenY, vz = fovFactor;
    var inv = 1 / Math.sqrt(vx * vx + vy * vy + vz * vz);
    vx *= inv; vy *= inv; vz *= inv;
    var y1 = cp * vy + sp * vz, z1 = -sp * vy + cp * vz, x1 = vx;
    var x2 = cy * x1 + sy * z1, z2 = -sy * x1 + cy * z1, y2 = y1;
    var lon = Math.atan2(x2, z2);
    var lat = Math.asin(clamp(y2, -1, 1));
    var u = (lon / (2 * PI) + 0.5) * panoW;
    var v = (0.5 - lat / PI) * panoH;
    u = u % panoW;
    if (u < 0) u += panoW;
    v = clamp(v, 0, panoH - 1);
    return [u, v];
  }

  function perspectiveRectToPano(rect, view, panoW, panoH) {
    var corners = [
      [rect.x, rect.y], [rect.x + rect.width, rect.y],
      [rect.x + rect.width, rect.y + rect.height], [rect.x, rect.y + rect.height]
    ];
    var minU = 1e9, maxU = -1e9, minV = 1e9, maxV = -1e9;
    corners.forEach(function (c) {
      var p = perspectiveToPano(c[0], c[1], view, panoW, panoH);
      minU = Math.min(minU, p[0]); maxU = Math.max(maxU, p[0]);
      minV = Math.min(minV, p[1]); maxV = Math.max(maxV, p[1]);
    });
    return { x: Math.trunc(minU), y: Math.trunc(minV), width: Math.trunc(maxU - minU), height: Math.trunc(maxV - minV) };
  }

  function computeFov(boxHeight, panoHeight, hStar, rHead, degMin, degMax, outputHeight) {
    rHead = rHead === undefined ? 0.15 : rHead;
    degMin = degMin === undefined ? 20 : degMin;
    degMax = degMax === undefined ? 110 : degMax;
    outputHeight = outputHeight || 480;
    var alphaPerson = (boxHeight / panoHeight) * Math.PI;
    var alphaHead = rHead * alphaPerson;
    var denom = Math.max(1e-6, hStar);
    var t = (outputHeight * Math.tan(alphaHead * 0.5)) / denom;
    var tMin = Math.tan((degMin * Math.PI / 180) * 0.5), tMax = Math.tan((degMax * Math.PI / 180) * 0.5);
    if (!isFinite(t) || t <= 0) t = tMax;
    t = Math.min(Math.max(t, tMin), tMax);
    return 2 * Math.atan(t) * 180 / Math.PI;
  }

  // sort_tracker.cpp Sort::inject: aim a virtual camera at the top of a person box.
  function viewerForBox(box, rows, cols, headHeightRatio) {
    var centerX = box.x + box.width * 0.5;
    var lambda = (centerX / cols) * 2 * Math.PI - Math.PI;
    var phiTop = (0.5 - box.y / rows) * Math.PI;
    var hStar = Math.trunc(480 * headHeightRatio);
    var fov = computeFov(box.height, rows, hStar);
    var pitch = phiTop - 0.3 * (fov * Math.PI / 180);
    return { yaw: lambda * 180 / Math.PI, pitch: pitch * 180 / Math.PI, fov: fov, hStar: hStar };
  }

  /* --- opnet_tracker.cpp: head + eye combination ------------------------ */
  function quatFromAxisAngle(angle, axis) {
    var u = unit(axis), s = Math.sin(angle / 2);
    return [Math.cos(angle / 2), s * u[0], s * u[1], s * u[2]];
  }
  function rotationFromTwoVectors(a, b) {
    var axis = cross(a, b), d = dot(a, b), len = norm(axis);
    var angle = Math.atan2(len, d);
    if (len === 0) { angle = 0; axis = [1, 0, 0]; }
    return quatFromAxisAngle(angle, axis);
  }
  function quatMul(a, b) {
    return [
      a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
      a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
      a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
      a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    ];
  }
  function quatNlerp(q0, q1, t) {
    var d = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
    var a = q0.slice();
    if (d < 0) a = a.map(function (v) { return -v; });
    var r = a.map(function (v, i) { return (1 - t) * v + t * q1[i]; });
    var n = Math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]);
    return r.map(function (v) { return v / n; });
  }
  function quatForward(q) {
    var w = q[0], x = q[1], y = q[2], z = q[3];
    return [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)];
  }

  var KALPHA = 0.4, CONF_MIN = 20, CONF_MAX = 40;

  function eyeWeight(confidence) {
    return clamp((confidence * 100 - CONF_MIN) / (CONF_MAX - CONF_MIN), 0, 1);
  }

  function combineGaze(headYawDeg, headPitchDeg, l2csYawDeg, l2csPitchDeg, confidence) {
    var compYaw = l2csYawDeg - KALPHA * headYawDeg;
    var compPitch = l2csPitchDeg - KALPHA * headPitchDeg;
    var headDir = dirFromYawPitch(headYawDeg * DEG, headPitchDeg * DEG);
    var qHead = rotationFromTwoVectors([0, 0, 1], headDir);
    var eyeDir = dirFromYawPitch(compYaw * DEG, compPitch * DEG);
    var qEye = rotationFromTwoVectors([0, 0, 1], eyeDir);
    var w = eyeWeight(confidence);
    var qWeighted = quatNlerp([1, 0, 0, 0], qEye, w);
    var f = quatForward(quatMul(qHead, qWeighted));
    var yaw = Math.atan2(f[0], f[2]) / DEG;
    var pitch = Math.asin(clamp(f[1], -1, 1)) / DEG;
    return {
      compYaw: compYaw, compPitch: compPitch, eyeWeight: w,
      yaw: yaw, pitch: pitch, yawInt: Math.trunc(yaw), pitchInt: Math.trunc(pitch)
    };
  }

  // model_core.cpp L2CSEstimator::decode_angle_from_logits (90 bins, 4 degrees each)
  function decodeL2cs(logits) {
    var max = Math.max.apply(null, logits), sum = 0;
    logits.forEach(function (l) { sum += Math.exp(l - max); });
    var expected = 0, maxProb = 0;
    var probs = logits.map(function (l, i) {
      var p = Math.exp(l - max) / sum;
      expected += p * i;
      maxProb = Math.max(maxProb, p);
      return p;
    });
    return { angle: expected * 4 - 180, confidence: maxProb, probs: probs };
  }

  /* --- gaze_bayes_mapper.cpp ------------------------------------------- */
  var CONFIG_DEFAULT = { sigma_x_deg: 22, sigma_y_deg: 11, omega_base: 1, omega_motion: 1.5, omega_null: 0, likelihood_null_baseline: 0.15 };
  var CONFIG_INI = { sigma_x_deg: 15, sigma_y_deg: 11, omega_base: 1, omega_motion: 1.4, omega_null: 0.4, likelihood_null_baseline: 0.2 };

  function rectCenter(r) { return [r.x + 0.5 * r.width, r.y + 0.5 * r.height]; }

  function motionFromHeadRect(rect, cache, id) {
    if (rect.width <= 0 || rect.height <= 0) return 0;
    var c = rectCenter(rect);
    var scale = Math.max(Math.sqrt(rect.width * rect.width + rect.height * rect.height), 1);
    if (!cache.has(id)) { cache.set(id, rect); return 0; }
    var pc = rectCenter(cache.get(id));
    var delta = Math.hypot(c[0] - pc[0], c[1] - pc[1]);
    cache.set(id, rect);
    return clamp(delta / scale, 0, 1);
  }

  function buildHeadFrame(forwardIn) {
    var forward = unit(forwardIn);
    if (norm(forward) <= 1e-6) return { right: [1, 0, 0], up: [0, 1, 0] };
    var worldUp = [0, 1, 0];
    if (Math.abs(dot(forward, worldUp)) > 0.999) worldUp = [0, 0, 1];
    var right = unit(cross(worldUp, forward));
    return { right: right, up: unit(cross(forward, right)) };
  }

  function anisotropicLikelihood(forward, frame, origin, target, sigmaX, sigmaY) {
    var to = unit(sub(target, origin));
    if (norm(to) <= 1e-6) return { value: 0, thetaX: 0, thetaY: 0, behind: true };
    var lx = dot(to, frame.right), ly = dot(to, frame.up), lz = dot(to, forward);
    if (lz <= 1e-4) return { value: 0, thetaX: Math.atan2(lx, lz), thetaY: Math.atan2(ly, lz), behind: true };
    var thetaX = Math.atan2(lx, lz), thetaY = Math.atan2(ly, lz);
    var exponent = (thetaX * thetaX) / (2 * sigmaX * sigmaX) + (thetaY * thetaY) / (2 * sigmaY * sigmaY);
    return { value: Math.exp(-exponent), thetaX: thetaX, thetaY: thetaY, behind: false };
  }

  function angleBetweenDeg(dir, origin, target) {
    var to = unit(sub(target, origin)), d = unit(dir);
    if (norm(d) <= 1e-6 || norm(to) <= 1e-6) return 180;
    return Math.acos(clamp(dot(d, to), -1, 1)) * (180 / PI);
  }

  function softmax(values) {
    if (!values.length) return [];
    var max = Math.max.apply(null, values), sum = 0;
    var out = values.map(function (v) { var e = Math.exp(v - max); sum += e; return e; });
    return out.map(function (v) { return v / sum; });
  }

  // gazes: [{id, start:[x,y,z], direction:[x,y,z], box:{x,y,width,height}}]
  // cache: a Map kept between frames. Omit it to supply `motion` on each gaze.
  function inferGaze(gazes, cfg, cache) {
    var result = { interactions: [], observers: [] };
    if (gazes.length < 2) return result;
    var sigmaX = cfg.sigma_x_deg * DEG, sigmaY = cfg.sigma_y_deg * DEG;
    var motion = new Map();
    if (cache) {
      var active = new Set(gazes.map(function (g) { return g.id; }));
      Array.from(cache.keys()).forEach(function (id) { if (!active.has(id)) cache.delete(id); });
    }
    gazes.forEach(function (g) {
      motion.set(g.id, cache ? motionFromHeadRect(g.box, cache, g.id) : (g.motion || 0));
    });

    gazes.forEach(function (observer, i) {
      var frame = buildHeadFrame(observer.direction);
      var forward = unit(observer.direction);
      var slots = [{ id: -1, gaze: null }];
      gazes.forEach(function (g, j) { if (j !== i) slots.push({ id: g.id, gaze: g }); });

      var raw = [cfg.omega_null], likes = [Math.max(cfg.likelihood_null_baseline, 1e-6)], geometry = [null];
      slots.slice(1).forEach(function (s) {
        raw.push(cfg.omega_base + cfg.omega_motion * motion.get(s.id));
        var l = anisotropicLikelihood(forward, frame, observer.start, s.gaze.start, sigmaX, sigmaY);
        likes.push(Math.max(l.value, 1e-6));
        geometry.push(l);
      });

      var priors = softmax(raw);
      var unnormalized = likes.map(function (l, t) { return l * priors[t]; });
      var sum = unnormalized.reduce(function (a, b) { return a + b; }, 0);
      if (sum <= 1e-12) sum = 1;
      var posteriors = unnormalized.map(function (v) { return v / sum; });

      var best = -1, bestP = -1;
      posteriors.forEach(function (p, t) { if (p > bestP) { bestP = p; best = slots[t].id; } });

      slots.slice(1).forEach(function (s, k) {
        result.interactions.push({
          from: observer.id, to: s.id,
          angle: angleBetweenDeg(observer.direction, observer.start, s.gaze.start),
          isLooking: best === s.id, probability: posteriors[k + 1]
        });
      });
      result.observers.push({
        id: observer.id, best: best,
        targets: slots.map(function (s, t) {
          return {
            id: s.id, raw: raw[t], prior: priors[t], likelihood: likes[t], posterior: posteriors[t],
            motion: s.id === -1 ? 0 : motion.get(s.id),
            thetaX: geometry[t] ? geometry[t].thetaX / DEG : null,
            thetaY: geometry[t] ? geometry[t].thetaY / DEG : null,
            behind: geometry[t] ? geometry[t].behind : false
          };
        })
      });
    });
    return result;
  }

  /* --- sort_tracker.cpp ------------------------------------------------ */
  function iou(a, b) {
    var x0 = Math.max(a.x, b.x), y0 = Math.max(a.y, b.y);
    var x1 = Math.min(a.x + a.width, b.x + b.width), y1 = Math.min(a.y + a.height, b.y + b.height);
    var inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
    var un = a.width * a.height + b.width * b.height - inter;
    return un > 0 ? inter / un : 0;
  }

  /* --- python_client: extract_labels.py and evaluation.py -------------- */
  // labels: [{start, end, src, target}], target null means GazeRobot.
  function acceptedTargets(labels, t) {
    var out = new Map();
    labels.forEach(function (l) {
      if (l.start <= t && t < l.end) {
        if (!out.has(l.src)) out.set(l.src, new Set());
        out.get(l.src).add(l.target);
      }
    });
    return out;
  }

  // edges: [{from, to, isLooking, probability}]
  function compareEdges(labels, t, edges) {
    var accepted = acceptedTargets(labels, t);
    var predicted = new Map();
    edges.forEach(function (e) { if (e.isLooking) predicted.set(e.from, e.to); });
    var mismatches = [];
    edges.forEach(function (e) {
      var set = accepted.get(e.from);
      if (!set) return;
      var p = predicted.has(e.from) ? predicted.get(e.from) : null;
      if (set.has(p)) return;
      if (e.isLooking) {
        var people = Array.from(set).filter(function (x) { return x !== null; });
        mismatches.push({ from: e.from, to: e.to, kind: people.length ? "wrong_target" : "false_positive", probability: e.probability });
      } else if (set.has(e.to)) {
        mismatches.push({ from: e.from, to: e.to, kind: "false_negative", probability: e.probability });
      }
    });
    return mismatches;
  }

  function auroc(probabilities, truth) {
    var pos = truth.filter(Boolean).length, neg = truth.length - pos;
    if (!pos || !neg) return null;
    var order = probabilities.map(function (p, i) { return [p, i]; }).sort(function (a, b) { return a[0] - b[0]; });
    var ranks = new Array(order.length), i = 0;
    while (i < order.length) {
      var j = i;
      while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j++;
      for (var k = i; k <= j; k++) ranks[order[k][1]] = (i + j) / 2 + 1;
      i = j + 1;
    }
    var sum = 0;
    truth.forEach(function (t, idx) { if (t) sum += ranks[idx]; });
    return (sum - pos * (pos + 1) / 2) / (pos * neg);
  }

  function precisionRecallF1(probabilities, truth, threshold) {
    var tp = 0, predicted = 0, actual = 0;
    probabilities.forEach(function (p, i) {
      var hit = p >= threshold;
      if (hit) predicted++;
      if (truth[i]) actual++;
      if (hit && truth[i]) tp++;
    });
    var precision = predicted ? tp / predicted : null, recall = actual ? tp / actual : null;
    var f1 = precision !== null && recall !== null && precision + recall > 0 ? 2 * precision * recall / (precision + recall) : null;
    return { precision: precision, recall: recall, f1: f1 };
  }

  return {
    DEG: DEG, CONFIG_DEFAULT: CONFIG_DEFAULT, CONFIG_INI: CONFIG_INI, KALPHA: KALPHA, CONF_MIN: CONF_MIN, CONF_MAX: CONF_MAX,
    clamp: clamp, unit: unit, dot: dot, cross: cross, sub: sub, norm: norm,
    dirFromYawPitch: dirFromYawPitch, globalGazeFromPanorama: globalGazeFromPanorama, addGaze: addGaze,
    perspectiveToPano: perspectiveToPano, perspectiveRectToPano: perspectiveRectToPano,
    computeFov: computeFov, viewerForBox: viewerForBox,
    eyeWeight: eyeWeight, combineGaze: combineGaze, decodeL2cs: decodeL2cs,
    inferGaze: inferGaze, softmax: softmax, angleBetweenDeg: angleBetweenDeg, motionFromHeadRect: motionFromHeadRect,
    iou: iou, acceptedTargets: acceptedTargets, compareEdges: compareEdges, auroc: auroc, precisionRecallF1: precisionRecallF1
  };
});
