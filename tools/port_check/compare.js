// node compare.js gen cases.txt            writes random inputs
// node compare.js check cases.txt out.txt  compares model.js with the C++ output
const fs = require("fs");
const path = require("path");
const M = require(path.join(__dirname, "..", "..", "assets", "js", "model.js"));

let seed = 12345;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
const range = (a, b) => a + (b - a) * rnd();
const round = (v, d = 3) => Number(v.toFixed(d));

function randGaze(id) {
  return [id, round(range(-150, 150)), round(range(-40, 40)), round(range(-150, 150)),
    round(range(-1, 1)), round(range(-0.5, 0.5)), round(range(-1, 1)),
    round(range(100, 400)), round(range(100, 300)), round(range(20, 90)), round(range(20, 90))];
}

function gen(file) {
  const lines = [];
  for (let i = 0; i < 300; i++) {
    lines.push(["combine", round(range(-70, 70)), round(range(-40, 40)), round(range(-80, 80)), round(range(-40, 40)), round(range(0, 0.7))].join(" "));
    lines.push(["addgaze", round(range(-180, 180)), round(range(-40, 40)), 60, Math.trunc(range(-90, 90)), Math.trunc(range(-50, 50)),
      Math.trunc(range(-80, 80)), Math.trunc(range(-40, 40)), Math.trunc(range(40, 200))].join(" "));
    lines.push(["rect", round(range(-180, 180)), round(range(-40, 40)), round(range(25, 100)),
      Math.trunc(range(0, 400)), Math.trunc(range(0, 300)), Math.trunc(range(30, 200)), Math.trunc(range(30, 170)), 3840, 1920].join(" "));
    lines.push(["fov", Math.trunc(range(40, 1200)), 1920, Math.trunc(range(30, 200))].join(" "));
    const n = 2 + Math.trunc(range(0, 3));
    const cfg = [round(range(5, 40)), round(range(5, 30)), round(range(-1, 2)), round(range(0, 4)), round(range(-1, 2)), round(range(0.01, 0.6))];
    const f1 = [], f2 = [];
    for (let k = 0; k < n; k++) {
      const g = randGaze(k + 1);
      f1.push(g);
      const h = g.slice();
      h[7] = round(h[7] + range(-40, 40));
      h[8] = round(h[8] + range(-40, 40));
      h[4] = round(h[4] + range(-0.2, 0.2));
      f2.push(h);
    }
    lines.push(["bayes", n, ...cfg, ...f1.flat(), ...f2.flat()].join(" "));
  }
  fs.writeFileSync(file, lines.join("\n") + "\n");
}

function check(casesFile, outFile) {
  const cases = fs.readFileSync(casesFile, "utf8").trim().split(/\r?\n/);
  const outs = fs.readFileSync(outFile, "utf8").trim().split(/\r?\n/);
  if (cases.length !== outs.length) throw new Error(`line count ${cases.length} vs ${outs.length}`);
  const worst = {};
  const fails = {};
  const note = (k, a, b, tol) => {
    const d = Math.abs(a - b);
    worst[k] = Math.max(worst[k] || 0, d);
    if (!(d <= tol)) fails[k] = (fails[k] || 0) + 1;
  };
  const counts = {};
  cases.forEach((line, i) => {
    const p = line.split(" ");
    const kind = p[0];
    const v = p.slice(1).map(Number);
    const o = outs[i].split(" ").slice(1).map(Number);
    counts[kind] = (counts[kind] || 0) + 1;
    if (kind === "combine") {
      const r = M.combineGaze(v[0], v[1], v[2], v[3], v[4]);
      note("combine.eye_weight", r.eyeWeight, o[0], 1e-4);
      note("combine.yaw", r.yaw, o[1], 2e-2);
      note("combine.pitch", r.pitch, o[2], 2e-2);
    } else if (kind === "addgaze") {
      const r = M.addGaze(v[0], v[1], v[3], v[4], [v[5], v[6], v[7]]);
      [...r.start].forEach((x, k) => note("addgaze.start", x, o[k], 1e-2));
      [...r.direction].forEach((x, k) => note("addgaze.direction", x, o[3 + k], 1e-3));
    } else if (kind === "rect") {
      const r = M.perspectiveRectToPano({ x: v[3], y: v[4], width: v[5], height: v[6] }, { yaw: v[0], pitch: v[1], fov: v[2] }, v[7], v[8]);
      note("rect.x", r.x, o[0], 2); note("rect.y", r.y, o[1], 2);
      note("rect.w", r.width, o[2], 2); note("rect.h", r.height, o[3], 2);
    } else if (kind === "fov") {
      note("fov", M.computeFov(v[0], v[1], v[2]), o[0], 1e-3);
    } else if (kind === "bayes") {
      const n = v[0];
      const cfg = { sigma_x_deg: v[1], sigma_y_deg: v[2], omega_base: v[3], omega_motion: v[4], omega_null: v[5], likelihood_null_baseline: v[6] };
      const parse = (off) => {
        const g = [];
        for (let k = 0; k < n; k++) {
          const a = v.slice(off + k * 11, off + (k + 1) * 11);
          g.push({ id: a[0], start: a.slice(1, 4), direction: a.slice(4, 7), box: { x: a[7], y: a[8], width: a[9], height: a[10] } });
        }
        return g;
      };
      const cache = new Map();
      M.inferGaze(parse(7), cfg, cache);
      const r = M.inferGaze(parse(7 + n * 11), cfg, cache).interactions;
      if (r.length * 5 !== o.length) throw new Error("edge count");
      r.forEach((e, k) => {
        const c = o.slice(k * 5, k * 5 + 5);
        note("bayes.from", e.from, c[0], 0); note("bayes.to", e.to, c[1], 0);
        note("bayes.angle", e.angle, c[2], 2e-2);
        note("bayes.is_looking", e.isLooking ? 1 : 0, c[3], 0);
        note("bayes.probability", e.probability, c[4], 1e-3);
      });
    }
  });
  console.log("cases", counts);
  Object.keys(worst).sort().forEach((k) => console.log(k.padEnd(22), "worst diff", worst[k].toExponential(2), fails[k] ? `FAILS ${fails[k]}` : "ok"));
  const failed = Object.keys(fails).length;
  console.log(failed ? "PORT CHECK FAILED" : "PORT CHECK PASSED");
  process.exit(failed ? 1 : 0);
}

const [mode, a, b] = process.argv.slice(2);
if (mode === "gen") gen(a); else check(a, b);
