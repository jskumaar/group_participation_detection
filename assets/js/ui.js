/* Shared helpers: DOM, theme, navigation, code links, sliders, canvas setup. */
(function () {
  "use strict";

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === "class") node.className = attrs[k];
      else if (k === "text") node.textContent = attrs[k];
      else if (k === "html") node.innerHTML = attrs[k];
      else if (k.slice(0, 2) === "on") node.addEventListener(k.slice(2), attrs[k]);
      else if (attrs[k] !== undefined && attrs[k] !== null) node.setAttribute(k, attrs[k]);
    });
    (children || []).forEach(function (c) {
      if (c) node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return node;
  }

  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function storageGet(key) { try { return localStorage.getItem(key); } catch (e) { return null; } }
  function storageSet(key, value) { try { localStorage.setItem(key, value); } catch (e) { /* private mode */ } }

  var themeListeners = [];
  function onTheme(fn) { themeListeners.push(fn); }

  function setupTheme() {
    var saved = storageGet("onboard-theme");
    if (saved) document.documentElement.setAttribute("data-theme", saved);
    function isDark() {
      var t = document.documentElement.getAttribute("data-theme");
      return t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    }
    function label() {
      document.querySelectorAll("[data-theme-toggle]").forEach(function (b) {
        b.textContent = isDark() ? "Light theme" : "Dark theme";
      });
    }
    document.querySelectorAll("[data-theme-toggle]").forEach(function (b) {
      b.addEventListener("click", function () {
        var next = isDark() ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", next);
        storageSet("onboard-theme", next);
        label();
        themeListeners.forEach(function (fn) { fn(); });
      });
    });
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
      label();
      themeListeners.forEach(function (fn) { fn(); });
    });
    label();
  }

  function setupNav() {
    var links = Array.from(document.querySelectorAll(".side nav a"));
    var select = document.getElementById("mobile-select");
    links.forEach(function (a) {
      if (select) select.appendChild(el("option", { value: a.getAttribute("href"), text: a.textContent }));
    });
    if (select) select.addEventListener("change", function () { location.hash = select.value; });
    var byId = {};
    links.forEach(function (a) { byId[a.getAttribute("href").slice(1)] = a; });
    var seen = new Map();
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { seen.set(e.target.id, e.isIntersecting ? e.boundingClientRect.top : null); });
      var current = null, bestTop = Infinity;
      seen.forEach(function (top, id) {
        if (top !== null && top < bestTop) { bestTop = top; current = id; }
      });
      if (!current) return;
      links.forEach(function (a) { a.classList.toggle("active", a === byId[current]); });
      if (select) select.value = "#" + current;
    }, { rootMargin: "0px 0px -70% 0px" });
    Object.keys(byId).forEach(function (id) {
      var target = document.getElementById(id);
      if (target) observer.observe(target);
    });
  }

  function refUrl(r) {
    return window.REPO.url + "/blob/" + window.REPO.commit + "/" + r.file + "#L" + r.line;
  }

  function fileUrl(path) {
    return window.REPO.url + "/blob/" + window.REPO.commit + "/" + path;
  }

  function refLabel(r) {
    return r.file.replace("src/c_impl/", "").replace("src/python_client/", "python_client/") + ":" + r.line;
  }

  function refLink(key, text) {
    var r = window.REFS[key];
    if (!r) throw new Error("unknown ref " + key);
    return el("a", { class: "ref", href: refUrl(r), target: "_blank", rel: "noopener", text: text || refLabel(r) });
  }

  function setupRefs(root) {
    root = root || document;
    root.querySelectorAll("[data-ref]").forEach(function (a) {
      var r = window.REFS[a.getAttribute("data-ref")];
      if (!r) throw new Error("unknown ref " + a.getAttribute("data-ref"));
      a.href = refUrl(r);
      a.target = "_blank";
      a.rel = "noopener";
      a.classList.add("ref");
      if (!a.textContent) a.textContent = refLabel(r);
    });
    root.querySelectorAll("[data-snippet]").forEach(function (box) {
      if (box.childElementCount) return;
      var r = window.REFS[box.getAttribute("data-snippet")];
      if (!r || !r.snippet) throw new Error("no snippet " + box.getAttribute("data-snippet"));
      box.classList.add("snippet");
      box.appendChild(el("a", { class: "src", href: refUrl(r), target: "_blank", rel: "noopener", text: refLabel(r) }));
      box.appendChild(el("pre", {}, [el("code", { text: r.snippet })]));
    });
  }

  function setupCommands() {
    document.querySelectorAll("pre.cmd-src").forEach(function (pre) {
      var box = el("div", { class: "cmd" });
      pre.parentNode.insertBefore(box, pre);
      box.appendChild(pre);
      var btn = el("button", { class: "btn copy", type: "button", text: "Copy" });
      btn.addEventListener("click", function () {
        var done = function () { btn.textContent = "Copied"; setTimeout(function () { btn.textContent = "Copy"; }, 1400); };
        if (navigator.clipboard) navigator.clipboard.writeText(pre.textContent.trim()).then(done, function () { btn.textContent = "Select it"; });
        else btn.textContent = "Select it";
      });
      box.appendChild(btn);
    });
  }

  function setupChecklist() {
    var boxes = document.querySelectorAll(".check input[type=checkbox]");
    var saved = {};
    try { saved = JSON.parse(storageGet("onboard-check") || "{}"); } catch (e) { saved = {}; }
    var count = document.getElementById("check-count");
    function update() {
      if (count) count.textContent = Array.from(boxes).filter(function (b) { return b.checked; }).length + " of " + boxes.length + " done";
    }
    boxes.forEach(function (b) {
      if (saved[b.id]) b.checked = true;
      b.addEventListener("change", function () {
        saved[b.id] = b.checked;
        storageSet("onboard-check", JSON.stringify(saved));
        update();
      });
    });
    update();
  }

  // A labelled range input. Returns {input, get, set}.
  function slider(parent, opts) {
    var id = "s" + Math.random().toString(36).slice(2, 9);
    var format = opts.format || function (v) { return String(v); };
    var input = el("input", { type: "range", id: id, min: opts.min, max: opts.max, step: opts.step || 1, value: opts.value });
    var output = el("output", { for: id, text: format(opts.value) });
    var box = el("div", { class: "slider" }, [el("label", { for: id, text: opts.label }), output, input]);
    parent.appendChild(box);
    function get() { return Number(input.value); }
    input.addEventListener("input", function () {
      output.textContent = format(get());
      if (opts.onInput) opts.onInput(get());
    });
    return {
      input: input,
      get: get,
      set: function (v) { input.value = v; output.textContent = format(get()); }
    };
  }

  // Sizes a canvas for the device. Returns a 2d context drawing in CSS-like units.
  function hidpi(canvas, width, height) {
    var ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    canvas.style.aspectRatio = width + " / " + height;
    var ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return ctx;
  }

  function canvasPoint(canvas, event, width, height) {
    var r = canvas.getBoundingClientRect();
    return [(event.clientX - r.left) * width / r.width, (event.clientY - r.top) * height / r.height];
  }

  var tip;
  function showTip(event, nodes) {
    if (!tip) { tip = el("div", { class: "tip" }); document.body.appendChild(tip); }
    tip.textContent = "";
    nodes.forEach(function (n) { tip.appendChild(n); });
    tip.style.display = "block";
    var x = Math.min(event.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    var y = Math.max(event.clientY - tip.offsetHeight - 10, 8);
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }
  function hideTip() { if (tip) tip.style.display = "none"; }

  var PEOPLE = {
    1: { name: "P1", cls: "p1", color: "--p1" },
    2: { name: "P2", cls: "p2", color: "--p2" },
    3: { name: "P3", cls: "p3", color: "--p3" }
  };

  function personColor(id) { return PEOPLE[id] ? css(PEOPLE[id].color) : css("--none"); }
  function personName(id) { return PEOPLE[id] ? PEOPLE[id].name : "no one"; }
  function personClass(id) { return PEOPLE[id] ? PEOPLE[id].cls : ""; }

  // One shared set of rendered bars for prior / likelihood / posterior style data.
  function bars(parent, rows, max) {
    parent.textContent = "";
    var box = el("div", { class: "bars" });
    rows.forEach(function (r) {
      var width = Math.max(0, Math.min(1, r.value / (max || 1))) * 100;
      box.appendChild(el("div", { class: "bar-row" + (r.best ? " best" : "") }, [
        el("span", { text: r.label }),
        el("div", { class: "track" }, [el("div", { class: "fill " + (r.cls || ""), style: "width:" + width.toFixed(1) + "%" })]),
        el("span", { class: "val", text: r.text !== undefined ? r.text : r.value.toFixed(3) })
      ]));
    });
    parent.appendChild(box);
  }

  window.UI = {
    el: el, css: css, slider: slider, hidpi: hidpi, canvasPoint: canvasPoint, bars: bars,
    showTip: showTip, hideTip: hideTip, onTheme: onTheme, refLink: refLink, refUrl: refUrl, fileUrl: fileUrl, setupRefs: setupRefs,
    personColor: personColor, personName: personName, personClass: personClass, storageGet: storageGet, storageSet: storageSet,
    boot: function () { setupTheme(); setupNav(); setupRefs(); setupCommands(); setupChecklist(); }
  };
})();
