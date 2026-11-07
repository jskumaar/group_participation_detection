/* System map, frame walkthrough and the YOLO decision widget. */
(function () {
  "use strict";
  var el = UI.el;
  var SVGNS = "http://www.w3.org/2000/svg";

  function svg(tag, attrs, parent) {
    var n = document.createElementNS(SVGNS, tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (parent) parent.appendChild(n);
    return n;
  }

  /* ---------------------------------------------------- system map */
  var NODES = {
    video: { x: 30, y: 60, t: "VideoSource", s: "media/video_source", d: "Wraps cv::VideoCapture. Reads one frame at a time and reports the frame index and the timestamp in nanoseconds. Those timestamps are what the labels get matched against.", f: ["src/c_impl/media/video_source.cpp"], r: "video_read" },
    ctrl: { x: 230, y: 60, t: "AppController", s: "app/app_controller", d: "The conductor. Owns the 25 ms timer, the selection state, and every output. onTick reads a frame, processTrackingFrame runs the pipeline, then it writes the CSV, emits UI signals and publishes to gRPC. It contains no vision logic.", f: ["src/c_impl/app/app_controller.cpp"], r: "process_tracking" },
    pipe: { x: 460, y: 60, t: "VisionPipeline", s: "pipelines/vision_pipeline", d: "All per-frame vision logic in one function, processFrame. It decides whether YOLO runs, keeps the list of people, loops over people to get a pose, then calls Sort and the gaze mapper. It has no Qt UI code, which makes it the easiest thing to reuse without the app.", f: ["src/c_impl/pipelines/vision_pipeline.cpp"], r: "process_frame" },
    opnet: { x: 460, y: 150, t: "OPNetTracker", s: "vision/opnet_tracker", d: "Runs the neural networks. detect() is the heart of it: localizer, pose estimator, L2CS, then the head and eye blend. It also holds TrackerConfig and the ONNX Runtime session setup. The network wrappers themselves are in vision/model_core.", f: ["src/c_impl/vision/opnet_tracker.cpp", "src/c_impl/vision/model_core.cpp"], r: "detect" },
    sort: { x: 460, y: 250, t: "Sort + PanoViewer", s: "vision/sort_tracker, 360_image_process", d: "Sort keeps IDs stable and creates one PanoViewer per person. PanoViewer holds the virtual camera: it renders the crop and converts results back to panorama pixels and world coordinates.", f: ["src/c_impl/vision/sort_tracker.cpp", "src/c_impl/vision/360_image_process.cpp"], r: "sort_update" },
    bayes: { x: 460, y: 350, t: "GazeBayesMapper", s: "vision/gaze_bayes_mapper", d: "Turns gaze rays into interaction edges with a posterior probability. It keeps the previous head boxes between frames to measure motion. Pure maths, with no dependency on the models.", f: ["src/c_impl/vision/gaze_bayes_mapper.cpp"], r: "bayes_infer" },
    window: { x: 30, y: 190, t: "MainWindow", s: "app/mainwindow", d: "Builds the UI, connects AppController signals to widgets, and owns the Settings dialog and the ini file. Business logic does not belong here.", f: ["src/c_impl/app/mainwindow.cpp"], r: "show_settings" },
    videow: { x: 30, y: 270, t: "VideoWidget", s: "app/videowidget", d: "Draws the frame and the boxes, and turns mouse clicks into boxClicked(index) during selection.", f: ["src/c_impl/app/videowidget.cpp"], r: "video_click" },
    vtk: { x: 30, y: 350, t: "GazeVisualizer", s: "app/gazevisualizer", d: "VTK 3D view of heads and gaze arrows. Very useful for checking geometry by eye, but it costs frame rate. Untick Show 3D for long runs.", f: ["src/c_impl/app/gazevisualizer.cpp"], r: "gv_update" },
    config: { x: 30, y: 460, t: "TrackerConfig", s: "vision/opnet_tracker.h", d: "One struct with almost every tunable number, including the nested GazeBayesConfig. Values come from code defaults, then tracker_settings.ini, then the Settings dialog.", f: ["src/c_impl/vision/opnet_tracker.h"], r: "tracker_config" },
    csv: { x: 230, y: 350, t: "CsvGazeWriter", s: "io/csv_gaze_writer", d: "Writes gaze_analysis.csv, one row per person per frame. It overwrites the file at each launch. The evaluation does not use it, but it is handy for quick plots.", f: ["src/c_impl/io/csv_gaze_writer.cpp"], r: "csv_write" },
    grpc: { x: 230, y: 460, t: "GrpcPipelinePublisher", s: "io/grpc_pipeline_*", d: "Runs the gRPC server. publish() converts results to protobuf and pushes them into a bounded queue. A stream thread sends from that queue, so the pipeline never waits for a client.", f: ["src/c_impl/io/grpc_pipeline_publisher.cpp", "src/c_impl/io/grpc_pipeline_stream_service.cpp"], r: "svc_publish" },
    client: { x: 750, y: 460, t: "grpc_client", s: "python_client/grpc_client.py", d: "Connects, waits for the tracker, and feeds every update to the logger and the evaluator. Reports are created on the first update, and a final summary is printed when the stream ends or on Ctrl+C.", f: ["src/python_client/grpc_client.py"], r: "client_once" },
    labels: { x: 750, y: 350, t: "extract_labels", s: "python_client/extract_labels.py", d: "Parses the labels CSV, answers which targets are accepted at a time, and finds mismatches. UniqueMismatchLogger merges consecutive mismatched frames into one event in report_N.jsonl.", f: ["src/python_client/extract_labels.py"], r: "labels_compare" },
    evalu: { x: 750, y: 250, t: "evaluation", s: "python_client/evaluation.py", d: "Keeps running totals and computes the final metrics: accuracy, confusion matrix, AUROC, average precision, Brier score, and a precision and recall table by threshold.", f: ["src/python_client/evaluation.py"], r: "eval_class" },
    reports: { x: 750, y: 150, t: "logs/report_N", s: ".jsonl and _summary.json", d: "The output of a run: one JSON line per mismatch event, plus a summary file. Compare summaries across runs to see if a change helped.", f: [], r: "eval_format" },
    cli: { x: 750, y: 60, t: "rapport_cli", s: "python_client/rapport_cli.py", d: "The command line: serve runs the client, and replay prints mismatch events from a report, filtered by duration.", f: ["src/python_client/rapport_cli.py"], r: "cli_parser" }
  };
  var W = 170, H = 56;

  var EDGES = [
    { a: "video", b: "ctrl", d: "M200 88 H230", l: "frame", lx: 205, ly: 80 },
    { a: "ctrl", b: "pipe", d: "M400 88 H460", l: "processFrame", lx: 398, ly: 80 },
    { a: "pipe", b: "opnet", d: "M460 100 H445 V178 H460" },
    { a: "pipe", b: "sort", d: "M445 178 V278 H460" },
    { a: "pipe", b: "bayes", d: "M445 278 V378 H460" },
    { a: "ctrl", b: "window", d: "M315 116 V218 H200", l: "Qt signals", lx: 222, ly: 212 },
    { a: "window", b: "videow", d: "M115 246 V270" },
    { a: "videow", b: "vtk", d: "M115 326 V350" },
    { a: "ctrl", b: "csv", d: "M345 116 V350" },
    { a: "ctrl", b: "grpc", d: "M262 116 V170 H216 V488 H230" },
    { a: "grpc", b: "client", d: "M400 488 H750", l: "PipelineUpdate stream on 127.0.0.1:50051", lx: 462, ly: 480 },
    { a: "client", b: "labels", d: "M835 460 V406" },
    { a: "labels", b: "evalu", d: "M835 350 V306" },
    { a: "evalu", b: "reports", d: "M835 250 V206" },
    { a: "cli", b: "client", d: "M920 88 H942 V488 H920" }
  ];

  function buildMap() {
    var root = document.getElementById("demo-map");
    root.appendChild(el("div", { class: "demo-head" }, [
      el("h4", { text: "Every moving part" }),
      el("p", { text: "Click a box, or tab to it and press Enter." })
    ]));
    var body = el("div", { class: "demo-body" });
    var scroller = el("div", { class: "scroll-x" });
    var fig = svg("svg", { viewBox: "0 0 970 540", class: "fig scrollable", role: "group", "aria-label": "System components", style: "min-width:720px" });
    scroller.appendChild(fig);
    svg("rect", { x: 8, y: 14, width: 688, height: 516, rx: 12, class: "lane" }, fig);
    svg("rect", { x: 714, y: 14, width: 248, height: 516, rx: 12, class: "lane" }, fig);
    svg("text", { x: 24, y: 40, class: "lane-t" }, fig).textContent = "tracker.exe  (C++, Qt, one process)";
    svg("text", { x: 730, y: 40, class: "lane-t" }, fig).textContent = "python client  (a second process)";

    var defs = svg("defs", {}, fig);
    var marker = svg("marker", { id: "arrow", viewBox: "0 0 8 8", refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
    svg("path", { d: "M0 0 L8 4 L0 8 z", fill: "currentColor", style: "color:var(--rule-strong)" }, marker);

    EDGES.forEach(function (e) {
      e.path = svg("path", { d: e.d, class: "edge", "marker-end": "url(#arrow)" }, fig);
      if (e.l) svg("text", { x: e.lx, y: e.ly, class: "edge-t" }, fig).textContent = e.l;
    });

    var detail = el("div", { style: "margin-top:12px;min-height:128px" });
    var nodes = {};
    function select(key) {
      Object.keys(nodes).forEach(function (k) { nodes[k].classList.toggle("on", k === key); });
      EDGES.forEach(function (e) { e.path.classList.toggle("hot", e.a === key || e.b === key); });
      var n = NODES[key];
      detail.textContent = "";
      detail.appendChild(el("h4", { text: n.t, style: "margin-top:0" }));
      detail.appendChild(el("p", { text: n.d, style: "margin:.2rem 0 .4rem;font-size:.92rem" }));
      var links = el("p", { class: "note", style: "margin:0" });
      links.appendChild(document.createTextNode("Start reading at "));
      links.appendChild(UI.refLink(n.r));
      n.f.forEach(function (f) {
        links.appendChild(document.createTextNode("  ·  "));
        links.appendChild(el("a", { class: "ref", href: UI.fileUrl(f), target: "_blank", rel: "noopener", text: f.split("/").pop() }));
      });
      detail.appendChild(links);
    }

    Object.keys(NODES).forEach(function (key) {
      var n = NODES[key];
      var g = svg("g", { class: "node", tabindex: 0, role: "button", "aria-label": n.t }, fig);
      svg("rect", { x: n.x, y: n.y, width: W, height: H, rx: 8 }, g);
      svg("text", { x: n.x + 12, y: n.y + 24 }, g).textContent = n.t;
      svg("text", { x: n.x + 12, y: n.y + 42, class: "sub" }, g).textContent = n.s;
      g.addEventListener("click", function () { select(key); });
      g.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(key); }
      });
      nodes[key] = g;
    });

    body.appendChild(scroller);
    body.appendChild(detail);
    root.appendChild(body);
    select("pipe");
  }

  /* ---------------------------------------------------- frame walkthrough */
  var STEPS = [
    { t: "Timer fires", who: "AppController::onTick", r: "on_tick",
      b: "A QTimer fires every 25 ms on the main thread. If processing takes longer, the next tick simply waits. The video is therefore not played at real time: timestamps come from the file, so results do not depend on speed.",
      d: "isPlaying_   = true\nisSelecting_ = false\njustSeeked_ = false" },
    { t: "Read a frame", who: "VideoSource::read", r: "video_read",
      b: "One BGR frame is decoded. The controller also records the current frame number and timestamp, which travel with the results to the Python client.",
      d: "cv::Mat frame       3840 × 1920, CV_8UC3 (BGR)\nframe_index         1843\nplayback_timestamp  61.494 s" },
    { t: "Check and roll the panorama", who: "VisionPipeline::applyPanoramaOffset", r: "pano_offset",
      b: "Frames that are not exactly 2:1 are rejected here. Then the panorama is rolled sideways by panorama_offset_px, which lets you move the seam away from a person who sits on it.",
      d: "panorama_offset_px = 0, so no roll\ncheck: cols == rows * 2  (3840 == 3840)" },
    { t: "Run YOLO if needed", who: "VisionPipeline::processFrame", r: "yolo_cond",
      b: "Most frames skip this. When it runs, new person boxes are matched to the known people by IoU. If matching fails while enough people are visible, the app pauses and asks you to select again. Matched boxes are then turned into freshly aimed PanoViewers by Sort::inject.",
      d: "tracks_ (one per person)\n  viewer    yaw -148.2°  pitch -9.1°  fov 41.3°\n  face_roi  empty, or carried from the last frame" },
    { t: "Render a crop per person", who: "PanoViewer::generatePerspectiveView", r: "remap",
      b: "Each person's PanoViewer uses cv::remap and its lookup tables to render a 640×480 perspective view. Everything from here to step 8 repeats for each person.",
      d: "perspective_view   640 × 480, CV_8UC3\ngrayscale          640 × 480, CV_8U\n(localizer and pose use gray, L2CS uses colour)" },
    { t: "Find the face", who: "OPNetTracker::detect", r: "detect",
      b: "If this person's ROI is still valid, the localizer is skipped. Otherwise it runs to find the face. If no face is found, this person is skipped for this frame and YOLO is requested for the next one.",
      d: "roi_state.last_roi   x 248.1  y 118.4  w 91.7  h 95.2\nlocalizer            skipped: ROI still agrees" },
    { t: "Estimate head pose", who: "PoseEstimator::run", r: "pose_run",
      b: "The pose network returns head rotation as a quaternion, plus position, size and a tight face box. The face box updates the ROI for the next frame, smoothed by 50%.",
      d: "rotation     quaternion (w, x, y, z)\ncenter       (294.2, 165.0)\nsize         48.3 px\nbox          the new ROI, after smoothing" },
    { t: "Estimate eyes and blend", who: "L2CSEstimator::run, then detect", r: "kalpha",
      b: "L2CS gives a gaze angle and a confidence. The two are combined into one direction with quaternions. Confidence below 20% means head only, and above 40% means the eye correction is used in full.",
      d: "head        yaw 12°   pitch -4°\nl2cs        yaw 31°   pitch -9°\nconfidence  0.34, so eye_weight 0.70\nPose ints   yaw 30    pitch -8" },
    { t: "Convert to world space", who: "PanoViewer::addGaze", r: "add_gaze",
      b: "Crop-relative position and direction become world coordinates, centred on the camera, using the viewer's yaw and pitch. The face box is also converted back to panorama pixels for matching.",
      d: "PanoViewer::gaze\n  start          (-47.7, -20.5, -94.7) cm\n  direction      ( 0.25, -0.28,  0.93)\n  box            panorama pixels\n  box_projected  crop pixels (used for motion)" },
    { t: "Assign IDs", who: "Sort::update", r: "sort_update",
      b: "All of the frame's gazes are matched to tracked people by box overlap, and each gets a stable personID from 1 to num_people.",
      d: "gazes[0].personID = 1\ngazes[1].personID = 2\ngazes[2].personID = 3" },
    { t: "Infer who looks at whom", who: "GazeBayesMapper::infer", r: "bayes_infer",
      b: "For each observer, one probability per other person plus the null target. The most likely target sets is_looking. If fewer than two people were found this frame, this returns no edges.",
      d: "from 1 to 2  angle 11.8  is_looking true   p 0.74\nfrom 1 to 3  angle 58.2  is_looking false  p 0.06\n(remaining 0.20 is P1 looking at no one)\n... and the same for observers 2 and 3" },
    { t: "Publish everything", who: "AppController::processTrackingFrame", r: "publish_call",
      b: "Back in the controller, the results fan out: the CSV row, the text label, overlay boxes, the 3D view, the video, and the gRPC queue. Publishing only copies into the queue, so it cannot stall on a slow client.",
      d: "PipelineUpdate\n  frame_index            1843\n  playback_timestamp_ns  61494000000\n  gazes                  3\n  interactions           6" },
    { t: "Score it in Python", who: "GazeEvaluator.handle_update", r: "eval_handle",
      b: "In the other process, the client looks up which targets are accepted at this timestamp and compares. Mismatches join a running event in report_N.jsonl, and totals build up for the final report.",
      d: "t = 61.494 s\nP1 accepted {P2}  predicted P2      correct\nP2 accepted {no one}  predicted P3  wrong\nP3 unlabeled                        skipped" }
  ];

  function buildWalk() {
    var root = document.getElementById("demo-walk");
    root.appendChild(el("div", { class: "demo-head" }, [
      el("h4", { text: "Frame walkthrough" }),
      el("p", { text: "Example values are illustrative. Use the buttons, or focus this panel and press the left and right arrow keys." })
    ]));
    var body = el("div", { class: "demo-body", tabindex: 0 });
    var dots = el("div", { class: "steps", role: "group", "aria-label": "Steps" });
    var title = el("h4", { style: "margin:0 0 .15rem;font-size:1.08rem" });
    var who = el("p", { class: "note", style: "margin:0 0 .5rem" });
    var text = el("p", { style: "font-size:.95rem;margin:.3rem 0" });
    var data = el("pre", { style: "margin:0" });
    var dataLabel = el("p", { class: "fig-label", text: "What exists after this step" });
    var prev = el("button", { class: "btn", type: "button", text: "Previous" });
    var next = el("button", { class: "btn primary", type: "button", text: "Next step" });
    var index = 0;
    var buttons = STEPS.map(function (s, i) {
      var b = el("button", { type: "button", text: String(i + 1), "aria-label": "Step " + (i + 1) + ": " + s.t });
      b.addEventListener("click", function () { show(i); });
      dots.appendChild(b);
      return b;
    });
    function show(i) {
      index = Math.max(0, Math.min(STEPS.length - 1, i));
      var s = STEPS[index];
      buttons.forEach(function (b, k) {
        b.classList.toggle("done", k < index);
        if (k === index) b.setAttribute("aria-current", "step"); else b.removeAttribute("aria-current");
      });
      title.textContent = (index + 1) + ". " + s.t;
      who.textContent = s.who + "  ·  ";
      who.appendChild(UI.refLink(s.r));
      text.textContent = s.b;
      data.textContent = s.d;
      prev.disabled = index === 0;
      next.disabled = index === STEPS.length - 1;
    }
    prev.addEventListener("click", function () { show(index - 1); });
    next.addEventListener("click", function () { show(index + 1); });
    body.addEventListener("keydown", function (e) {
      if (e.key === "ArrowRight") { e.preventDefault(); show(index + 1); }
      if (e.key === "ArrowLeft") { e.preventDefault(); show(index - 1); }
    });
    var cols = el("div", { class: "cols even" }, [
      el("div", {}, [title, who, text]),
      el("div", {}, [dataLabel, data])
    ]);
    body.appendChild(dots);
    body.appendChild(cols);
    body.appendChild(el("div", { class: "btn-row", style: "margin-top:12px" }, [prev, next]));
    root.appendChild(body);
    show(0);
  }

  /* ---------------------------------------------------- YOLO decision */
  function buildYolo() {
    var root = document.getElementById("demo-yolo");
    root.appendChild(el("div", { class: "demo-head" }, [
      el("h4", { text: "Decide whether YOLO runs on this frame" }),
      el("p", { text: "Set the state of the pipeline. The result follows processFrame for 3 people." })
    ]));
    var body = el("div", { class: "demo-body cols even" });
    var left = el("div");
    var state = { need: false, empty: false, few: false, interval: false };
    function toggle(key, label) {
      var input = el("input", { type: "checkbox" });
      input.addEventListener("change", function () { state[key] = input.checked; update(); });
      left.appendChild(el("label", { class: "toggle" }, [input, el("span", { html: label })]));
    }
    toggle("need", "<code>need_yolo_update()</code>: a person's face or pose was lost");
    toggle("empty", "<code>tracks_.empty()</code>: no people tracked yet");
    toggle("few", "<code>tracks_.size() &lt; num_people</code>: someone was dropped");
    toggle("interval", "Every 30th frame (<code>yolo_check_interval</code>)");
    var raw = UI.slider(left, { label: "People YOLO finds", min: 0, max: 6, value: 3, onInput: update });
    var matched = UI.slider(left, { label: "Of those, matched to tracked people", min: 0, max: 3, value: 3, onInput: update });
    var out = el("div");
    function update() {
      out.textContent = "";
      var run = state.need || state.empty || (state.few && state.interval);
      var verdict = el("div", { class: "callout " + (run ? "warn" : "good") });
      verdict.appendChild(el("p", { html: run ? "<strong>YOLO runs on this frame.</strong>" : "<strong>YOLO is skipped.</strong> The existing crops are reused, which is the fast path." }));
      out.appendChild(verdict);
      if (!run) return;
      var r = raw.get(), m = Math.min(matched.get(), r);
      var msg;
      if (m < 3 && r >= 3) {
        msg = "<strong>Tracking lost.</strong> Only " + m + " of 3 people matched, but at least 3 are visible, so the pipeline cannot tell who is who. It sets <code>requestSelection</code>, playback pauses, and you click everyone again.";
      } else if (m < 3) {
        msg = "<strong>Partial update.</strong> Only " + r + " people are visible, so someone may be hidden. The " + m + " matched people get new crops, and the rest are not processed until a later check finds them.";
      } else {
        msg = "<strong>All matched.</strong> Boxes update and every person gets a re-aimed virtual camera. ROI state resets, so each localizer runs once.";
      }
      out.appendChild(el("p", { html: msg, class: "note" }));
    }
    body.appendChild(left);
    body.appendChild(out);
    root.appendChild(body);
    update();
  }

  buildMap();
  buildWalk();
  buildYolo();
})();
