/* Searchable tour of every source file in c_impl and python_client. */
(function () {
  "use strict";
  var el = UI.el;
  var C = "src/c_impl/", P = "src/python_client/";

  var FILES = [
    { dir: "c_impl/app", p: C + "app/main.cpp", s: "Program entry. Attaches to the parent console so qDebug output is visible, then starts Qt and MainWindow.", w: "You need startup behaviour such as a command line flag.", r: ["main"] },
    { dir: "c_impl/app", p: C + "app/app_controller.cpp", s: "The application flow: open video, selection, play and pause, seek, the timer, and fanning results out to the UI, CSV and gRPC. The header is app_controller.h.", w: "You change when frames are processed or where results go.", r: ["on_tick", "process_tracking", "toggle", "selection", "create_stream"] },
    { dir: "c_impl/app", p: C + "app/mainwindow.cpp", s: "Builds the window, connects signals, owns the Settings dialog, and reads and writes tracker_settings.ini.", w: "You add a setting or a control.", r: ["show_settings", "save_defaults", "load_defaults"] },
    { dir: "c_impl/app", p: C + "app/videowidget.cpp", s: "Paints the frame and boxes, and maps mouse clicks to person boxes during selection.", w: "You change overlays or click behaviour.", r: ["video_click"] },
    { dir: "c_impl/app", p: C + "app/gazevisualizer.cpp", s: "VTK 3D scene of head spheres and gaze arrows.", w: "You want a better debugging view of the geometry.", r: ["gv_update"] },
    { dir: "c_impl/domain", p: C + "domain/types.h", s: "Plain data shared across layers: InteractionPair, VisionFrameResult, PipelineFrameContext.", w: "You add data that must travel between the pipeline and outputs.", r: ["types", "frame_result"] },
    { dir: "c_impl/pipelines", p: C + "pipelines/vision_pipeline.cpp", s: "The per-frame vision logic: YOLO gating, matching boxes to people, the pose loop, Sort, and the gaze mapper.", w: "You change the order or logic of pipeline stages.", r: ["process_frame", "yolo_cond", "update_valid", "pose_loop", "infer_call"] },
    { dir: "c_impl/vision", p: C + "vision/opnet_tracker.cpp", s: "Loads the models and runs them per crop. detect() holds ROI tracking and the head and eye blend. The header holds TrackerConfig and Pose.", w: "You change how gaze is computed, or add a tunable.", r: ["detect", "kalpha", "pose_out", "models_path", "tracker_config"] },
    { dir: "c_impl/vision", p: C + "vision/model_core.cpp", s: "Thin ONNX Runtime wrappers: Localizer, PoseEstimator, L2CSEstimator and Reframer. Each owns its preprocessing and output decoding.", w: "You swap a model or change its input handling.", r: ["localizer_run", "pose_run", "l2cs_run", "l2cs_decode", "reframer_run"] },
    { dir: "c_impl/vision", p: C + "vision/360_image_process.cpp", s: "PanoViewer: the virtual camera, remap tables, FOV choice, converting boxes back to panorama pixels, and gaze conversion to world space.", w: "You touch projection or coordinate maths.", r: ["remap", "compute_fov", "add_gaze", "global_gaze", "persp_to_equi"] },
    { dir: "c_impl/vision", p: C + "vision/sort_tracker.cpp", s: "SORT tracking, display ID assignment, and Sort::inject, which aims a PanoViewer from a YOLO box.", w: "You work on ID stability or camera aiming.", r: ["sort_update", "reassign", "inject"] },
    { dir: "c_impl/vision", p: C + "vision/gaze_bayes_mapper.cpp", s: "The Bayesian model: motion, priors, likelihoods, posteriors, and edge output. The header holds GazeBayesConfig.", w: "You change how interactions are inferred.", r: ["bayes_infer", "bayes_like", "bayes_motion", "bayes_cfg"] },
    { dir: "c_impl/media", p: C + "media/video_source.cpp", s: "A small wrapper around cv::VideoCapture. The preferred frame size has no effect on video files.", w: "You add a live camera or another input.", r: ["video_read", "video_ts"] },
    { dir: "c_impl/io", p: C + "io/grpc_pipeline_stream_service.cpp", s: "The gRPC service: converts results to protobuf, holds the bounded queue, and streams it to the client.", w: "You add a stream field, or change queue behaviour.", r: ["svc_map", "svc_publish", "svc_stream"] },
    { dir: "c_impl/io", p: C + "io/grpc_pipeline_publisher.cpp", s: "Owns the gRPC server lifecycle: start and stop.", w: "You change how the server is hosted.", r: ["pub_start"] },
    { dir: "c_impl/io", p: C + "io/csv_gaze_writer.cpp", s: "Writes gaze_analysis.csv, with one row per person per frame.", w: "You want different columns in the CSV.", r: ["csv_write"] },
    { dir: "c_impl/proto", p: C + "proto/rapport.proto", s: "The contract between the two processes. Regenerate Python stubs after any change. C++ regenerates at build time.", w: "You change what is streamed.", r: ["proto_update", "proto_edge", "proto_service"] },
    { dir: "c_impl/third_party", p: C + "third_party/KalmanTracker.cpp", s: "Kalman filter for one box. Modified from the original SORT code to decay velocity while unmatched and to reset velocity after a seek.", w: "You tune motion prediction.", r: ["kalman_predict"] },
    { dir: "c_impl/third_party", p: C + "third_party/Hungarian.cpp", s: "Standard Hungarian algorithm, used to match trackers to detections. Rarely needs changing.", w: "Almost never.", r: [] },
    { dir: "c_impl", p: C + "CMakeLists.txt", s: "The build: hard-coded dependency paths, protobuf and gRPC code generation, and the source list. A new .cpp file must be added here.", w: "You add a source file or a dependency.", r: ["cmake_sources", "cmake_proto", "cmake_toolchain"] },
    { dir: "python_client", p: P + "rapport_cli.py", s: "Command line: serve runs the client, and replay prints mismatch events filtered by duration.", w: "You add a command or flag.", r: ["cli_parser", "cli_serve", "cli_replay"] },
    { dir: "python_client", p: P + "grpc_client.py", s: "Connects to the tracker, waits for it if needed, loops over updates, and writes reports when the stream ends.", w: "You change the client's flow or output files.", r: ["client_once", "client_forever", "client_idmap"] },
    { dir: "python_client", p: P + "extract_labels.py", s: "Parses labels, works out accepted targets at a time, finds mismatches, and logs them as events.", w: "You change scoring rules or the label format.", r: ["labels_pattern", "labels_accepted", "labels_compare", "labels_logger"] },
    { dir: "python_client", p: P + "evaluation.py", s: "All the metrics: accuracy, confusion, AUROC, average precision, Brier, thresholds, and the printed report.", w: "You add or change a metric.", r: ["eval_class", "eval_handle", "eval_edges", "eval_auroc", "eval_format"] },
    { dir: "python_client", p: P + "build_scripts/proto_codegen.py", s: "Regenerates the Python stubs from rapport.proto.", w: "You change the proto.", r: ["codegen"] },
    { dir: "python_client", p: P + "generated/rapport_pb2.py", s: "Generated protobuf messages, along with rapport_pb2_grpc.py. Committed to git. Do not edit them by hand.", w: "Never by hand: run proto_codegen.py.", r: [] },
    { dir: "python_client", p: P + "requirements.txt", s: "Python dependencies.", w: "You add a package.", r: [] }
  ];

  var NAMES = {
    main: "main", on_tick: "AppController::onTick", process_tracking: "AppController::processTrackingFrame", toggle: "AppController::togglePlayPause",
    selection: "AppController::enterSelectionMode", create_stream: "createGrpcStream", show_settings: "MainWindow::showSettings", save_defaults: "saveDefaults",
    load_defaults: "loadDefaultsInto", video_click: "VideoWidget::mousePressEvent", gv_update: "GazeVisualizer::updateData", types: "InteractionPair",
    frame_result: "VisionFrameResult", process_frame: "VisionPipeline::processFrame", yolo_cond: "the YOLO decision", update_valid: "VisionPipeline::updateValidDetections",
    pose_loop: "the per-person pose loop", infer_call: "the call to the gaze mapper", detect: "OPNetTracker::detect", kalpha: "the head and eye blend", pose_out: "building the Pose",
    models_path: "where models are loaded from", tracker_config: "TrackerConfig", localizer_run: "Localizer::run", pose_run: "PoseEstimator::run", l2cs_run: "L2CSEstimator::run",
    l2cs_decode: "decode_angle_from_logits", reframer_run: "Reframer::run", remap: "PanoViewer::buildRemapTables", compute_fov: "computeFOVForPersonBox", add_gaze: "PanoViewer::addGaze",
    global_gaze: "global_gaze_from_panorama", persp_to_equi: "convertPerspectiveRectToEquirectangular", sort_update: "Sort::update", reassign: "display ID assignment", inject: "Sort::inject",
    bayes_infer: "GazeBayesMapper::infer", bayes_like: "anisotropicLikelihood", bayes_motion: "motionFromHeadRect", bayes_cfg: "GazeBayesConfig", video_read: "VideoSource::read",
    video_ts: "VideoSource::currentTimestampNs", svc_map: "mapToPipelineUpdate", svc_publish: "RapportStreamImpl::publish", svc_stream: "RapportStreamImpl::StreamPipelineUpdates",
    pub_start: "GrpcPipelinePublisher::start", csv_write: "CsvGazeWriter::writeFrame", proto_update: "message PipelineUpdate", proto_edge: "message InteractionEdge", proto_service: "service RapportStream",
    kalman_predict: "KalmanTracker::predict", cmake_sources: "the source list", cmake_proto: "protobuf code generation", cmake_toolchain: "vcpkg setup", cli_parser: "build_parser",
    cli_serve: "cmd_serve", cli_replay: "cmd_replay", client_once: "run_client_once", client_forever: "serve_forever", client_idmap: "parse_person_id_map", labels_pattern: "LABEL_PATTERN",
    labels_accepted: "accepted_targets", labels_compare: "compare_update", labels_logger: "UniqueMismatchLogger", eval_class: "GazeEvaluator", eval_handle: "GazeEvaluator.handle_update",
    eval_edges: "_edge_metrics", eval_auroc: "_auroc", eval_format: "format_summary", codegen: "main"
  };

  function build() {
    var root = document.getElementById("demo-tree");
    var body = el("div", { class: "demo-body" });
    var search = el("input", { type: "search", placeholder: "Try: threshold, queue, ROI, label, yaw", "aria-label": "Search files" });
    body.appendChild(search);
    var grid = el("div", { class: "tree-grid", style: "margin-top:11px" });
    var tree = el("div", { class: "tree", role: "group", "aria-label": "Files" });
    var detail = el("div", { "aria-live": "polite" });
    grid.appendChild(tree); grid.appendChild(detail);
    body.appendChild(grid);
    root.appendChild(body);
    var current = FILES[6];

    function show(file) {
      current = file;
      detail.textContent = "";
      detail.appendChild(el("h4", { text: file.p.replace("src/", ""), style: "margin-top:2px;overflow-wrap:anywhere" }));
      detail.appendChild(el("p", { text: file.s, style: "font-size:.94rem" }));
      detail.appendChild(el("p", { html: "<strong>Change this when:</strong> ", style: "font-size:.94rem" }, [file.w]));
      if (file.r.length) {
        detail.appendChild(el("p", { class: "fig-label", text: "Key places", style: "margin-top:12px" }));
        var list = el("ul", { style: "margin-top:0" });
        file.r.forEach(function (k) {
          if (!NAMES[k]) throw new Error("unnamed ref " + k);
          list.appendChild(el("li", {}, [UI.refLink(k, NAMES[k])]));
        });
        detail.appendChild(list);
      }
      detail.appendChild(el("p", {}, [el("a", { href: UI.fileUrl(file.p), target: "_blank", rel: "noopener", text: "Open the file on GitHub" })]));
      renderList();
    }
    function renderList() {
      var q = search.value.trim().toLowerCase();
      tree.textContent = "";
      var dir = null, count = 0;
      FILES.forEach(function (f) {
        if (q && (f.p + " " + f.s + " " + f.w).toLowerCase().indexOf(q) < 0) return;
        count++;
        if (f.dir !== dir) { dir = f.dir; tree.appendChild(el("div", { class: "dir", text: dir + "/" })); }
        var b = el("button", { type: "button", text: f.p.split("/").pop(), "aria-current": String(f === current) });
        b.addEventListener("click", function () { show(f); });
        tree.appendChild(b);
      });
      if (!count) tree.appendChild(el("p", { class: "note", text: "No files match." }));
    }
    search.addEventListener("input", renderList);
    show(current);
  }

  build();
})();
