"""Generate assets/js/refs.js: line numbers and verbatim snippets from one commit.

Usage: python tools/gen_refs.py <path to repo> [commit]
Every pattern must match exactly once (or set "nth"), so a renamed function fails
loudly here instead of leaving a stale link on the site.
"""
import json
import subprocess
import sys
from pathlib import Path

C = "src/c_impl/"
P = "src/python_client/"

# key: (file, text the line must contain, optional dict: snip=number of lines, nth=which match)
REFS = {
    "main": (C + "app/main.cpp", "int main(int argc"),
    "ctrl_ctor": (C + "app/app_controller.cpp", "AppController::AppController(QObject* parent)"),
    "tick_interval": (C + "app/app_controller.cpp", "tickTimer_.setInterval(25);"),
    "csv_open": (C + "app/app_controller.cpp", 'csvWriter_.open("gaze_analysis.csv");'),
    "create_stream": (C + "app/app_controller.cpp", "createGrpcStream() {"),
    "open_video": (C + "app/app_controller.cpp", "void AppController::openVideoPath"),
    "toggle": (C + "app/app_controller.cpp", "void AppController::togglePlayPause"),
    "seek": (C + "app/app_controller.cpp", "void AppController::seekFrame"),
    "on_box": (C + "app/app_controller.cpp", "void AppController::onBoxClicked"),
    "on_tick": (C + "app/app_controller.cpp", "void AppController::onTick"),
    "selection": (C + "app/app_controller.cpp", "void AppController::enterSelectionMode"),
    "process_tracking": (C + "app/app_controller.cpp", "void AppController::processTrackingFrame"),
    "publish_call": (C + "app/app_controller.cpp", "domain::PipelineFrameContext publishContext;", {"snip": 5}),
    "types": (C + "domain/types.h", "struct InteractionPair {", {"snip": 7}),
    "frame_result": (C + "domain/types.h", "struct VisionFrameResult"),
    "process_frame": (C + "pipelines/vision_pipeline.cpp", "domain::VisionFrameResult VisionPipeline::processFrame"),
    "yolo_cond": (C + "pipelines/vision_pipeline.cpp", "if (tracker_.need_yolo_update() || tracks_.empty()", {"snip": 2}),
    "update_valid": (C + "pipelines/vision_pipeline.cpp", "VisionPipeline::updateValidDetections("),
    "request_sel": (C + "pipelines/vision_pipeline.cpp", "out.requestSelection = true;", {"nth": 1}),
    "pose_loop": (C + "pipelines/vision_pipeline.cpp", "for (auto& track : tracks_) {", {"snip": 18}),
    "sort_call": (C + "pipelines/vision_pipeline.cpp", "gazes = object_tracker_.update(gazes);"),
    "infer_call": (C + "pipelines/vision_pipeline.cpp", "out.interactions = gaze_bayes_mapper_.infer"),
    "pano_offset": (C + "pipelines/vision_pipeline.cpp", "VisionPipeline::applyPanoramaOffset"),
    "aspect_check": (C + "pipelines/vision_pipeline.cpp", "if (panoFrame.cols != panoFrame.rows * 2) return out;", {"nth": 1}),
    "tracker_config": (C + "vision/opnet_tracker.h", "struct TrackerConfig {", {"snip": 18}),
    "pose_struct": (C + "vision/opnet_tracker.h", "typedef struct yaw_pitch_roll {", {"snip": 8}),
    "needs_yolo_decl": (C + "vision/opnet_tracker.h", "bool needs_yolo_update;"),
    "need_yolo_fn": (C + "vision/opnet_tracker.h", "float need_yolo_update()"),
    "models_path": (C + "vision/opnet_tracker.cpp", "fs::path exe_dir = fs::current_path()", {"snip": 1}),
    "opnet_run": (C + "vision/opnet_tracker.cpp", "std::optional<Pose> OPNetTracker::run"),
    "detect": (C + "vision/opnet_tracker.cpp", "std::optional<Pose> OPNetTracker::detect", {"snip": 18}),
    "kalpha": (C + "vision/opnet_tracker.cpp", "constexpr float kAlpha"),
    "conf_comment": (C + "vision/opnet_tracker.cpp", "// Scale L2CS contribution by confidence", {"snip": 10}),
    "q_combine": (C + "vision/opnet_tracker.cpp", "cv::Quatf q_gaze_standard ="),
    "pose_out": (C + "vision/opnet_tracker.cpp", "Pose out {", {"snip": 7}),
    "yolo_scale": (C + "vision/opnet_tracker.cpp", "void OPNetTracker::yolo_scale"),
    "run_yolo": (C + "vision/opnet_tracker.cpp", "OPNetTracker::run_yolo"),
    "prepare_input": (C + "vision/opnet_tracker.cpp", "void OPNetTracker::prepare_input_image"),
    "localizer_run": (C + "vision/model_core.cpp", "std::pair<float, cv::Rect2f> Localizer::run("),
    "pose_run": (C + "vision/model_core.cpp", "std::optional<PoseEstimator::Face> PoseEstimator::run("),
    "l2cs_run": (C + "vision/model_core.cpp", "std::optional<L2CSEstimator::HeadPose> L2CSEstimator::run"),
    "l2cs_decode": (C + "vision/model_core.cpp", "float L2CSEstimator::decode_angle_from_logits"),
    "l2cs_expected": (C + "vision/model_core.cpp", "return expected_bin * 4.f - 180.f;", {"snip": 1}),
    "reframer_run": (C + "vision/model_core.cpp", "std::vector<Reframer::DetectedPeople> Reframer::run"),
    "reframer_extract": (C + "vision/model_core.cpp", "void Reframer::extract_detections"),
    "sort_update": (C + "vision/sort_tracker.cpp", "std::vector<PanoViewer::gaze> Sort::update"),
    "id_plus_one": (C + "vision/sort_tracker.cpp", "g.personID = trackers_[match.x].m_id + 1;"),
    "reassign": (C + "vision/sort_tracker.cpp", "if (expected_num_people_ > 0) {"),
    "inject": (C + "vision/sort_tracker.cpp", "std::vector<Sort::Track> Sort::inject"),
    "inject_aim": (C + "vision/sort_tracker.cpp", "float center_x = det.x + det.width * 0.5f;", {"snip": 12}),
    "sort_track": (C + "vision/sort_tracker.h", "struct Track {", {"snip": 4}),
    "sort_ctor": (C + "vision/sort_tracker.h", "Sort(int max_age = 1000, double iou_threshold = 0.15);"),
    "kalman_predict": (C + "third_party/KalmanTracker.cpp", "StateType KalmanTracker::predict()"),
    "kf_count": (C + "vision/sort_tracker.cpp", "KalmanTracker::kf_count = 0;"),
    "remap": (C + "vision/360_image_process.cpp", "void PanoViewer::buildRemapTables"),
    "add_gaze": (C + "vision/360_image_process.cpp", "PanoViewer::gaze PanoViewer::addGaze"),
    "global_gaze": (C + "vision/360_image_process.cpp", "cv::Vec3f global_gaze_from_panorama", {"snip": 18}),
    "compute_fov": (C + "vision/360_image_process.cpp", "float PanoViewer::computeFOVForPersonBox"),
    "persp_to_equi": (C + "vision/360_image_process.cpp", "cv::Rect PanoViewer::convertPerspectiveRectToEquirectangular"),
    "seam": (C + "vision/360_image_process.cpp", "// seam spanning case: left as-is"),
    "face_roi": (C + "vision/360_image_process.h", "struct FaceRoiState {", {"snip": 4}),
    "gaze_struct": (C + "vision/360_image_process.h", "struct gaze{", {"snip": 7}),
    "bayes_cfg": (C + "vision/gaze_bayes_mapper.h", "struct GazeBayesConfig {", {"snip": 8}),
    "bayes_motion": (C + "vision/gaze_bayes_mapper.cpp", "float motionFromHeadRect("),
    "bayes_like": (C + "vision/gaze_bayes_mapper.cpp", "float anisotropicLikelihood("),
    "bayes_soft": (C + "vision/gaze_bayes_mapper.cpp", "void softmaxInPlace("),
    "bayes_infer": (C + "vision/gaze_bayes_mapper.cpp", "std::vector<domain::InteractionPair> GazeBayesMapper::infer("),
    "bayes_scores": (C + "vision/gaze_bayes_mapper.cpp", "//null target score and likelihood", {"snip": 18}),
    "bayes_early": (C + "vision/gaze_bayes_mapper.cpp", "if (gazes.size() < 2) return interactions;"),
    "proto_edge": (C + "proto/rapport.proto", "message InteractionEdge {", {"snip": 9}),
    "proto_update": (C + "proto/rapport.proto", "message PipelineUpdate {", {"snip": 8}),
    "proto_service": (C + "proto/rapport.proto", "service RapportStream {", {"snip": 3}),
    "svc_publish": (C + "io/grpc_pipeline_stream_service.cpp", "void RapportStreamImpl::publish(", {"snip": 13}),
    "svc_stream": (C + "io/grpc_pipeline_stream_service.cpp", "grpc::Status RapportStreamImpl::StreamPipelineUpdates("),
    "svc_map": (C + "io/grpc_pipeline_stream_service.cpp", "rapport::PipelineUpdate mapToPipelineUpdate("),
    "pub_start": (C + "io/grpc_pipeline_publisher.cpp", "bool GrpcPipelinePublisher::start()"),
    "csv_write": (C + "io/csv_gaze_writer.cpp", "void CsvGazeWriter::writeFrame("),
    "video_ts": (C + "media/video_source.cpp", "std::uint64_t VideoSource::currentTimestampNs()"),
    "video_read": (C + "media/video_source.cpp", "bool VideoSource::read("),
    "save_defaults": (C + "app/mainwindow.cpp", "void saveDefaults(const TrackerConfig& cfg)"),
    "load_defaults": (C + "app/mainwindow.cpp", "bool loadDefaultsInto(TrackerConfig& cfg)"),
    "show_settings": (C + "app/mainwindow.cpp", "void MainWindow::showSettings()"),
    "settings_collect": (C + "app/mainwindow.cpp", "auto collectConfig = [&]() {"),
    "video_click": (C + "app/videowidget.cpp", "void VideoWidget::mousePressEvent"),
    "gv_update": (C + "app/gazevisualizer.cpp", "void GazeVisualizer::updateData"),
    "cmake_toolchain": (C + "CMakeLists.txt", "# vcpkg toolchain for protobuf/gRPC"),
    "cmake_proto": (C + "CMakeLists.txt", "add_custom_command("),
    "cmake_sources": (C + "CMakeLists.txt", "set(SOURCES"),
    "cli_serve": (P + "rapport_cli.py", "def cmd_serve("),
    "cli_replay": (P + "rapport_cli.py", "def cmd_replay("),
    "cli_parser": (P + "rapport_cli.py", "def build_parser()"),
    "cli_guard": (P + "rapport_cli.py", "class MismatchDurationGuard"),
    "client_once": (P + "grpc_client.py", "def run_client_once("),
    "client_forever": (P + "grpc_client.py", "def serve_forever("),
    "client_ready": (P + "grpc_client.py", "wait_for_ready=True"),
    "client_idmap": (P + "grpc_client.py", "def parse_person_id_map("),
    "labels_pattern": (P + "extract_labels.py", "LABEL_PATTERN = re.compile"),
    "labels_accepted": (P + "extract_labels.py", "def accepted_targets(", {"snip": 9}),
    "labels_compare": (P + "extract_labels.py", "def compare_update("),
    "labels_logger": (P + "extract_labels.py", "class UniqueMismatchLogger"),
    "eval_class": (P + "evaluation.py", "class GazeEvaluator"),
    "eval_handle": (P + "evaluation.py", "def handle_update("),
    "eval_auroc": (P + "evaluation.py", "def _auroc("),
    "eval_edges": (P + "evaluation.py", "def _edge_metrics("),
    "eval_format": (P + "evaluation.py", "def format_summary("),
    "codegen": (P + "build_scripts/proto_codegen.py", "def main()"),
}


def main():
    repo = Path(sys.argv[1])
    commit = sys.argv[2] if len(sys.argv) > 2 else "HEAD"
    full = subprocess.check_output(["git", "-C", str(repo), "rev-parse", commit], text=True).strip()
    cache = {}
    out = {}
    for key, spec in REFS.items():
        path, needle = spec[0], spec[1]
        opts = spec[2] if len(spec) > 2 else {}
        if path not in cache:
            text = subprocess.check_output(["git", "-C", str(repo), "show", f"{full}:{path}"]).decode("utf-8-sig")
            cache[path] = text.replace("\r\n", "\n").split("\n")
        lines = cache[path]
        hits = [i for i, line in enumerate(lines) if needle in line]
        nth = opts.get("nth")
        if nth is None and len(hits) != 1:
            raise SystemExit(f"{key}: {len(hits)} matches for {needle!r} in {path}")
        if nth is not None and len(hits) <= nth:
            raise SystemExit(f"{key}: no match {nth} for {needle!r} in {path}")
        index = hits[nth or 0]
        entry = {"file": path, "line": index + 1}
        if opts.get("snip"):
            chunk = lines[index:index + opts["snip"]]
            pad = min((len(l) - len(l.lstrip()) for l in chunk if l.strip()), default=0)
            entry["snippet"] = "\n".join(l[pad:] for l in chunk)
        out[key] = entry

    target = Path(__file__).resolve().parent.parent / "assets" / "js" / "refs.js"
    target.write_text(
        "/* Generated by tools/gen_refs.py. Do not edit by hand. */\n"
        f"window.REPO = {json.dumps({'url': 'https://github.com/jskumaar/group_participation_detection', 'commit': full})};\n"
        f"window.REFS = {json.dumps(out, indent=0)};\n",
        encoding="utf-8",
    )
    print(f"wrote {len(out)} refs at {full[:7]}")


if __name__ == "__main__":
    main()
