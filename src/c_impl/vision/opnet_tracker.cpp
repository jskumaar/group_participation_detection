/**
 * @file opnet_tracker.cpp
 * @brief OPNet tracker implementation for head pose estimation and person detection
 */

 #include "vision/opnet_tracker.h"

 #include "vision/360_image_process.h"
 #include <QDebug>
 #include <onnxruntime_session_options_config_keys.h>
 
 #ifdef _WIN32
 #define M_PI 3.14159265358979323846
 #endif
 
 namespace fs = std::filesystem;
 
 namespace {
 
 // Lightweight models often run fastest with a small intra-op pool (2–4 threads).
 constexpr int kOrtIntraOpThreads = 4;
 constexpr int kOrtInterOpThreads = 1;
 
 Ort::SessionOptions makeCpuSessionOptions()
 {
     Ort::SessionOptions options;
     options.SetIntraOpNumThreads(kOrtIntraOpThreads);
     options.SetInterOpNumThreads(kOrtInterOpThreads);
     options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
     options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
     options.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning, "0");
     options.AddConfigEntry(kOrtSessionOptionsConfigAllowInterOpSpinning, "0");
     return options;
 }
 
 /** Yaw/pitch (deg) from PoseNet quaternion (col-0 convention, image frame). */
 void posenet_yaw_pitch_deg(const cv::Quatf& q, float& yaw_deg, float& pitch_deg)
 {
     const cv::Matx33f R = q.toRotMat3x3(cv::QUAT_ASSUME_UNIT);
     const auto mx = R.col(0);
     const float yaw = std::atan2(mx(2), mx(0));
     const float pitch = -std::atan2(
         -mx(1),
         std::sqrt(mx(2) * mx(2) + mx(0) * mx(0)));
     constexpr float kRadToDeg = 180.f / static_cast<float>(M_PI);
     yaw_deg = yaw * kRadToDeg;
     pitch_deg = pitch * kRadToDeg;
 }
 
 } // namespace
 
 OPNetTracker::OPNetTracker()
     : yolo_net_input_(INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, CV_8UC3, cv::Scalar(0))
 {
     if(!initialize()) {
         std::cerr << "Failed to initialize OPNetTracker." << std::endl;
         exit(1);
     }
 }
 
 bool OPNetTracker::initialize() {
     try{
         env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "opnet-tracker");
         Ort::SessionOptions session_options = makeCpuSessionOptions();
         allocator_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
         fs::path exe_dir = fs::current_path().parent_path().parent_path();
 
 #ifdef _WIN32
         std::wstring model_path = (exe_dir / L"models" / L"head-localizer.onnx").wstring();
         localizer_.emplace(allocator_info, Ort::Session(env, model_path.c_str(), session_options));
         model_path = (exe_dir / L"models" / L"head-pose-0.3-big-quantized.onnx").wstring();
         poseestimator_.emplace(allocator_info, Ort::Session(env, model_path.c_str(), session_options));
         model_path = (exe_dir / L"models" / L"L2CSNet_gaze360.onnx").wstring();
         l2cs_estimator_.emplace(allocator_info, Ort::Session(env, model_path.c_str(), session_options));
         model_path = (exe_dir / L"models" / L"yolo11n.onnx").wstring();
         reframer_.emplace(allocator_info, Ort::Session(env, model_path.c_str(), session_options), config_.confidence_threshold, config_.nms_threshold);
 #else
         std::string model_path = (exe_dir / "models" / "head-localizer.onnx").string();
         localizer_.emplace(allocator_info, Ort::Session(env, model_path.c_str(), session_options));
         model_path = (exe_dir / "models" / "head-pose-0.3-big-quantized.onnx").string();
         poseestimator_.emplace(allocator_info, Ort::Session(env, model_path.c_str(), session_options));
         model_path = (exe_dir / "models" / "L2CSNet_gaze360.onnx").string();
         l2cs_estimator_.emplace(allocator_info, Ort::Session(env, model_path.c_str(), session_options));
         model_path = (exe_dir / "models" / "yolo11n.onnx").string();
         reframer_.emplace(allocator_info, Ort::Session(env, model_path.c_str(), session_options), config_.confidence_threshold, config_.nms_threshold);
 #endif
     }
     catch (const Ort::Exception &e)
     {
         std::cerr << "Failed to initialize the neural network models. ONNX error message: "
             << e.what() << std::endl;
         return false;
     }
     catch (const std::exception &e)
     {
         std::cerr << "Failed to initialize the neural network models. Error message: " << e.what() << std::endl;
         return false;
     }
     return true;
 }
 
 void OPNetTracker::setConfig(const TrackerConfig& config) {
     config_ = config;
     if (reframer_) {
         reframer_->setThresholds(config_.confidence_threshold, config_.nms_threshold);
     }
 }
 
 CamIntrinsics make_intrinsics(const cv::Mat& img, int fov)
 {
     const int w = img.cols, h = img.rows;
     const double diag_fov = fov * M_PI / 180.;
     const double fov_w = 2.*atan(tan(diag_fov/2.)/sqrt(1. + h/(double)w * h/(double)w));
     const double fov_h = 2.*atan(tan(diag_fov/2.)/sqrt(1. + w/(double)h * w/(double)h));
     const double focal_length_w = 1. / tan(.5 * fov_w);
     const double focal_length_h = 1. / tan(.5 * fov_h);
 
     return {
         (float)focal_length_w,
         (float)focal_length_h,
         (float)fov_w,
         (float)fov_h
     };
 }
 
 cv::Quatf image_to_world(cv::Quatf q)
 {
     std::swap(q[1], q[3]);
     q[1] = -q[1];
     q[2] = -q[2];
     q[3] = -q[3];
     return q;
 }
 
 cv::Quatf rotation_from_two_vectors(const cv::Vec3f &a, const cv::Vec3f &b)
 {
     const cv::Vec3f axis = a.cross(b);
     const float dot = a.dot(b);
     const float len = static_cast<float>(cv::norm(axis));
     cv::Vec3f normed_axis = axis / len;
     float angle = std::atan2(len, dot);
     if (!(std::isfinite(normed_axis[0]) && std::isfinite(normed_axis[1]) && std::isfinite(normed_axis[2])))
     {
         angle = 0.f;
         normed_axis = cv::Vec3f{1.f, 0.f, 0.f};
     }
     return cv::Quatf::createFromAngleAxis(angle, normed_axis);
 }
 
 cv::Quatf compute_rotation_correction(const cv::Point3f& p)
 {
     return rotation_from_two_vectors({-1.f, 0.f, 0.f}, p);
 }
 
 cv::Vec3f image_to_world(float x, float y, float size, float reference_size_in_mm, const cv::Size2i& image_size, const CamIntrinsics& intrinsics)
 {
     const float head_size_vertical = 2.f*size;
     const float xpos = -(intrinsics.focal_length_w * image_size.width * 0.5f) / head_size_vertical * reference_size_in_mm;
     const float zpos = (x / image_size.width * 2.f - 1.f) * xpos / intrinsics.focal_length_w;
     const float ypos = (y / image_size.height * 2.f - 1.f) * xpos / intrinsics.focal_length_h;
     return {xpos, ypos, zpos};
 }
 
 RawPose OPNetTracker::transform_to_world_pose(const cv::Quatf &face_rotation, const cv::Point2f& face_xy, float face_size)
 {
     const cv::Vec3f face_world_pos = image_to_world(
         face_xy.x, face_xy.y, face_size, config_.head_size_mm,
         grayscale.size(),
         intrinsics_);
 
     const cv::Quatf rot_correction = compute_rotation_correction(face_world_pos);
     cv::Quatf rot = rot_correction * image_to_world(face_rotation);
 
     return {rot, face_world_pos};
 }
 
 cv::Rect make_crop_rect_for_aspect(const cv::Size &size, int aspect_w, int aspect_h)
 {
     auto [w, h] = size;
     if ( w*aspect_h > aspect_w*h )
     {
         const int new_w = (aspect_w*h)/aspect_h;
         return cv::Rect((w - new_w)/2, 0, new_w, h);
     }
     else
     {
         const int new_h = (aspect_h*w)/aspect_w;
         return cv::Rect(0, (h - new_h)/2, w, new_h);
     }
 }
 
 cv::Rect make_crop_rect_multiple_of(const cv::Size &size, int multiple)
 {
     const int new_w = (size.width / multiple) * multiple;
     const int new_h = (size.height / multiple) * multiple;
     return cv::Rect(
         (size.width-new_w)/2,
         (size.height-new_h)/2,
         new_w,
         new_h
     );
 }
 
 cv::Rect2f expand_roi_centered(const cv::Rect2f &box, float scale)
 {
     if (scale <= 1.f)
         return box;
     const float cx = box.x + 0.5f * box.width;
     const float cy = box.y + 0.5f * box.height;
     const float w = box.width * scale;
     const float h = box.height * scale;
     return {cx - 0.5f * w, cy - 0.5f * h, w, h};
 }
 
 void OPNetTracker::prepare_input_image(cv::Mat &img){
     if (img.rows*4 != img.cols*3)
     {
         img = img(make_crop_rect_for_aspect(img.size(), 4, 3));
     }
 
     img = img(make_crop_rect_multiple_of(img.size(), 4));
 
     if (img.cols > 640)
     {
         cv::pyrDown(img, downsized_original_images_[0]);
         img = downsized_original_images_[0];
     }
     if (img.cols > 640)
     {
         cv::pyrDown(img, downsized_original_images_[1]);
         img = downsized_original_images_[1];
     }
     cv::cvtColor(img, grayscale, cv::COLOR_BGR2GRAY);
 }
 
 std::optional<Pose> OPNetTracker::run(cv::Mat frame, int fov){
     prepare_input_image(frame);
     intrinsics_ = make_intrinsics(frame, fov);
     return detect();
 }
 std::optional<Pose> OPNetTracker::detect(){
    auto [p, rect] = localizer_->run(grayscale);
    if(p < config_.localizer_threshold){
        needs_yolo_update = true;
        return std::nullopt;
    }

    const cv::Rect localizer_box(
        static_cast<int>(rect.x),
        static_cast<int>(rect.y),
        static_cast<int>(rect.width),
        static_cast<int>(rect.height));
    auto face = poseestimator_->run(grayscale, localizer_box);
    if (!face.has_value()) {
        return std::nullopt;
    }

    const cv::Rect2f l2cs_roi = expand_roi_centered(face->box, config_.roi_zoom);
    auto gaze = l2cs_estimator_->run(grayscale, l2cs_roi);
    if (!gaze.has_value()) {
        return std::nullopt;
    }

    const RawPose raw_pose = transform_to_world_pose(face->rotation, face->center, face->size);

    float head_yaw_deg = 0.f;
    float head_pitch_deg = 0.f;
    
    // FIX: Extract angles from the world-corrected raw_pose.rotation instead of the unmapped face->rotation
    posenet_yaw_pitch_deg(raw_pose.rotation, head_yaw_deg, head_pitch_deg);
    
    // --- COMPENSATION DAMPENING SCALAR ADDED HERE ---
    // Transform absolute gaze back into the local head coordinate space to prevent 
    // double-rotations when combined with head pose below.
    constexpr float kAlpha = 0.4f; // Compensation dampening scalar (adjust between 0.35 and 0.5)
    const float compensated_gaze_yaw_deg = gaze->yaw_deg - (kAlpha * head_yaw_deg);
    const float compensated_gaze_pitch_deg = gaze->pitch_deg - (kAlpha * head_pitch_deg);

    constexpr float kDegToRad = static_cast<float>(M_PI / 180.0);
    const float l2cs_yaw_rad = compensated_gaze_yaw_deg * kDegToRad;
    const float l2cs_pitch_rad = compensated_gaze_pitch_deg * kDegToRad;

    // --- UPDATED COMBINATION LOGIC ---
    // Align coordinate systems before combining:
    // Extract the head angles and rebuild the quaternion in the unified Z-forward space.
    cv::Vec3f head_dir = vision::dir_from_yaw_pitch(head_yaw_deg * kDegToRad, head_pitch_deg * kDegToRad);
    cv::Quatf q_head_standard = rotation_from_two_vectors(cv::Vec3f(0.f, 0.f, 1.f), head_dir);

    // Build the eye quaternion in the same Z-forward space using the compensated local gaze
    cv::Vec3f eye_dir = vision::dir_from_yaw_pitch(l2cs_yaw_rad, l2cs_pitch_rad);
    cv::Quatf q_eye_standard = rotation_from_two_vectors(cv::Vec3f(0.f, 0.f, 1.f), eye_dir);

    
// Scale L2CS contribution by confidence: 0 weight at <=12%, full weight at >=25%.
     constexpr float kL2csConfMinPct = 20.f;
     constexpr float kL2csConfMaxPct = 40.f;
     const float conf_pct = gaze->confidence * 100.f;
     
     // std::clamp keeps the value strictly between 0.0 and 1.0
     const float eye_weight = std::clamp(
         (conf_pct - kL2csConfMinPct) / (kL2csConfMaxPct - kL2csConfMinPct), 
         0.f, 
         1.f
     );
    const cv::Quatf q_identity{1.f, 0.f, 0.f, 0.f};
    const cv::Quatf q_eye_weighted = cv::Quatf::nlerp(
        q_identity, q_eye_standard, eye_weight, cv::QUAT_ASSUME_UNIT);

    // Combine them in the standard Z-forward coordinate system
    cv::Quatf q_gaze_standard = q_head_standard * q_eye_weighted;

    float combined_yaw_deg = 0.f;
    float combined_pitch_deg = 0.f;
    vision::yaw_pitch_deg_from_rot_mat(
        q_gaze_standard.toRotMat3x3(cv::QUAT_ASSUME_UNIT),
        combined_yaw_deg,
        combined_pitch_deg);
    // ---------------------------------

    qDebug() << "[OPNet] poseestimator(col0) yaw_deg=" << head_yaw_deg
             << "pitch_deg=" << head_pitch_deg
             << "l2cs raw yaw_deg=" << gaze->yaw_deg
             << "l2cs comp yaw_deg=" << compensated_gaze_yaw_deg
             << "confidence=" << gaze->confidence
             << "eye_weight=" << eye_weight
             << "combined yaw_deg=" << combined_yaw_deg
             << "pitch_deg=" << combined_pitch_deg;

    Pose out {
        face->box,
        static_cast<int>(combined_yaw_deg),
        static_cast<int>(combined_pitch_deg),
        static_cast<int>(-raw_pose.position[2] * 0.1f),
        static_cast<int>(raw_pose.position[1] * 0.1f),
        static_cast<int>(-raw_pose.position[0] * 0.1f)};
    return out;
}
 
 void OPNetTracker::yolo_scale(cv::Mat& img) {
     cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
 
     if (img.cols >= img.rows)
     {
         resizeScales = img.cols / (float)INPUT_IMG_WIDTH;
         cv::resize(img, img, cv::Size(INPUT_IMG_WIDTH, int(img.rows / resizeScales)));
     }
     else
     {
         resizeScales = img.rows / (float)INPUT_IMG_HEIGHT;
         cv::resize(img, img, cv::Size(int(img.cols / resizeScales), INPUT_IMG_HEIGHT));
     }
     yolo_net_input_.setTo(0);
     padX = (INPUT_IMG_WIDTH - img.cols) / 2;
     padY = (INPUT_IMG_HEIGHT - img.rows) / 2;
     img.copyTo(yolo_net_input_(cv::Rect(padX, padY, img.cols, img.rows)));
 }
 
 std::vector<cv::Rect> OPNetTracker::yolo_unscale(std::vector<Reframer::DetectedPeople> &detections){
     std::vector<cv::Rect> boxes;
     for (auto& det : detections) {
         det.bbox.x = static_cast<int>((det.bbox.x - padX) * resizeScales);
         det.bbox.y = static_cast<int>((det.bbox.y - padY) * resizeScales);
         det.bbox.width = static_cast<int>(det.bbox.width * resizeScales);
         det.bbox.height = static_cast<int>(det.bbox.height * resizeScales);
         boxes.push_back(det.bbox);
     }
     return boxes;
 }
 
 std::vector<cv::Rect> OPNetTracker::run_yolo(const cv::Mat& frame){
     if (yolo_pano_work_.rows != frame.rows || yolo_pano_work_.cols != frame.cols || yolo_pano_work_.type() != frame.type())
         yolo_pano_work_.create(frame.rows, frame.cols, frame.type());
     frame.copyTo(yolo_pano_work_);
     yolo_scale(yolo_pano_work_);
     auto data = reframer_->run(yolo_net_input_);
     return yolo_unscale(data);
 }