// if using mac, convert wstrings to strings. opposite if windows


#include "tracker.h"


// NEED FOR WINDOWS
#define M_PI 3.14159265358979323846

namespace fs = std::filesystem;

OPNetTracker::OPNetTracker(){
    if(!initialize()) {
        std::cerr << "Failed to initialize OPNetTracker." << std::endl;
        exit(1);
    }
}

bool OPNetTracker::initialize() {
    try{
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "opnet-tracker");
        Ort::SessionOptions session_options;
        session_options.SetInterOpNumThreads(2);
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        allocator_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        fs::path exe_dir = fs::current_path().parent_path().parent_path().parent_path();
        //MAC
        // std::string model_path = (exe_dir / L"models" / L"head-localizer.onnx").string();
        //WINDOWS 
        std::wstring model_path = (exe_dir / L"models" / L"head-localizer.onnx").wstring();
        localizer_.emplace(allocator_info, Ort::Session(env, model_path.c_str(), session_options));
        //MAC
        // model_path = (exe_dir / L"models" / L"head-pose-0.3-big-quantized.onnx").string();
        //WINDOWS
        model_path = (exe_dir / L"models" / L"head-pose-0.3-big-quantized.onnx").wstring();
        poseestimator_.emplace(allocator_info, Ort::Session(env, model_path.c_str(), session_options));
        //MAC
        // model_path = (exe_dir / L"models" / L"yolo11n.onnx").string();
        //WINDOWS
        model_path = (exe_dir / L"models" / L"yolo11n.onnx").wstring();
        reframer_.emplace(allocator_info, Ort::Session(env, model_path.c_str(), session_options), CONFIDENCE_THRESHOLD, NMS_THRESHOLD);
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

CamIntrinsics make_intrinsics(const cv::Mat& img, int fov)
{
    const int w = img.cols, h = img.rows;
    const double diag_fov = fov * M_PI / 180.;
    const double fov_w = 2.*atan(tan(diag_fov/2.)/sqrt(1. + h/(double)w * h/(double)w));
    const double fov_h = 2.*atan(tan(diag_fov/2.)/sqrt(1. + w/(double)h * w/(double)h));
    const double focal_length_w = 1. / tan(.5 * fov_w);
    const double focal_length_h = 1. / tan(.5 * fov_h);
    /*  a
      ______  <--- here is sensor area
      |    /
      |   /
    f |  /
      | /  2 x angle is the fov
      |/
        <--- here is the hole of the pinhole camera

      So, a / f = tan(fov / 2)
      => f = a/tan(fov/2)
      What is a?
      1 if we define f in terms of clip space where the image plane goes from -1 to 1. Because a is the half-width.
    */

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

cv::Rect make_crop_rect_for_aspect(const cv::Size &size, int aspect_w, int aspect_h)
{
    auto [w, h] = size;
    if ( w*aspect_h > aspect_w*h )
    {
        // Image is too wide
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

cv::Vec3f image_to_world(float x, float y, float size, float reference_size_in_mm, const cv::Size2i& image_size, const CamIntrinsics& intrinsics)
{

    const float head_size_vertical = 2.f*size;  // Size from the model is more like half the real vertical size of a human head.
    const float xpos = -(intrinsics.focal_length_w * image_size.width * 0.5f) / head_size_vertical * reference_size_in_mm;
    const float zpos = (x / image_size.width * 2.f - 1.f) * xpos / intrinsics.focal_length_w;
    const float ypos = (y / image_size.height * 2.f - 1.f) * xpos / intrinsics.focal_length_h;
    return {xpos, ypos, zpos};
}

cv::Quatf rotation_from_two_vectors(const cv::Vec3f &a, const cv::Vec3f &b)
{
    // |axis| = |a| * |b| * sin(alpha)
    const cv::Vec3f axis = a.cross(b);
    // dot = |a|*|b|*cos(alpha)
    const float dot = a.dot(b);
    const float len = cv::norm(axis);
    cv::Vec3f normed_axis = axis / len;
    float angle = std::atan2(len, dot);
    if (!(std::isfinite(normed_axis[0]) && std::isfinite(normed_axis[1]) && std::isfinite(normed_axis[2])))
    {
        angle = 0.f;
        normed_axis = cv::Vec3f{1.,0.,0.};
    }
    return cv::Quatf::createFromAngleAxis(angle, normed_axis);
}

cv::Quatf compute_rotation_correction(const cv::Point3f& p)
{
    return rotation_from_two_vectors(
        {-1.f,0.f,0.f}, p);
}

cv::Vec3f rotate(const cv::Quatf& q, const cv::Vec3f& v)
{
    cv::Matx33f R = q.toRotMat3x3();  // quaternion → rotation matrix
    return R * v;                     // apply rotation
}

RawPose OPNetTracker::transform_to_world_pose(const cv::Quatf &face_rotation, const cv::Point2f& face_xy, const float face_size)
{
    const cv::Vec3f face_world_pos = image_to_world(
        face_xy.x, face_xy.y, face_size, HEAD_SIZE_MM,
        grayscale.size(),
        intrinsics_);

    const cv::Quatf rot_correction = compute_rotation_correction(face_world_pos);

    cv::Quatf rot = rot_correction * image_to_world(face_rotation);

    // But this is in general not the location of the rotation joint in the neck.
    // So we need an extra offset. Which we determine by computing
    // z,y,z-pos = head_joint_loc + R_face * offset
    const cv::Vec3f local_offset = cv::Vec3f{-100, 0, 0};
    const cv::Vec3f offset = rotate(rot, local_offset);
    const cv::Vec3f pos = face_world_pos + offset;

    return { rot, pos };
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

Pose OPNetTracker::run(cv::Mat frame, int fov){
    prepare_input_image(frame);
    intrinsics_ = make_intrinsics(frame, fov);
    return detect();
}


Pose OPNetTracker::detect(){
    auto [p, rect] = localizer_->run(grayscale);
    auto face = poseestimator_->run(grayscale, rect);

    RawPose raw_pose = transform_to_world_pose((*face).rotation, (*face).center, (*face).size);

    auto rot_mat = raw_pose.rotation.toRotMat3x3(cv::QUAT_ASSUME_UNIT);

    const auto& mx = rot_mat.col(0);
    const auto& my = rot_mat.col(1);
    const auto& mz = rot_mat.col(2);

    const float yaw = std::atan2(mx(2), mx(0));
    const float pitch = -std::atan2(-mx(1), std::sqrt(mx(2)*mx(2)+mx(0)*mx(0)));
    // For the roll angle we recognize that the matrix entries in the second row contain cos(pitch)*cos(roll), and
    // cos(pitch)*sin(roll). Using atan2 eliminates the common pitch factor and we obtain the roll angle.
    const float roll = std::atan2(-mz(1), my(1));

    constexpr double rad2deg = 180/M_PI;

    return {(float)(yaw*rad2deg), (float)(pitch*rad2deg), (float)(roll*-rad2deg), (float)(-raw_pose.position[2]*0.1), (float)(raw_pose.position[1]*0.1), (float)(-raw_pose.position[0]*0.1), p};
}

cv::Mat OPNetTracker::yolo_scale(cv::Mat& img) {
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
    cv::Mat tempImg = cv::Mat::zeros(INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, CV_8UC3);
    padX = (INPUT_IMG_WIDTH - img.cols) / 2;
    padY = (INPUT_IMG_HEIGHT - img.rows) / 2;
    img.copyTo(tempImg(cv::Rect(padX, padY, img.cols, img.rows)));
    return tempImg;
}

std::vector<cv::Rect> OPNetTracker::yolo_unscale(std::vector<Reframer::DetectedPeople> &detections){
    std::vector<cv::Rect> boxes;
    for (auto& det : detections) {
        // Unscale the bounding box
        det.bbox.x = static_cast<int>((det.bbox.x - padX) * resizeScales);
        det.bbox.y = static_cast<int>((det.bbox.y - padY) * resizeScales);
        det.bbox.width = static_cast<int>(det.bbox.width * resizeScales);
        det.bbox.height = static_cast<int>(det.bbox.height * resizeScales);
        boxes.push_back(det.bbox);
    }
    return boxes;
}

std::vector<cv::Rect> OPNetTracker::run_yolo(cv::Mat frame){
	frame = frame.clone();
    frame = yolo_scale(frame);
    auto data = reframer_->run(frame);
    return yolo_unscale(data);
}
