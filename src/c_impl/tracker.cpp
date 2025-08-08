#include "tracker.h"

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
        allocator_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        fs::path exe_dir = fs::current_path().parent_path().parent_path();
        std::wstring model_path = (exe_dir / L"models" / L"head-localizer.onnx").wstring();
        localizer_.emplace(allocator_info, Ort::Session(env, model_path.c_str(), session_options));
        model_path = (exe_dir / L"models" / L"head-pose-0.3-big-quantized.onnx").wstring();
        poseestimator_.emplace(allocator_info, Ort::Session(env, model_path.c_str(), session_options));
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

rotation_output OPNetTracker::run(cv::Mat frame){
    prepare_input_image(frame);

    return detect();
}

rotation_output OPNetTracker::detect(){
    auto [p, rect] = localizer_->run(grayscale);
    auto face = poseestimator_->run(grayscale, rect);
    auto pose = image_to_world((*face).rotation);
    auto rot_mat = pose.toRotMat3x3(cv::QUAT_ASSUME_UNIT);

    const auto& mx = rot_mat.col(0);
    const auto& my = rot_mat.col(1);
    const auto& mz = rot_mat.col(2);

    const float yaw = std::atan2(mx(2), mx(0));
    const float pitch = -std::atan2(-mx(1), std::sqrt(mx(2)*mx(2)+mx(0)*mx(0)));
    // For the roll angle we recognize that the matrix entries in the second row contain cos(pitch)*cos(roll), and
    // cos(pitch)*sin(roll). Using atan2 eliminates the common pitch factor and we obtain the roll angle.
    const float roll = std::atan2(-mz(1), my(1));

    constexpr double rad2deg = 180/M_PI;

    return {(float)(yaw*rad2deg), (float)(pitch*rad2deg), (float)(roll*-rad2deg)};
}