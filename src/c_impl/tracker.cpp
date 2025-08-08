#include "tracker.h"

#define M_PI 3.14159265358979323846


namespace fs = std::filesystem;

OPNetTracker::OPNetTracker(){
    if(!initialize()) {
        std::cerr << "Failed to initialize OPNetTracker." << std::endl;
        exit(1);
    }
    fs::path exe_dir = fs::current_path().parent_path().parent_path();
    std::string yolo_path = (exe_dir / "models" / "yolov5s.onnx").string();
    net = cv::dnn::readNet(yolo_path.c_str());
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

std::vector<cv::Mat> OPNetTracker::pre_process(const cv::Mat& input_image) {
    cv::Mat blob;
    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    return outputs;
}

// Postprocess outputs — no class name vector needed
std::vector<Detection> OPNetTracker::post_process(const cv::Mat& input_image, std::vector<cv::Mat>& outputs) {
    std::vector<Detection> results;
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    float* data = (float*)outputs[0].data;

    const int dimensions = 85;  // x, y, w, h, obj_conf, 80 class scores
    const int rows = 25200;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {
            float* classes_scores = data + 5;
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(cv::Mat(1, 80, CV_32FC1, classes_scores), 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD) {
                int id = class_id.x;
                if (id == 0) {  // Only keep people
                    float cx = data[0];
                    float cy = data[1];
                    float w = data[2];
                    float h = data[3];
                    int left = int((cx - 0.5 * w) * x_factor);
                    int top = int((cy - 0.5 * h) * y_factor);
                    int width = int(w * x_factor);
                    int height = int(h * y_factor);

                    results.push_back({ cv::Rect(left, top, width, height), confidence, id});
                }
            }
        }
        data += dimensions;
    }

    return results;
}

std::vector<cv::Rect> OPNetTracker::detect_people_in_frame(const cv::Mat &frame) {
    std::vector<cv::Mat> detections = pre_process(frame);
    std::vector<Detection> people = post_process(frame, detections);

    // Prepare for NMS
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    for (const auto &det : people) {
        boxes.push_back(det.box);
        confidences.push_back(det.confidence);
    }

    // Apply NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

    // Gather final boxes
    std::vector<cv::Rect> final_boxes;
    for (int idx : indices) {
        final_boxes.push_back(boxes[idx]);
    }

    std::cout << "Detected " << final_boxes.size() << " people." << std::endl;

    return final_boxes;
}