#include "tracker.h"


namespace fs = std::filesystem;


bool OpNetTracker::initialize() {
    try{
        env(ORT_LOGGING_LEVEL_ERROR, "opnet-tracker");
        Ort::SessionOptions session_options;
        session_options.SetInterOpNumThreads(2);
        session_options.SetIntraOpNumThreads(1);
        allocator_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        fs::path exe_dir = fs::current_path().parent_path().parent_path();
        std::string model_path = (exe_dir / "models" / "head-localizer.onnx").string();
        localizer = Ort::Session(env, model_path.c_str(), session_options);
        model_path = (exe_dir / "models" / "head-pose-0.3-big-quantized.onnx").string();
        posenet = Ort::Session(env, model_path.c_str(), session_options);
    }
    catch (const Ort::Exception& e) {
        std::cerr << "Ort::Exception: " << e.what() << std::endl;
        return false;
    }
    return true;
}




float to_degrees(float rad) {
    return rad * 180.0f / static_cast<float>(CV_PI);
}

int main() {
    env(ORT_LOGGING_LEVEL_WARNING, "opnet-tracker");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::MemoryInfo allocator_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    fs::path exe_dir = fs::current_path().parent_path().parent_path();
    std::wstring model_path = (exe_dir / L"models" / L"head-localizer.onnx").wstring();
    Ort::Session localizer(env, model_path.c_str(), session_options);
    model_path = (exe_dir / L"models" / L"head-pose-0.3-big-quantized.onnx").wstring();
    Ort::Session posenet(env, model_path.c_str(), session_options);

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return 1;
    }

    while (true) {
        cv::Mat frame, gray;
        cap >> frame;
        if (frame.empty()) break;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::resize(gray, gray, cv::Size(288, 224));
        gray.convertTo(gray, CV_32F, 1.0 / 255);

        std::array<int64_t, 4> loc_shape{1, 1, 224, 288};
        std::vector<float> loc_input(gray.begin<float>(), gray.end<float>());
        Ort::Value loc_tensor = Ort::Value::CreateTensor<float>(
            allocator_info, loc_input.data(), loc_input.size(), loc_shape.data(), loc_shape.size());

        Ort::AllocatorWithDefaultOptions allocator;
        auto loc_input_names = localizer.GetInputNames();
        auto loc_output_names = localizer.GetOutputNames();
        const char* loc_input_name = loc_input_names[0].c_str();
        const char* loc_output_name = loc_output_names[0].c_str();
        auto loc_output = localizer.Run(Ort::RunOptions{nullptr},
                                        &loc_input_name, &loc_tensor, 1,
                                        &loc_output_name, 1);
        float* loc_data = loc_output[0].GetTensorMutableData<float>();
        float x = loc_data[0], y = loc_data[1], size = loc_data[2], conf = loc_data[3];
        
        // Debug output
        std::cout << "Localizer output - x: " << x << ", y: " << y << ", size: " << size << ", conf: " << conf << std::endl;

        if (conf < 0.3f) {  // Temporarily lowered threshold for debugging
            cv::imshow("OPNet Tracker", frame);
            if (cv::waitKey(1) == 27) break;
            continue;
        }

        // Scale coordinates from normalized to pixel coordinates
        // Assuming the model outputs normalized coordinates (0-1 range)
        float scaled_x = x * frame.cols;
        float scaled_y = y * frame.rows;
        float scaled_size = abs(size) * std::min(frame.cols, frame.rows); // Take absolute value and scale
        
        std::cout << "Scaled coordinates - x: " << scaled_x << ", y: " << scaled_y << ", size: " << scaled_size << std::endl;

        cv::Rect roi((int)(scaled_x - scaled_size / 2), (int)(scaled_y - scaled_size / 2), (int)scaled_size, (int)scaled_size);
        roi &= cv::Rect(0, 0, frame.cols, frame.rows);
        
        // Safety check: ensure ROI is valid and not empty
        if (roi.width <= 0 || roi.height <= 0 || roi.area() == 0) {
            std::cerr << "Invalid ROI detected: " << roi << std::endl;
            cv::imshow("OPNet Tracker", frame);
            if (cv::waitKey(1) == 27) break;
            continue;
        }
        
        cv::Mat crop = frame(roi).clone();
        
        // Additional safety check: ensure crop is not empty
        if (crop.empty()) {
            std::cerr << "Empty crop detected" << std::endl;
            cv::imshow("OPNet Tracker", frame);
            if (cv::waitKey(1) == 27) break;
            continue;
        }
        
        cv::resize(crop, crop, cv::Size(112, 112));
        cv::cvtColor(crop, crop, cv::COLOR_BGR2RGB);
        crop.convertTo(crop, CV_32F, 1.0 / 255);

        std::array<int64_t, 4> pose_shape{1, 3, 112, 112};
        std::vector<float> pose_input;
        for (int c = 0; c < 3; ++c)
            for (int i = 0; i < 112; ++i)
                for (int j = 0; j < 112; ++j)
                    pose_input.push_back(crop.at<cv::Vec3f>(i, j)[c]);

        Ort::Value pose_tensor = Ort::Value::CreateTensor<float>(
            allocator_info, pose_input.data(), pose_input.size(), pose_shape.data(), pose_shape.size());

        auto pose_input_names = posenet.GetInputNames();
        auto pose_output_names = posenet.GetOutputNames();
        const char* pose_input_name = pose_input_names[0].c_str();
        const char* pose_output_name = pose_output_names[0].c_str();

        auto pose_output = posenet.Run(Ort::RunOptions{nullptr},
                                       &pose_input_name, &pose_tensor, 1,
                                       &pose_output_name, 1);

        float* out = pose_output[0].GetTensorMutableData<float>();
        float yaw = to_degrees(out[0]);
        float pitch = to_degrees(out[1]);
        float roll = to_degrees(out[2]);

        std::string label = cv::format("Y: %.1f, P: %.1f, R: %.1f", yaw, pitch, roll);
        cv::putText(frame, label, {20, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 0}, 2);
        cv::rectangle(frame, roi, {255, 0, 0}, 2);

        cv::imshow("OPNet Tracker", frame);
        if (cv::waitKey(1) == 27) break;  // ESC to quit
    }

    return 0;
}
