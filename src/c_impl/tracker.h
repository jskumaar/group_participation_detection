#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <filesystem>
#include <optional>
#include "model_core.h"

typedef struct yaw_pitch_roll {
    float yaw;
    float pitch;
    float roll;
} rotation_output;

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};


class OPNetTracker {
    public:
        OPNetTracker();
        ~OPNetTracker() = default;
        rotation_output run(cv::Mat frame);
        std::vector<cv::Rect> detect_people_in_frame(const cv::Mat &frame);
    private:
        bool initialize();
        void prepare_input_image(cv::Mat &img);
        rotation_output detect();
        cv::Mat grayscale;
        Ort::Env env{nullptr};  // Keep environment alive
        Ort::MemoryInfo allocator_info{nullptr};
        std::optional<Localizer> localizer_;
        std::optional<PoseEstimator> poseestimator_;
        std::optional<cv::Rect2f> last_roi;
        std::array<cv::Mat,2> downsized_original_images_ = {}; // Image pyramid
        bool initialized_ = false;  // Track initialization state
        cv::dnn::Net net;
        std::vector<cv::Mat> pre_process(const cv::Mat& input_image);
        std::vector<Detection> post_process(const cv::Mat& input_image, std::vector<cv::Mat>& outputs);


        static constexpr float NMS_THRESHOLD = 0.45f;
        static constexpr float CONFIDENCE_THRESHOLD = 0.75f;
        static constexpr float INPUT_WIDTH = 640.0f;
        static constexpr float INPUT_HEIGHT = 640.0f;
        static constexpr float SCORE_THRESHOLD = 0.5f;
};