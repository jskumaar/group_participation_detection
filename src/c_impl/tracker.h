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


class OPNetTracker {
    public:
        OPNetTracker();
        ~OPNetTracker() = default;
        rotation_output run(cv::Mat frame);
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
};