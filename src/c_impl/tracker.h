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
        std::vector<cv::Rect> run_yolo(cv::Mat frame);
    private:
        bool initialize();
        void prepare_input_image(cv::Mat &img);
        cv::Mat yolo_scale(cv::Mat& img);
        std::vector<cv::Rect> yolo_unscale(std::vector<Reframer::DetectedPeople> &detections);
        rotation_output detect();
        cv::Mat grayscale;
        Ort::Env env{nullptr};  // Keep environment alive
        Ort::MemoryInfo allocator_info{nullptr};
        std::optional<Localizer> localizer_;
        std::optional<PoseEstimator> poseestimator_;
        std::optional<Reframer> reframer_;
        std::optional<cv::Rect2f> last_roi;
        std::array<cv::Mat,2> downsized_original_images_ = {}; // Image pyramid

        float resizeScales;
        int padX;
        int padY;
        

        static constexpr float NMS_THRESHOLD = 0.5f;
        static constexpr float CONFIDENCE_THRESHOLD = 0.6f;
        inline static constexpr int INPUT_IMG_WIDTH = 640;
        inline static constexpr int INPUT_IMG_HEIGHT = 640;
};