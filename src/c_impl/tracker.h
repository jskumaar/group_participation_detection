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
    float x;
    float y;
    float z;
} Pose;

typedef struct RawPose {
    cv::Quatf rotation;
    cv::Vec3f position;
} RawPose;

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

struct CamIntrinsics
{
    float focal_length_w;
    float focal_length_h;
    float fov_w;
    float fov_h;
};


class OPNetTracker {
    public:
        OPNetTracker();
        ~OPNetTracker() = default;
        Pose run(cv::Mat frame, int fov);
        std::vector<cv::Rect> run_yolo(cv::Mat frame);
    private:
        bool initialize();
        void prepare_input_image(cv::Mat &img);
        RawPose transform_to_world_pose(const cv::Quatf &face_rotation, const cv::Point2f& face_xy, const float face_size);
        cv::Mat yolo_scale(cv::Mat& img);
        std::vector<cv::Rect> yolo_unscale(std::vector<Reframer::DetectedPeople> &detections);
        Pose detect();
        cv::Mat grayscale;
        Ort::Env env{nullptr};  // Keep environment alive
        Ort::MemoryInfo allocator_info{nullptr};
        std::optional<Localizer> localizer_;
        std::optional<PoseEstimator> poseestimator_;
        std::optional<Reframer> reframer_;
        std::optional<cv::Rect2f> last_roi;
        std::array<cv::Mat,2> downsized_original_images_ = {}; // Image pyramid
        CamIntrinsics intrinsics_;
        float resizeScales;
        int padX;
        int padY;


        static constexpr float HEAD_SIZE_MM = 200.f;
        static constexpr float NMS_THRESHOLD = 0.5f;
        static constexpr float CONFIDENCE_THRESHOLD = 0.5f;
        inline static constexpr int INPUT_IMG_WIDTH = 640;
        inline static constexpr int INPUT_IMG_HEIGHT = 640;
};