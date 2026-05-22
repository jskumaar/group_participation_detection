#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/quaternion.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cmath>
#include <filesystem>
#include <optional>

#include "vision/model_core.h"
#include "vision/gaze_bayes_mapper.h"

typedef struct yaw_pitch_roll {
    cv::Rect2f rect;
    int yaw;
    int pitch;
    int x;
    int y;
    int z;
} Pose;

typedef struct RawPose {
    cv::Quatf rotation;
    cv::Vec3f position;
} RawPose;

struct CamIntrinsics
{
    float focal_length_w;
    float focal_length_h;
    float fov_w;
    float fov_h;
};

struct TrackerConfig {
    float head_size_mm = 200.f;
    float nms_threshold = 0.3f;
    float confidence_threshold = 0.5f;
    float localizer_threshold = 0.5f;
    float roi_zoom = 1.25f;

    float velocity_decay = 0.2f;
    float iou_threshold = 0.15f;
    int num_people = 3;

    vision::GazeBayesConfig gaze_bayes;

    int yolo_check_interval = 30;
    float head_height_ratio = 0.13f;
    float yolorerun_threshold = 0.5f;
    int panorama_offset_px = 0;
};

class OPNetTracker {
    public:
        OPNetTracker();
        ~OPNetTracker() = default;
        std::optional<Pose> run(cv::Mat frame, int fov);
        std::vector<cv::Rect> run_yolo(const cv::Mat& frame);

        void setConfig(const TrackerConfig& config);
        TrackerConfig getConfig() const { return config_; }

        void yolo_updated() { needs_yolo_update = false; }
        float need_yolo_update() { return needs_yolo_update; }

    private:
        bool initialize();
        void prepare_input_image(cv::Mat &img);
        RawPose transform_to_world_pose(const cv::Quatf &face_rotation, const cv::Point2f& face_xy, float face_size);
        void yolo_scale(cv::Mat& img);
        std::vector<cv::Rect> yolo_unscale(std::vector<Reframer::DetectedPeople> &detections);
        std::optional<Pose> detect();
        cv::Mat grayscale;
        Ort::Env env{nullptr};
        Ort::MemoryInfo allocator_info{nullptr};
        std::optional<Localizer> localizer_;
        std::optional<PoseEstimator> poseestimator_;
        std::optional<L2CSEstimator> l2cs_estimator_;
        std::optional<Reframer> reframer_;
        std::array<cv::Mat,2> downsized_original_images_ = {};
        CamIntrinsics intrinsics_;
        float resizeScales;
        int padX;
        int padY;

        bool needs_yolo_update;
        TrackerConfig config_;
        cv::Mat yolo_pano_work_;
        cv::Mat yolo_net_input_;

        inline static constexpr int INPUT_IMG_WIDTH = 640;
        inline static constexpr int INPUT_IMG_HEIGHT = 640;

        std::optional<cv::Rect2f> last_roi_;
        std::optional<cv::Rect2f> last_localizer_roi_;
};
