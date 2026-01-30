#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <filesystem>
#include <optional>
#include "model_core.h"


typedef struct yaw_pitch_roll {
    cv::Rect2f rect;
    int yaw;
    int pitch;
    int roll;
    int x;
    int y;
    int z;
	float confidence;
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
    
    float velocity_decay = 0.2f;
    float iou_threshold = 0.15f;
    int num_people = 3;
    float max_angle_deg = 20.0f;
    
    float eye_boost_knee = 13.0f;
    float eye_boost_steepness = 0.10f;
    float eye_boost_max = 0.7f;
    
    int yolo_check_interval = 30;
    float head_height_ratio = 0.13f;
    float yolorerun_threshold = 0.5f;
};


class OPNetTracker {
    public:
        OPNetTracker();
        ~OPNetTracker() = default;
        std::optional<Pose> run(cv::Mat frame, int fov);
        std::vector<cv::Rect> run_yolo(cv::Mat frame);
        
        void setConfig(const TrackerConfig& config);
        TrackerConfig getConfig() const { return config_; }

        // Expose the reframer/localizer input height for external use
        static int getInputHeight() { return INPUT_IMG_HEIGHT; }
        void yolo_updated() { needs_yolo_update = false; }
        float need_yolo_update() { return needs_yolo_update; }

    private:
        bool initialize();
        void prepare_input_image(cv::Mat &img);
        RawPose transform_to_world_pose(const cv::Quatf &face_rotation, const cv::Point2f& face_xy, const float face_size);
        cv::Mat yolo_scale(cv::Mat& img);
        std::vector<cv::Rect> yolo_unscale(std::vector<Reframer::DetectedPeople> &detections);
        std::optional<Pose> detect();
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

        bool needs_yolo_update;



        TrackerConfig config_;

        // static constexpr float HEAD_SIZE_MM = 200.f;
        // static constexpr float NMS_THRESHOLD = 0.3f;
        // static constexpr float CONFIDENCE_THRESHOLD = 0.5f;
        inline static constexpr int INPUT_IMG_WIDTH = 640;
        inline static constexpr int INPUT_IMG_HEIGHT = 640;
};