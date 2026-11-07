#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/quaternion.hpp>
#include <onnxruntime_cxx_api.h>
#include <array>
#include <atomic>
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
        /** How many people can be processed at the same time. */
        static constexpr int kWorkerCount = 3;

        OPNetTracker();
        ~OPNetTracker() = default;

        /**
         * Runs the per-person models on one perspective crop. Safe to call from several
         * threads at once, as long as each thread uses a different worker_index.
         */
        std::optional<Pose> run(int worker_index, cv::Mat frame, int fov, FaceRoiState& roi_state);
        std::vector<cv::Rect> run_yolo(const cv::Mat& frame);

        void setConfig(const TrackerConfig& config);
        TrackerConfig getConfig() const { return config_; }

        void yolo_updated() { needs_yolo_update = false; }
        float need_yolo_update() { return needs_yolo_update; }

    private:
        /** The models and buffers that a thread writes to while processing one person. */
        struct Worker {
            std::optional<Localizer> localizer;
            std::optional<PoseEstimator> pose_estimator;
            std::optional<L2CSEstimator> l2cs_estimator;
            cv::Mat grayscale;
            std::array<cv::Mat, 2> downsized_images = {};
            CamIntrinsics intrinsics;
        };

        bool initialize();
        void prepare_input_image(Worker& worker, cv::Mat &img);
        RawPose transform_to_world_pose(const Worker& worker, const cv::Quatf &face_rotation, const cv::Point2f& face_xy, float face_size);
        void yolo_scale(cv::Mat& img);
        std::vector<cv::Rect> yolo_unscale(std::vector<Reframer::DetectedPeople> &detections);
        std::optional<Pose> detect(Worker& worker, const cv::Mat& frame_bgr, FaceRoiState& roi_state);
        Ort::Env env{nullptr};
        Ort::MemoryInfo allocator_info{nullptr};
        std::array<Worker, kWorkerCount> workers_;
        std::optional<Reframer> reframer_;
        float resizeScales;
        int padX;
        int padY;

        std::atomic<bool> needs_yolo_update{false};
        TrackerConfig config_;
        cv::Mat yolo_pano_work_;
        cv::Mat yolo_net_input_;

        inline static constexpr int INPUT_IMG_WIDTH = 640;
        inline static constexpr int INPUT_IMG_HEIGHT = 640;
};
