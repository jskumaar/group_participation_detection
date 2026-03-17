#include "pipelines/vision_pipeline.h"

#include <algorithm>
#include <cmath>

namespace pipelines {

static float eyeBoost(const TrackerConfig& config, float yaw_deg) {
    float y = std::abs(yaw_deg);
    float max_boost = config.eye_boost_max;
    float knee = config.eye_boost_knee;
    float steepness = config.eye_boost_steepness;
    float extra = max_boost * (1.0f - 1.0f / (1.0f + std::exp((y - knee) * steepness)));
    return 1.0f + extra;
}

VisionPipeline::VisionPipeline()
    : tracker_(),
      object_tracker_(),
      pano_viewer_() {}

void VisionPipeline::setConfig(const TrackerConfig& cfg) {
    tracker_.setConfig(cfg);
    object_tracker_.setIOUThreshold(cfg.iou_threshold);
    object_tracker_.setNumPeople(cfg.num_people);
    KalmanTracker::decay_velocity_factor = cfg.velocity_decay;
    pano_viewer_.setMaxAngle(cfg.max_angle_deg);
}

void VisionPipeline::setExpectedPeople(int n) {
    object_tracker_.setNumPeople(n);
}

cv::Mat VisionPipeline::applyPanoramaOffset(const cv::Mat& panoFrame) const {
    if (panorama_offset_ == 0) return panoFrame;
    int w = panoFrame.cols;
    int h = panoFrame.rows;
    int shift = panorama_offset_ % w;
    if (shift < 0) shift += w;
    if (shift == 0) return panoFrame;

    cv::Mat result;
    cv::Mat left_part = panoFrame(cv::Rect(w - shift, 0, shift, h));
    cv::Mat right_part = panoFrame(cv::Rect(0, 0, w - shift, h));
    cv::hconcat(left_part, right_part, result);
    return result;
}

domain::VisionFrameResult VisionPipeline::prepareSelection(const cv::Mat& panoFrame) {
    domain::VisionFrameResult out;

    if (panoFrame.empty()) {
        out.statusText = "Empty frame.";
        return out;
    }
    if (panoFrame.cols != panoFrame.rows * 2) {
        out.statusText = "Panorama frame is not a valid size.";
        return out;
    }

    cv::Mat shifted = applyPanoramaOffset(panoFrame);
    std::vector<cv::Rect> raw_people = tracker_.run_yolo(shifted);

    lastYoloDetections_ = raw_people;
    out.yoloDetections = raw_people;
    out.yoloActive = true;
    out.requestSelection = true;
    out.statusText = "Selection required.";
    return out;
}

void VisionPipeline::setSelectedDetections(const std::vector<cv::Rect>& selected, int panoRows, int panoCols) {
    validYoloDetections_ = selected;
    lastYoloDetections_ = selected;
    if (panoRows > 0 && panoCols > 0) {
        tracks_ = object_tracker_.inject(validYoloDetections_, panoRows, panoCols, tracker_.getConfig().head_height_ratio);
    } else {
        tracks_.clear();
    }
}

std::pair<std::vector<cv::Rect>, std::vector<cv::Rect>> VisionPipeline::updateValidDetections(
    const std::vector<cv::Rect>& newDetections,
    const std::vector<cv::Rect>& previousValid) const {

    std::vector<cv::Rect> injected;
    std::vector<cv::Rect> nextValid = previousValid;

    std::vector<bool> usedNew(newDetections.size(), false);
    float threshold = tracker_.getConfig().yolorerun_threshold;

    for (size_t i = 0; i < nextValid.size(); ++i) {
        int bestIdx = -1;
        float bestIoU = -1.0f;

        for (size_t j = 0; j < newDetections.size(); ++j) {
            if (usedNew[j]) continue;

            cv::Rect intersect = nextValid[i] & newDetections[j];
            float intersectArea = static_cast<float>(intersect.area());
            float unionArea = static_cast<float>(nextValid[i].area() + newDetections[j].area() - intersectArea);
            if (unionArea <= 0) continue;

            float iou = intersectArea / unionArea;
            if (iou > threshold && iou > bestIoU) {
                bestIoU = iou;
                bestIdx = static_cast<int>(j);
            }
        }

        if (bestIdx != -1) {
            nextValid[i] = newDetections[bestIdx];
            injected.push_back(newDetections[bestIdx]);
            usedNew[bestIdx] = true;
        }
    }

    return {injected, nextValid};
}

domain::VisionFrameResult VisionPipeline::processFrame(const cv::Mat& panoFrame, bool justSeeked, bool enableTracking) {
    domain::VisionFrameResult out;
    if (panoFrame.empty()) return out;
    if (panoFrame.cols != panoFrame.rows * 2) {
        out.statusText = "Panorama frame is not a valid size.";
        return out;
    }

    cv::Mat shifted = applyPanoramaOffset(panoFrame);

    yoloFrameCount_++;
    bool interval_check = (yoloFrameCount_ % (int)tracker_.getConfig().yolo_check_interval == 0);
    if (interval_check) yoloFrameCount_ = 0;

    // Always expose most recent detections for overlays/debug
    out.yoloDetections = lastYoloDetections_;

    // YOLO refresh logic
    if (tracker_.need_yolo_update() || tracks_.empty() ||
        (tracks_.size() < (size_t)tracker_.getConfig().num_people && interval_check)) {
        std::vector<cv::Rect> raw_people = tracker_.run_yolo(shifted);
        out.yoloActive = true;

        std::vector<cv::Rect> people;
        if (!validYoloDetections_.empty() && enableTracking) {
            auto [injected, nextValid] = updateValidDetections(raw_people, validYoloDetections_);

            if (injected.size() < (size_t)tracker_.getConfig().num_people &&
                raw_people.size() >= (size_t)tracker_.getConfig().num_people) {
                // Lost tracking: UI should request selection.
                lastYoloDetections_ = raw_people;
                out.yoloDetections = raw_people;
                out.requestSelection = true;
                out.statusText = "Tracking lost: please re-select people.";
                return out;
            }

            people = injected;
            validYoloDetections_ = nextValid;
        } else {
            people = raw_people;
        }

        lastYoloDetections_ = people;
        out.yoloDetections = people;

        if (enableTracking) {
            tracks_ = object_tracker_.inject(people, shifted.rows, shifted.cols, tracker_.getConfig().head_height_ratio);
            tracker_.yolo_updated();
        }
    }

    // Pose + gaze per track
    std::vector<PanoViewer::gaze> gazes;
    cv::Mat perspective_view;
    for (const auto& track : tracks_) {
        if (!track.viewer) continue;
        perspective_view = track.viewer->generatePerspectiveView(shifted);
        auto pose = tracker_.run(perspective_view, static_cast<int>(track.viewer->getFOV()));
        if (!pose.has_value()) continue;

        float yaw_boost = eyeBoost(tracker_.getConfig(), static_cast<float>(pose->yaw));

        PanoViewer::gaze g = track.viewer->addGaze(
            track.viewer->getYaw(),
            track.viewer->getPitch(),
            track.viewer->getFOV(),
            pose->yaw * yaw_boost,
            pose->pitch,
            cv::Vec3f(pose->x, pose->y, pose->z));
        g.box = track.viewer->convertPerspectiveRectToEquirectangular(pose->rect, shifted.cols, shifted.rows);
        gazes.emplace_back(g);
        perspective_view.release();
    }

    if (enableTracking) {
        if (justSeeked) object_tracker_.resetTrackerVelocities();
        gazes = object_tracker_.update(gazes);
        if (justSeeked) object_tracker_.resetTrackerVelocities();
    }

    out.gazes = gazes;

    // Default overlays (UI can choose colors)
    for (const auto& g : gazes) {
        domain::OverlayBox b;
        b.box = g.box;
        b.id = g.personID;
        b.color_bgr = 0x00FF00;
        out.overlays.push_back(b);
    }
    for (const auto& r : lastYoloDetections_) {
        domain::OverlayBox b;
        b.box = r;
        b.id = -1;
        b.color_bgr = 0x0000FF;
        out.overlays.push_back(b);
    }

    out.yoloActive = tracker_.need_yolo_update() != 0;
    return out;
}

} // namespace pipelines

