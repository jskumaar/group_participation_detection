#include "pipelines/vision_pipeline.h"

#include <algorithm>

namespace pipelines {

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
    const int panoramaOffset = tracker_.getConfig().panorama_offset_px;
    if (panoramaOffset == 0) return panoFrame;
    int w = panoFrame.cols;
    int h = panoFrame.rows;
    int shift = panoramaOffset % w;
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

    out.yoloDetections = raw_people;
    out.yoloActive = true;
    out.requestSelection = true;
    out.statusText = "Selection required.";
    return out;
}

void VisionPipeline::setSelectedDetections(const std::vector<cv::Rect>& selected, int panoRows, int panoCols) {
    const int n = tracker_.getConfig().num_people;
    if ((int)selected.size() != n) return;

    trackedBoxes_ = selected;
    if (panoRows > 0 && panoCols > 0) {
        tracks_ = object_tracker_.inject(trackedBoxes_, panoRows, panoCols, tracker_.getConfig().head_height_ratio);
    } else {
        tracks_.clear();
    }
}

VisionPipeline::ValidDetectionUpdate VisionPipeline::updateValidDetections(
    const std::vector<cv::Rect>& newDetections,
    const std::vector<cv::Rect>& previousValid) const {

    ValidDetectionUpdate out;
    out.updatedYOLOBoxes = previousValid;
    out.updateIndices.assign(previousValid.size(), 0);

    std::vector<bool> usedNew(newDetections.size(), false);
    float threshold = tracker_.getConfig().yolorerun_threshold;

    for (size_t i = 0; i < out.updatedYOLOBoxes.size(); ++i) {
        int bestIdx = -1;
        float bestIoU = -1.0f;

        for (size_t j = 0; j < newDetections.size(); ++j) {
            if (usedNew[j]) continue;

            cv::Rect intersect = out.updatedYOLOBoxes[i] & newDetections[j];
            float intersectArea = static_cast<float>(intersect.area());
            float unionArea = static_cast<float>(out.updatedYOLOBoxes[i].area() + newDetections[j].area() - intersectArea);
            if (unionArea <= 0) continue;

            float iou = intersectArea / unionArea;
            if (iou > threshold && iou > bestIoU) {
                bestIoU = iou;
                bestIdx = static_cast<int>(j);
            }
        }

        if (bestIdx != -1) {
            out.updatedYOLOBoxes[i] = newDetections[bestIdx];
            out.updateIndices[i] = 1;
            usedNew[bestIdx] = true;
        }
    }

    return out;
}

domain::VisionFrameResult VisionPipeline::processFrame(const cv::Mat& panoFrame, bool justSeeked) {
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

    const int num_people = tracker_.getConfig().num_people;
    const bool slots_ready = (int)trackedBoxes_.size() == num_people;

    out.yoloDetections.clear();

    if (tracker_.need_yolo_update() || tracks_.empty() ||
        (tracks_.size() < (size_t)num_people && interval_check)) {
        std::vector<cv::Rect> raw_people = tracker_.run_yolo(shifted);
        out.yoloActive = true;

        std::vector<cv::Rect> people;
        if (slots_ready) {
            ValidDetectionUpdate assoc = updateValidDetections(raw_people, trackedBoxes_);
            const size_t matchedSlots =
                static_cast<size_t>(std::count(assoc.updateIndices.begin(), assoc.updateIndices.end(), char(1)));

            //if less matched slots than num_people, request re-selection
            //after re-selection, app updates trackedBoxes_ and tracks_
            if (matchedSlots < (size_t)num_people &&
                raw_people.size() >= (size_t)num_people) {
                out.yoloDetections = raw_people;
                out.requestSelection = true;
                out.statusText = "Tracking lost: please re-select people.";
                return out;
            }
            //if there are fewer matched slots than num_people, 
            //but also fewer raw people than num_people, then keep track of stale slots
            for (size_t i = 0; i < assoc.updatedYOLOBoxes.size(); ++i) {
                if (assoc.updateIndices[i]) people.push_back(assoc.updatedYOLOBoxes[i]);
            }
            trackedBoxes_ = std::move(assoc.updatedYOLOBoxes);
            out.yoloDetections = people;
        } else {
            people = raw_people;
            out.yoloDetections = people;
        }

        tracks_ = object_tracker_.inject(people, shifted.rows, shifted.cols, tracker_.getConfig().head_height_ratio);
        tracker_.yolo_updated();
    }

    // Pose + gaze per track
    std::vector<PanoViewer::gaze> gazes;
    cv::Mat perspective_view;
    for (const auto& track : tracks_) {
        if (!track.viewer) continue;
        perspective_view = track.viewer->generatePerspectiveView(shifted);
        auto pose = tracker_.run(perspective_view, static_cast<int>(track.viewer->getFOV()));
        if (!pose.has_value()) continue;

        PanoViewer::gaze g = track.viewer->addGaze(
            track.viewer->getYaw(),
            track.viewer->getPitch(),
            track.viewer->getFOV(),
            pose->yaw,
            pose->pitch,
            cv::Vec3f(pose->x, pose->y, pose->z));
        g.box = track.viewer->convertPerspectiveRectToEquirectangular(pose->rect, shifted.cols, shifted.rows);
        gazes.emplace_back(g);
        perspective_view.release();
    }

    if (justSeeked) object_tracker_.resetTrackerVelocities();
    gazes = object_tracker_.update(gazes);
    if (justSeeked) object_tracker_.resetTrackerVelocities();

    out.gazes = gazes;

    for (const auto& g : gazes) {
        domain::OverlayBox b;
        b.box = g.box;
        b.id = g.personID;
        b.color_bgr = 0x00FF00;
        out.overlays.push_back(b);
    }
    for (const auto& r : out.yoloDetections) {
        domain::OverlayBox b;
        b.box = r;
        b.id = -1;
        b.color_bgr = 0x0000FF;
        out.overlays.push_back(b);
    }

    return out;
}

} // namespace pipelines
