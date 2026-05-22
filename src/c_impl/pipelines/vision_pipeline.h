#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

#include "vision/sort_tracker.h"
#include "vision/opnet_tracker.h"
#include "vision/gaze_bayes_mapper.h"

#include "domain/types.h"

namespace pipelines {

class VisionPipeline {
public:
    VisionPipeline();

    TrackerConfig getConfig() const { return tracker_.getConfig(); }
    void setConfig(const TrackerConfig& cfg);

    void setExpectedPeople(int n);

    domain::VisionFrameResult prepareSelection(const cv::Mat& panoFrame);
    void setSelectedDetections(const std::vector<cv::Rect>& selected, int panoRows, int panoCols);

    domain::VisionFrameResult processFrame(const cv::Mat& panoFrame, bool justSeeked);

private:
    struct ValidDetectionUpdate {
        std::vector<cv::Rect> updatedYOLOBoxes;
        /** Same length as updatedYOLOBoxes; 1 when that slot matched this frame. */
        std::vector<char> updateIndices;
    };

    const cv::Mat& applyPanoramaOffset(const cv::Mat& panoFrame);

    ValidDetectionUpdate updateValidDetections(
        const std::vector<cv::Rect>& newDetections,
        const std::vector<cv::Rect>& previousValid) const;

    OPNetTracker tracker_;
    Sort object_tracker_;
    vision::GazeBayesMapper gaze_bayes_mapper_;

    std::vector<Sort::Track> tracks_;
    int yoloFrameCount_ = 0;

    /** Per-person boxes: empty before selection, then exactly num_people; association updates in place. */
    std::vector<cv::Rect> trackedBoxes_;

    /** Reused buffer for horizontal panorama roll (avoids per-frame hconcat allocation). */
    cv::Mat shifted_pano_;
};

} // namespace pipelines
