#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

#include "vision/sort_tracker.h"
#include "vision/opnet_tracker.h"
#include "vision/360_image_process.h"

#include "domain/types.h"

namespace pipelines {

class VisionPipeline {
public:
    VisionPipeline();

    TrackerConfig getConfig() const { return tracker_.getConfig(); }
    void setConfig(const TrackerConfig& cfg);

    void setExpectedPeople(int n);

    // Selection lifecycle
    domain::VisionFrameResult prepareSelection(const cv::Mat& panoFrame);
    void setSelectedDetections(const std::vector<cv::Rect>& selected, int panoRows, int panoCols);
    size_t selectedCount() const { return trackedBoxes_.size(); }

    // Main processing
    domain::VisionFrameResult processFrame(const cv::Mat& panoFrame, bool justSeeked);

    PanoViewer& panoViewer() { return pano_viewer_; }
    const PanoViewer& panoViewer() const { return pano_viewer_; }

private:
    struct ValidDetectionUpdate {
        std::vector<cv::Rect> updatedYOLOBoxes;
        /** Same length as updatedYOLOBoxes; 1 when that slot matched this frame. */
        std::vector<char> updateIndices;
    };

    cv::Mat applyPanoramaOffset(const cv::Mat& panoFrame) const;

    ValidDetectionUpdate updateValidDetections(
        const std::vector<cv::Rect>& newDetections,
        const std::vector<cv::Rect>& previousValid) const;

    OPNetTracker tracker_;
    Sort object_tracker_;
    PanoViewer pano_viewer_;

    std::vector<Sort::Track> tracks_;
    int yoloFrameCount_ = 0;
    bool justSeeked_ = false;

    /** Per-person boxes: empty before selection, then exactly num_people; association updates in place. */
    std::vector<cv::Rect> trackedBoxes_;
};

} // namespace pipelines

