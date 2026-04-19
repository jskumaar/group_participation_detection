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

    void setPanoramaOffsetPx(int offsetPx) { panorama_offset_ = offsetPx; }
    int panoramaOffsetPx() const { return panorama_offset_; }

    void setExpectedPeople(int n);

    // Selection lifecycle
    domain::VisionFrameResult prepareSelection(const cv::Mat& panoFrame);
    void setSelectedDetections(const std::vector<cv::Rect>& selected, int panoRows, int panoCols);
    size_t selectedCount() const { return validYoloDetections_.size(); }

    // Main processing
    domain::VisionFrameResult processFrame(const cv::Mat& panoFrame, bool justSeeked, bool enableTracking);

    PanoViewer& panoViewer() { return pano_viewer_; }
    const PanoViewer& panoViewer() const { return pano_viewer_; }

private:
    cv::Mat applyPanoramaOffset(const cv::Mat& panoFrame) const;

    std::pair<std::vector<cv::Rect>, std::vector<cv::Rect>> updateValidDetections(
        const std::vector<cv::Rect>& newDetections,
        const std::vector<cv::Rect>& previousValid) const;

    OPNetTracker tracker_;
    Sort object_tracker_;
    PanoViewer pano_viewer_;

    std::vector<Sort::Track> tracks_;
    int yoloFrameCount_ = 0;
    bool justSeeked_ = false;

    std::vector<cv::Rect> lastYoloDetections_;
    std::vector<cv::Rect> validYoloDetections_;

    int panorama_offset_ = 0;
};

} // namespace pipelines

