#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "vision/360_image_process.h"

namespace domain {

struct OverlayBox {
    cv::Rect box;
    int id = -1;                // -1 means "unassigned"
    int color_bgr = 0x00FF00;   // packed BGR (0xBBGGRR) for non-Qt layers
};

struct VisionFrameResult {
    std::vector<PanoViewer::gaze> gazes;
    std::vector<OverlayBox> overlays;

    std::vector<cv::Rect> yoloDetections;  // raw detections used for selection/debug

    bool yoloActive = false;
    bool requestSelection = false;         // pipeline requests the UI to re-select people

    std::string statusText;                // optional status for UI
};

struct InteractionPair {
    int from_person_id = -1;
    int to_person_id = -1;
    float angle_deg = 0.0f;
    bool is_looking = false;
};

struct PipelineFrameContext {
    std::uint64_t frame_index = 0;
    std::uint64_t playback_timestamp_ns = 0;
    std::vector<InteractionPair> interactions;
};

} // namespace domain

