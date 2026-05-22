#pragma once

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "vision/360_image_process.h"

namespace domain {

struct InteractionPair {
    int from_person_id = -1;
    int to_person_id = -1;
    float angle_deg = 0.0f;
    bool is_looking = false;
};

struct VisionFrameResult {
    std::vector<PanoViewer::gaze> gazes;
    std::vector<InteractionPair> interactions;

    std::vector<cv::Rect> yoloDetections;  // raw detections used for selection/debug

    bool yoloActive = false;
    bool requestSelection = false;         // pipeline requests the UI to re-select people
};

struct PipelineFrameContext {
    std::uint64_t frame_index = 0;
    std::uint64_t playback_timestamp_ns = 0;
    std::vector<InteractionPair> interactions;
};

} // namespace domain
