#pragma once

#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

#include "domain/types.h"
#include "vision/360_image_process.h"

namespace vision {

struct GazeBayesConfig {
    float sigma_x_deg = 22.f;
    float sigma_y_deg = 11.f;
    float omega_base = 1.f;
    float omega_motion = 1.5f;
    float omega_null = 0.f;
    float likelihood_null_baseline = 0.15f;
};

class GazeBayesMapper {
public:
    void reset();

    std::vector<domain::InteractionPair> infer(
        const std::vector<PanoViewer::gaze>& gazes,
        const GazeBayesConfig& cfg);

private:
    std::unordered_map<int, cv::Rect2f> prev_box_projected_;
};

} // namespace vision
