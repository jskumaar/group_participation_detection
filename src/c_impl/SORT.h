#pragma once
#include <vector>
#include <set>
#include <opencv2/opencv.hpp>
#include "sort-cpp/sort-c++/Hungarian.h"
#include "sort-cpp/sort-c++/KalmanTracker.h"

#include "360_image_process.h"

// SORT output. a track object with bbox and id

class Sort {
public:
    struct Track {
    std::shared_ptr<PanoViewer> viewer = nullptr;
    };

    Sort(int max_age = 1000, int min_hits = 3, double iou_threshold = 0.15);
    //^ default contructor

    void setIOUThreshold(double threshold) { iou_threshold_ = threshold; }

    // Process detections for one frame and return current active tracks
    // viewer is forward-declared; include 360_image_process.h in SORT.cpp where implementation needs it
    std::vector<PanoViewer::gaze> update(const std::vector<PanoViewer::gaze>& detections);
    std::vector<Track> inject(std::vector<cv::Rect> detections, int rows, int cols, float head_height_ratio = 0.13f);

private:
    double getIOU(const cv::Rect2f& bb_test, const cv::Rect2f& bb_gt);

    std::vector<KalmanTracker> trackers_;
    int frame_count_;
    int max_age_;
    int min_hits_;
    double iou_threshold_;
};



