#pragma once

#include <vector>
#include <set>
#include <map>

#include <opencv2/opencv.hpp>

#include "third_party/sort-cpp/sort-c++/Hungarian.h"
#include "third_party/sort-cpp/sort-c++/KalmanTracker.h"

#include "vision/360_image_process.h"

class Sort {
public:
    struct Track {
        std::shared_ptr<PanoViewer> viewer = nullptr;
    };

    Sort(int max_age = 1000, int min_hits = 3, double iou_threshold = 0.15);

    void setIOUThreshold(double threshold) { iou_threshold_ = threshold; }

    std::vector<PanoViewer::gaze> update(const std::vector<PanoViewer::gaze>& detections);
    std::vector<Track> inject(std::vector<cv::Rect> detections, int rows, int cols, float head_height_ratio = 0.13f);

    void setNumPeople(int n) { expected_num_people_ = n; }
    void resetTrackerVelocities();

private:
    double getIOU(const cv::Rect2f& bb_test, const cv::Rect2f& bb_gt);

    std::vector<KalmanTracker> trackers_;
    int frame_count_;
    int max_age_;
    int min_hits_;
    double iou_threshold_;

    int expected_num_people_ = 0;
    std::map<int, int> reassign_map_;
};

