
#include <vector>
#include <set>
#include <opencv2/opencv.hpp>
#include "sort-cpp/sort-c++/Hungarian.h"
#include "sort-cpp/sort-c++/KalmanTracker.h"



// SORT output. a track object with bbox and id

class Sort {
public:
    struct Track {
    int id;             // track ID
    cv::Rect2f box;     // pixel coordinates
    };

    Sort(int max_age = 1, int min_hits = 3, double iou_threshold = 0.3);
    //^ default contructor

    // Process detections for one frame and return current active tracks
    std::vector<Track> update(const std::vector<cv::Rect>& detections);

private:
    double getIOU(const cv::Rect2f& bb_test, const cv::Rect2f& bb_gt);

    std::vector<KalmanTracker> trackers_;
    int frame_count_;
    int max_age_;
    int min_hits_;
    double iou_threshold_;
};



