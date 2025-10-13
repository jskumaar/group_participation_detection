#include "SORT.h"
#include <set>
#include <limits>

Sort::Sort(int max_age, int min_hits, double iou_threshold)
    : frame_count_(0), max_age_(max_age), min_hits_(min_hits), iou_threshold_(iou_threshold) {
    KalmanTracker::kf_count = 0;
}

double Sort::getIOU(const cv::Rect2f& bb_test, const cv::Rect2f& bb_gt) {
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;
    if (un < std::numeric_limits<float>::epsilon()) return 0.0;
    return static_cast<double>(in / un);
}

std::vector<Sort::Track> Sort::update(const std::vector<cv::Rect>& detections) {

    // Early exit if no trackers and no detections
    if (trackers_.empty() && detections.empty()) {
        return {};
    }

    // 1. Predict new locations of existing trackers
    std::vector<cv::Rect2f> predictedBoxes;
    for (auto it = trackers_.begin(); it != trackers_.end();) {
        cv::Rect2f pBox = it->predict();
        if (pBox.width > 0 && pBox.height > 0 && pBox.x >= 0 && pBox.y >= 0) {
            predictedBoxes.push_back(pBox);
            ++it;
        } else {
            it = trackers_.erase(it);
        }
    }

    unsigned trackNum = predictedBoxes.size();
    unsigned detNum = detections.size();

    // 2. Create IOU cost matrix only if we have trackers and detections
    std::vector<std::vector<double>> iouMatrix;
    
    iouMatrix.resize(trackNum, std::vector<double>(detNum, 0));
    for (unsigned i = 0; i < trackNum; i++) {
        for (unsigned j = 0; j < detNum; j++) {
            iouMatrix[i][j] = 1 - getIOU(predictedBoxes[i], detections[j]);
        }
    }


    // 3. Solve assignment
    std::vector<int> assignment(trackNum, -1);
    if (trackNum > 0 && detNum > 0) {
        HungarianAlgorithm HungAlgo;
        HungAlgo.Solve(iouMatrix, assignment);
    }

    // 4. Find matches and unmatched
    std::set<int> unmatchedDetections;
    std::set<int> unmatchedTrajectories;
    std::vector<cv::Point> matchedPairs;

    // All detections unmatched if no trackers
    if (trackNum == 0) {
        for (unsigned j = 0; j < detNum; ++j) unmatchedDetections.insert(j);
    }
    // All trackers unmatched if no detections
    if (detNum == 0) {
        for (unsigned i = 0; i < trackNum; ++i) unmatchedTrajectories.insert(i);
    }

    // Handle normal case
    if (trackNum > 0 && detNum > 0) {
        // Unmatched detections
        if (detNum > trackNum) {
            std::set<int> allItems, matchedItems;
            for (unsigned n = 0; n < detNum; n++) allItems.insert(n);
            for (unsigned i = 0; i < trackNum; ++i)
                if (assignment[i] != -1) matchedItems.insert(assignment[i]);
            std::set_difference(allItems.begin(), allItems.end(),
                                matchedItems.begin(), matchedItems.end(),
                                std::inserter(unmatchedDetections, unmatchedDetections.begin()));
        }
        // Unmatched trackers
        else if (detNum < trackNum) {
            for (unsigned i = 0; i < trackNum; ++i)
                if (assignment[i] == -1) unmatchedTrajectories.insert(i);
        }

        // Filter matches by IOU threshold
        for (unsigned i = 0; i < trackNum; ++i) {
            if (assignment[i] == -1) continue;
            if (1 - iouMatrix[i][assignment[i]] < iou_threshold_) {
                unmatchedTrajectories.insert(i);
                unmatchedDetections.insert(assignment[i]);
            } else {
                matchedPairs.push_back(cv::Point(i, assignment[i]));
            }
        }
    }

    // 5. Update matched trackers safely
    for (auto& match : matchedPairs) {
        if (match.x < trackers_.size() && match.y < detections.size()) {
            trackers_[match.x].update(detections[match.y]);
        } else {
            std::cerr << "Warning: invalid match indices: "
                      << match.x << ", " << match.y << std::endl;
        }
    }

    // 6. Create new trackers for unmatched detections
    for (int idx : unmatchedDetections) {
        if (idx < detections.size()) {
            cv::Rect det = detections[idx];
            if (det.width > 0 && det.height > 0) {
                KalmanTracker tracker(det);
                trackers_.push_back(tracker);
            } else {
                std::cerr << "Skipping invalid detection: " << det << std::endl;
            }
        }
    }

    // 7. Collect results and remove dead trackers
    std::vector<Track> result;
    for (auto it = trackers_.begin(); it != trackers_.end();) {
        if ((it->m_time_since_update < 1) &&
            (it->m_hit_streak >= min_hits_ || frame_count_ <= min_hits_)) {
            Track t;
            t.id = it->m_id + 1;
            t.box = it->get_state();
            result.push_back(t);
        }
        if (it->m_time_since_update > max_age_) {
            it = trackers_.erase(it);
        } else {
            ++it;
        }
    }

    return result;
}






