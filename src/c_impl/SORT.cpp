#include "SORT.h"
#include <set>
#include <limits>
#include <iostream>
#include "360_image_process.h"

#define M_PI 3.14159265358979323846

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

static float wrapX(float x, int width)
{
    x = std::fmod(x, width);
    if (x < 0) x += width;
    return x;
}

std::vector<PanoViewer::gaze> Sort::update(const std::vector<PanoViewer::gaze>& detections) {
    std::vector<PanoViewer::gaze> result_gazes;
    result_gazes.reserve(detections.size());

    // Early exit if no trackers and no detections
    if (trackers_.empty() && detections.empty()) {
        return {};
    }

    // 1. Predict new locations of existing trackers
    std::vector<cv::Rect2f> predictedBoxes;
    for (auto it = trackers_.begin(); it != trackers_.end();) {
        cv::Rect2f pBox = it->predict();
        if (pBox.width <= 0 || pBox.height <= 0) {
            it = trackers_.erase(it);
            continue;
        }

        // Check for NaN in prediction
        if (std::isnan(pBox.x) || std::isnan(pBox.y) || std::isnan(pBox.width) || std::isnan(pBox.height)) {
             std::cerr << "Warning: Tracker predicted NaN. Removing." << std::endl;
             it = trackers_.erase(it);
             continue;
        }
        
        predictedBoxes.push_back(pBox);
        ++it;
    }

    unsigned trackNum = predictedBoxes.size();
    unsigned detNum = detections.size();

    // 2. Create IOU cost matrix
    std::vector<std::vector<double>> iouMatrix;
    iouMatrix.resize(trackNum, std::vector<double>(detNum, 0));
    for (unsigned i = 0; i < trackNum; i++) {
        for (unsigned j = 0; j < detNum; j++) {
            iouMatrix[i][j] = 1 - getIOU(predictedBoxes[i], detections[j].box);
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

    if (trackNum == 0) {
        for (unsigned j = 0; j < detNum; ++j) unmatchedDetections.insert(j);
    }
    if (detNum == 0) {
        for (unsigned i = 0; i < trackNum; ++i) unmatchedTrajectories.insert(i);
    }

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

    // 5. Update matched trackers and assign IDs
    for (auto& match : matchedPairs) {
        trackers_[match.x].update(detections[match.y].box);
        
        PanoViewer::gaze g = detections[match.y];
        g.personID = trackers_[match.x].m_id + 1; // 1-based ID
        result_gazes.push_back(g);
    }

    // 6. Create new trackers for unmatched detections and assign IDs
    for (int idx : unmatchedDetections) {
        cv::Rect det = detections[idx].box;
        KalmanTracker tracker(det);
        trackers_.push_back(tracker);
        
        PanoViewer::gaze g = detections[idx];
        g.personID = tracker.m_id + 1;
        result_gazes.push_back(g);
    }

    // 7. Remove dead trackers and clean up reassign map
    for (auto it = trackers_.begin(); it != trackers_.end();) {
        if (it->m_time_since_update > max_age_) {
            reassign_map_.erase(it->m_id + 1); // remove internal ID from map
            it = trackers_.erase(it);
        } else {
            ++it;
        }
    }

    // 8. Reassign IDs if needed
    if (expected_num_people_ > 0) {
        std::set<int> active_display_ids;
        std::vector<PanoViewer::gaze*> need_reassignment;
        
        // Phase 1: Lock in native valid IDs (Priority 1)
        for (auto& gaze : result_gazes) {
            if (gaze.personID <= expected_num_people_) {
                active_display_ids.insert(gaze.personID);
            }
        }

        // Phase 2: Validate existing mappings and identify unmapped overflows
        // Note: Iterate again to handle the high IDs
        for (auto& gaze : result_gazes) {
             if (gaze.personID <= expected_num_people_) continue; // Already processed in Phase 1

             int internal_id = gaze.personID;
             
             if (reassign_map_.count(internal_id)) {
                 int mapped_id = reassign_map_[internal_id];
                 // Check collision with native IDs or previously accepted mappings
                 if (active_display_ids.count(mapped_id)) {
                     // Collision! Native ID returned or slot taken. Discard mapping.
                     reassign_map_.erase(internal_id);
                     need_reassignment.push_back(&gaze);
                 } else {
                     // Safe to keep mapping
                     gaze.personID = mapped_id;
                     active_display_ids.insert(mapped_id);
                 }
             } else {
                 // New overflow
                 need_reassignment.push_back(&gaze);
             }
        }

        // Phase 3: Assign missing IDs to unmapped items
        for (auto* gaze : need_reassignment) {
            int internal_id = gaze->personID;
            int assigned_id = -1;
            
            // Find first missing ID
            for (int i = 1; i <= expected_num_people_; ++i) {
                if (active_display_ids.find(i) == active_display_ids.end()) {
                    assigned_id = i;
                    break;
                }
            }
            
            if (assigned_id != -1) {
                reassign_map_[internal_id] = assigned_id;
                gaze->personID = assigned_id;
                active_display_ids.insert(assigned_id);
            } else {
                // No gaps available. Leave as overflow ID.
            }
        }
    }
    
    return result_gazes;
}

std::vector<Sort::Track> Sort::inject(std::vector<cv::Rect> detections, int rows, int cols, float head_height_ratio) {
    std::vector<Track> result;
    for (const auto& det : detections) {
        Track t;
        t.viewer = std::make_shared<PanoViewer>();
        
        // Compute virtual cam parameters directly from the detection rect
        // 1) compute yaw from bbox center x (panorama -> longitude)
        float center_x = det.x + det.width * 0.5f;
        double lambda = (center_x / (double)cols) * 2.0 * M_PI - M_PI;
        float yaw_deg = static_cast<float>(lambda * 180.0 / M_PI);

        // 2) compute phi (latitude) of the bbox TOP pixel in panorama
        double top_v = static_cast<double>(det.y); // top row of bbox in pano pixels
        double phi_top = (0.5 - (top_v / (double)rows)) * M_PI; // radians
        
        // 3) compute desired vertical FOV (deg)
        int h_star = static_cast<int>(t.viewer->getOutputHeight() * head_height_ratio); // e.g., head ~13% of input height
        float theta_deg = t.viewer->computeFOVForPersonBox(det, rows, h_star);
        double theta_rad = theta_deg * M_PI / 180.0;

        // 4) set pitch
        double pitch_rad = phi_top - 0.3 * theta_rad;
        float pitch_deg = static_cast<float>(pitch_rad * 180.0 / M_PI);
        
        t.viewer->setYaw(yaw_deg);
        t.viewer->setPitch(pitch_deg);
        t.viewer->setFOV(theta_deg);

        result.push_back(t);
    }
    return result;
}








