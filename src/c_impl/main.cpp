/**
 * @file main.cpp
 * @brief Main application for 360-degree panoramic video gaze tracking
 *
 * This application processes 360-degree panoramic videos to track people and their gazes,
 * determining when individuals are looking at each other.
 */

#include "360_image_process.h"
#include "tracker.h"

#include "SORT.h"
#include <iostream>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <string>
#include <chrono>
#include <sstream>
#include <fstream>
#include <map>


// NEED FOR WINDOWS
#define M_PI 3.14159265358979323846
OPNetTracker tracker;

#define GLOBAL_FRAME_WIDTH 2880
#define GLOBAL_FRAME_HEIGHT 1440
#define LOCALIZER_THRESHOLD 0.5


Sort object_tracker;
std::vector<Sort::Track> tracks;
PanoViewer pano_viewer;

void visualize(FILE* gp, int R, std::vector<PanoViewer::gaze> gazes) {

    // Draw loop (could be in a while(true) for realtime)
    fprintf(gp, "splot '-' with points pt 7 lc rgb 'black' title 'Sphere',"
        "'-' with points pt 7 ps 1.5 lc rgb 'red' title 'People',"
        "'-' with vectors nohead lc rgb 'blue' title 'Gaze'\n");

    // Sphere points (mesh)
    int n_theta = 40, n_phi = 20;
    for (int i = 0; i <= n_phi; ++i) {
        double phi = M_PI * i / n_phi;
        for (int j = 0; j <= n_theta; ++j) {
            double theta = 2 * M_PI * j / n_theta;
            double x = R * std::sin(phi) * std::cos(theta);
            double y = R * std::cos(phi);
            double z = R * std::sin(phi) * std::sin(theta);
            fprintf(gp, "%f %f %f\n", x, y, z);
        }
    }
    fprintf(gp, "e\n");

    // People positions
    for (auto& p : gazes)
        fprintf(gp, "%f %f %f\n", p.start.x, p.start.y, p.start.z);
    fprintf(gp, "e\n");

    for (auto& p : gazes) {
        double gx = p.start.x + p.direction[0] * 200;
        double gy = p.start.y + p.direction[1] * 200;
        double gz = p.start.z + p.direction[2] * 200;
        fprintf(gp, "%f %f %f %f %f %f\n", p.start.x, p.start.y, p.start.z, gx - p.start.x, gy - p.start.y, gz - p.start.z);
    }
    fprintf(gp, "e\n");
    fflush(gp);
}

float eyeBoost(float yaw_deg) {
    float y = std::abs(yaw_deg);

    // Tunable parameters
    float max_boost = 0.7f;   // At extreme yaw → +50% multiplier (1.5x)
    float knee = 13.0f;       // Where curve begins rising rapidly
    float steepness = 0.10f;  // Controls how fast it rises

    // Increasing sigmoid from 0 → max_boost
    float extra = max_boost * (1.0f - 1.0f / (1.0f + std::exp((y - knee) * steepness)));

    return 1.0f + extra;
}

// Configuration
const char* VIDEO_PATH = "demo_1_stitched.mp4";

int main() {
    cv::VideoCapture cap(VIDEO_PATH);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open camera input 1. Trying input 0..." << std::endl;
    }


    cap.set(cv::CAP_PROP_FRAME_WIDTH, GLOBAL_FRAME_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, GLOBAL_FRAME_HEIGHT);
    cap.set(cv::CAP_PROP_POS_FRAMES, 6780);

    cv::Mat pano_frame;

#ifdef _WIN32 
    FILE* gp = _popen("gnuplot -persistent", "w");
#else
    FILE* gp = popen("gnuplot -persistent", "w");
#endif

//    cv::namedWindow("360° Viewer", cv::WINDOW_NORMAL);  // Changed from WINDOW_AUTOSIZE
//cv::resizeWindow("360° Viewer", 640, 480);

    if (!gp) { std::cerr << "Failed to start gnuplot\n"; return 1; }

    // Set up 3D plot
    fprintf(gp, "set term wxt size 800,600\n");
    fprintf(gp, "set view equal xyz\n");
    fprintf(gp, "set view 60, 30\n");
    fprintf(gp, "set xrange [%d:-%d]\n", 200, 200);
    fprintf(gp, "set yrange [%d:-%d]\n", 200, 200);
    fprintf(gp, "set zrange [-%d:%d]\n", 200, 200);
    fprintf(gp, "set ticslevel 0\n");

    fprintf(gp, "set xlabel 'X (Right)'\n");
    fprintf(gp, "set ylabel 'Y (Up)'\n");
    fprintf(gp, "set zlabel 'Z (Forward)'\n");

    std::vector<PanoViewer::gaze> gazes;
    gazes.reserve(20); // Pre-allocate for typical number of people
    std::vector<cv::Rect> people;
    cv::Mat perspective_view;

    // FPS calculation variables
    using clock_t = std::chrono::steady_clock;
    auto fps_last_time = clock_t::now();
    int fps_frame_count = 0;
    double fps = 0.0;

    // Cache commonly used values
    const cv::Scalar green_color(0, 255, 0);
    const cv::Scalar black_color(0, 0, 0);
    constexpr double font_scale = 1.0;
    constexpr int thickness = 2;
    std::string id_text;
    id_text.reserve(20); // Pre-allocate string buffer

    // array of gaze_windows based on index per person

    // NOT NEEDED FOR TESTING PURPOSES 
    // std::vector<PanoViewer::gaze_window> gaze_windows(num_people);
    // for (int i = 0; i < num_people; i++) {
    //     gaze_windows[i].personID = i;
    //     gaze_windows[i].max_window_length = 20; 
    //     gaze_windows[i].counts.resize(num_people, 0);
    //     gaze_windows[i].sights.clear();
    // }
    float localizer_confidence = 0;
	bool yolo_update = false;

    std::ofstream csv_file("gaze_analysis.csv");
    if (csv_file.is_open()) {
        csv_file << "Frame,PersonID,BoxCenterX,BoxCenterY,GazeStartX,GazeStartY,GazeStartZ,GazeDirX,GazeDirY,GazeDirZ,LookingAtIDs\n";
    }

    while (true) {
		yolo_update = false;
        cap >> pano_frame;
        if(pano_frame.cols != pano_frame.rows * 2) {
            std::cerr << "Error: Frame dimensions are not as expected (width should be twice the height)." << std::endl;
            break;
        } 

        // Update FPS counters
        fps_frame_count++;
        auto now = clock_t::now();
        std::chrono::duration<double> elapsed = now - fps_last_time;
        if (elapsed.count() >= 1.0) {
            fps = fps_frame_count / elapsed.count();
            fps_frame_count = 0;
            fps_last_time = now;
            std::cout << "FPS: " << fps << std::endl;
        }


        if (localizer_confidence < LOCALIZER_THRESHOLD) {
            printf("YOLO Running: \t");
            people = tracker.run_yolo(pano_frame);
            tracks = object_tracker.update(people, pano_frame.rows, pano_frame.cols);
            localizer_confidence = 1.0f;
			yolo_update = true;
        }

        // Generate perspective view
        const size_t num_tracks = tracks.size();
        for (size_t i = 0; i < num_tracks; ++i) {
            const Sort::Track& track = tracks[i];

            // Draw bbox and ID on panorama
            cv::Rect bbox = track.box; // convert Rect2f -> Rect (truncates)
            // clamp coordinates
            bbox.x = std::max(0, bbox.x);
            bbox.y = std::max(0, bbox.y);
            bbox.width = std::min(bbox.width, pano_frame.cols - bbox.x);
            bbox.height = std::min(bbox.height, pano_frame.rows - bbox.y);
            cv::rectangle(pano_frame, bbox, green_color, 2);
            
            // Reuse pre-allocated string buffer
            id_text = "ID: ";
            id_text += std::to_string(track.id);
            
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(id_text, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
            const cv::Point text_pt(bbox.x, std::max(20, bbox.y - 10));
            
            // draw filled rectangle behind text for readability
            cv::rectangle(pano_frame, 
                         cv::Point(text_pt.x, text_pt.y - text_size.height - 4), 
                         cv::Point(text_pt.x + text_size.width, text_pt.y + 4), 
                         black_color, cv::FILLED);
            cv::putText(pano_frame, id_text, text_pt, cv::FONT_HERSHEY_SIMPLEX, font_scale, green_color, thickness);

            perspective_view = track.viewer->generatePerspectiveView(pano_frame, yolo_update);
            Pose pose = tracker.run(perspective_view, static_cast<int>(track.viewer->getFOV()));
            if (pose.confidence < LOCALIZER_THRESHOLD) {
                localizer_confidence = pose.confidence;
                continue;
            }
            
            const float yaw_boost = eyeBoost(pose.yaw);

            PanoViewer::gaze g = track.viewer->addGaze(track.viewer->getYaw(), 
                                                      -track.viewer->getPitch(), 
                                                      track.viewer->getFOV(), 
                                                      pose.yaw * yaw_boost, 
                                                      pose.pitch, 
                                                      cv::Vec3f(pose.x, pose.y, pose.z));
            g.boxCenter = cv::Point2f(track.box.x + track.box.width / 2.0f, track.box.y + track.box.height / 2.0f);
            g.personID = track.id;
            gazes.emplace_back(g);
            perspective_view.release();
        }
		//pano_viewer.gaze_analysis(gazes);

        // Save to CSV
        pano_viewer.saveGazeAnalysis(csv_file, (long)cap.get(cv::CAP_PROP_POS_FRAMES), gazes);
        //visualize(gp, 10, gazes);
        gazes.clear();
        //imshow("360° Viewer", pano_frame);
        //char key = cv::waitKey(100) & 0xFF;
        //if (key == 27) break; // ESC to exit
    }

#ifdef _WIN32
    _pclose(gp);
#else
    pclose(gp);
#endif
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
