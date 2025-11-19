#include "360_image_process.h"
#include "tracker.h"

#include "SORT.h"
#include <iostream>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <string>


// NEED FOR WINDOWS
#define M_PI 3.14159265358979323846

PanoViewer viewer;
OPNetTracker tracker;

#define GLOBAL_FRAME_WIDTH 2880
#define GLOBAL_FRAME_HEIGHT 1440
#define LOCALIZER_THRESHOLD 0.5


Sort object_tracker;
std::vector<Sort::Track> tracks;

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


int main() {

    // change based on where ur vid is located
    // Atindrah's video path
    // char* vid_path = "/Users/atind/Downloads/demo_1_stitched.mp4";
    // NAVEEN'S video path
    char* vid_path = "demo_1_stitched.mp4";
    cv::VideoCapture cap(vid_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open camera input 1. Trying input 0..." << std::endl;
    }


    cap.set(cv::CAP_PROP_FRAME_WIDTH, GLOBAL_FRAME_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, GLOBAL_FRAME_HEIGHT);
    //cap.set(cv::CAP_PROP_POS_FRAMES, 100);
    cap.set(cv::CAP_PROP_POS_FRAMES, 6780);

    cv::Mat pano_frame;

    cv::namedWindow("360° Viewer", cv::WINDOW_NORMAL);  // Changed from WINDOW_AUTOSIZE
    cv::resizeWindow("360° Viewer", 640, 480);
    //cv::namedWindow("Perspective View", cv::WINDOW_NORMAL);  // Changed from WINDOW_AUTOSIZE
    //cv::resizeWindow("Perspective View", 640, 480);



    // MAC
    // FILE* gp = popen("gnuplot -persistent", "w");
    // USE ONE BELOW FOR WINDOWS 
    FILE* gp = _popen("gnuplot -persistent", "w");

    if (!gp) { std::cerr << "Failed to start gnuplot\n"; return 1; }

    // Set up 3D plot
    fprintf(gp, "set term wxt size 800,600\n");
    fprintf(gp, "set view equal xyz\n");
    fprintf(gp, "set view 60, 30\n");
    fprintf(gp, "set xrange [-%d:%d]\n", 200, 200);
    fprintf(gp, "set yrange [-%d:%d]\n", 200, 200);
    fprintf(gp, "set zrange [-%d:%d]\n", 200, 200);
    fprintf(gp, "set ticslevel 0\n");

    fprintf(gp, "set xlabel 'X (Right)'\n");
    fprintf(gp, "set ylabel 'Y (Up)'\n");
    fprintf(gp, "set zlabel 'Z (Forward)'\n");

    std::vector<PanoViewer::gaze> gazes;
    std::vector<cv::Rect> people;
    cv::Mat perspective_view;

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
    while (true) {
        cap >> pano_frame;


        if (localizer_confidence < LOCALIZER_THRESHOLD) {
            printf("YOLO Running");
            people = tracker.run_yolo(pano_frame);
            tracks = object_tracker.update(people, pano_frame.rows, pano_frame.cols, viewer);
            localizer_confidence = 1;
        }

        // Generate perspective view
         // Counter for unique window names
        //cv::Scalar color;
        for (const Sort::Track& track : tracks) {

            // apply pose and FOV
            viewer.setYaw(track.yaw);
            viewer.setPitch(track.pitch);
            viewer.setFOV(track.fov);


            perspective_view = viewer.generatePerspectiveView(pano_frame);
            Pose pose = tracker.run(perspective_view, static_cast<int>(viewer.getFOV()));
            if (pose.confidence < LOCALIZER_THRESHOLD) {
                localizer_confidence = pose.confidence;
                continue;
            }
            printf("personID: %d, yaw: %f, pitch: %f, roll: %f, x: %f, y: %f, z: %f\n", track.id, pose.yaw, pose.pitch, pose.roll, pose.x, pose.y, pose.z);
            PanoViewer::gaze gaze = viewer.addGaze(track.id, track.yaw, -track.pitch, track.fov, pose.yaw*1.5, -pose.pitch, cv::Vec3f(pose.x, pose.y, pose.z));
            gazes.push_back(gaze);
        //    //auto pano_pixel = viewer.rayToPanoPixel(cv::Vec3f(gaze.start.x, gaze.start.y, gaze.start.z),
        //    //    gaze.direction,
        //    //    pose.z,
        //    //    GLOBAL_FRAME_WIDTH,
        //    //    GLOBAL_FRAME_HEIGHT);

        //    //printf("Pano Pixel: x=%d, y=%d\n", pano_pixel.x, pano_pixel.y);


        ////    int intersect = PanoViewer::bbox_intersections(cv::Point(pano_pixel.x, pano_pixel.y), cv::Point(person.x, person.y), tracks);
        ////    cv::circle(pano_frame, pano_pixel, 40, color, 2);
        ////    std::string text = "Person: " + std::to_string(person_id) + " looking at person: " + std::to_string(intersect);
        ////    cv::putText(pano_frame, text, cv::Point(person.x, person.y), cv::FONT_HERSHEY_SIMPLEX, 1.0, color, 2);


        ////    cv::rectangle(pano_frame, person, cv::Scalar(0, 255, 0), 2);  // Green rectangle, 2px thick

        ////    // NOT NEEDED FOR TESTING PURPOSES 
        ////    // PanoViewer::add_frame_to_window(gaze_windows[person_id], person_id, intersect);
        ////    // PanoViewer::print_gaze_window(gaze_windows[person_id], pano_frame, scaled);
            perspective_view.release();
        }
        visualize(gp, 90, gazes);
        gazes.clear();
        imshow("360° Viewer", pano_frame);
        pano_frame.release();
        char key = cv::waitKey(1) & 0xFF;
        if (key == 27) break; // ESC to exit
    }

    // MAC
    // pclose(gp);
    // USE ONE BELOW FOR WINDOWS
    _pclose(gp);
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
