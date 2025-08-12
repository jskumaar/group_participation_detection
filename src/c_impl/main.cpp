#include "360_image_process.h"
#include "tracker.h"
#include <iostream>
#include <vector>
#include <cstdio>

#define M_PI 3.14159265358979323846

PanoViewer viewer;
OPNetTracker tracker;

#define GLOBAL_FRAME_WIDTH 2880
#define GLOBAL_FRAME_HEIGHT 1440


void visualize(FILE* gp, int R, std::vector<PanoViewer::gaze> gazes){

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
        double gx = p.start.x + p.direction[0] * 100;
        double gy = p.start.y + p.direction[1] * 100; 
        double gz = p.start.z + p.direction[2] * 100;
        fprintf(gp, "%f %f %f %f %f %f\n", p.start.x, p.start.y, p.start.z, gx - p.start.x, gy - p.start.y, gz - p.start.z);
    }
    fprintf(gp, "e\n"); 
    fflush(gp);
}

int main() {
    cv::VideoCapture cap("demo_1_stitched.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open camera input 1. Trying input 0..." << std::endl;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, GLOBAL_FRAME_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, GLOBAL_FRAME_HEIGHT);

    cv::Mat pano_frame;

    cv::namedWindow("360° Viewer", cv::WINDOW_NORMAL);  // Changed from WINDOW_AUTOSIZE
    cv::resizeWindow("360° Viewer", 640, 480);


    FILE* gp = _popen("gnuplot -persistent", "w");
    if (!gp) { std::cerr << "Failed to start gnuplot\n"; return 1; }

    // Set up 3D plot
    fprintf(gp, "set term wxt size 800,600\n");
    fprintf(gp, "set view equal xyz\n");
    fprintf(gp, "set view 60, 30\n");
    fprintf(gp, "set xrange [-%d:%d]\n", 100, 100);
    fprintf(gp, "set yrange [-%d:%d]\n", 100, 100);
    fprintf(gp, "set zrange [-%d:%d]\n", 100, 100);
    fprintf(gp, "set ticslevel 0\n");

    fprintf(gp, "set xlabel 'X (Right)'\n");
    fprintf(gp, "set ylabel 'Y (Up)'\n");
    fprintf(gp, "set zlabel 'Z (Forward)'\n");

    std::vector<PanoViewer::gaze> gazes;
    size_t frame_count = 0;
    cv::Mat perspective_view;
    std::vector<cv::Rect> people;
    while (true) {
        cap >> pano_frame;
        cv::resize(pano_frame, pano_frame, cv::Size(2880,1440)); // new width, height
        if (pano_frame.empty()) {
            std::cerr << "Failed to capture frame!" << std::endl;
            break;
        }
        // Ensure we have the right aspect ratio (2:1 for equirectangular)
        if (pano_frame.cols / (float)pano_frame.rows != 2.0f) {
            printf("Error: Incorrect aspect ratio\n");
            exit(1);
        }
        if(frame_count % 10 == 0){
            printf("Running YOLO on frame %zu\n", frame_count);
            people = tracker.run_yolo(pano_frame);
        }
        
        // Generate perspective view
        int person_id = 0;  // Counter for unique window names
        //person 0 = suresh
        //person 1 = naveen
        //person 2 = other
        for(const cv::Rect& person : people){
            if(person_id == 0){
                viewer.setFOV(55);
            }
            else if(person_id == 1){
                viewer.setFOV(30);
            }
            else if(person_id == 2){
                viewer.setFOV(40);
            }
            int yaw = ((person.x + person.width / 2) * 0.125) - 180.0;
            viewer.setYaw(yaw);
            perspective_view = viewer.generatePerspectiveView(pano_frame);
            rotation_output pose = tracker.run(perspective_view);
            printf("personID: %d, yaw: %f, pitch: %f\n", person_id, pose.yaw, pose.pitch);
            PanoViewer::gaze gaze = viewer.addGaze(person_id, pose.yaw, pose.pitch, person);
            gazes.push_back(gaze);
            auto pano_pixel = viewer.rayToPanoPixel(cv::Vec3f(gaze.start.x, gaze.start.y, gaze.start.z),
                                  gaze.direction,
                                  90.0f,
                                  GLOBAL_FRAME_WIDTH,
                                  GLOBAL_FRAME_HEIGHT);

            printf("Pano Pixel: x=%d, y=%d\n", pano_pixel.x, pano_pixel.y);
            cv::circle(pano_frame, pano_pixel, 10, cv::Scalar(255, 0, 0), 2);
            cv::rectangle(pano_frame, person, cv::Scalar(0, 255, 0), 2);  // Green rectangle, 2px thick
            person_id++;
            perspective_view.release();
        }
        visualize(gp, 90, gazes);
        gazes.clear();
        imshow("360° Viewer", pano_frame);
        char key = cv::waitKey(1) & 0xFF;
        if (key == 27) break; // ESC to exit
        frame_count++;
    }

    _pclose(gp);
    cv::destroyAllWindows();
    return 0;
}
