#include <opencv2/opencv.hpp>
#include <iostream>
#include "tracker.h"

int main() {
    // Open the default camera (camera index 0)
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the camera." << std::endl;
        return -1;
    }
    cv::Mat frame;
    OPNetTracker tracker;

    while (true) {
        cap >> frame;  // Capture a new frame from the camera
        if (frame.empty()) {
            std::cerr << "Error: Blank frame grabbed." << std::endl;
            break;
        }
        cv::imshow("Live Camera Feed", frame);  // Show the frame in a window
        Pose pose = tracker.run(frame, 70);
        printf("Yaw: %.2f, Pitch: %.2f, Roll: %.2f, X: %.2f, Y: %.2f, Z: %.2f\n", pose.yaw, pose.pitch, pose.roll, pose.x, pose.y, pose.z);
        // Exit on ESC key
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
