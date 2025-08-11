#include "360_image_process.h"
#include "tracker.h"

PanoViewer viewer;
OPNetTracker tracker;

#define GLOBAL_FRAME_WIDTH 2880
#define GLOBAL_FRAME_HEIGHT 1440

int main() {
    cv::VideoCapture cap(1);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open camera input 1. Trying input 0..." << std::endl;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, GLOBAL_FRAME_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, GLOBAL_FRAME_HEIGHT);

    cv::Mat pano_frame;

    cv::namedWindow("360° Viewer", cv::WINDOW_NORMAL);  // Changed from WINDOW_AUTOSIZE
    cv::resizeWindow("360° Viewer", 640, 480);
    while (true) {
        cap >> pano_frame;
        if (pano_frame.empty()) {
            std::cerr << "Failed to capture frame!" << std::endl;
            break;
        }
        // Ensure we have the right aspect ratio (2:1 for equirectangular)
        if (pano_frame.cols / (float)pano_frame.rows != 2.0f) {
            printf("Error: Incorrect aspect ratio\n");
            exit(1);
        }
        auto people = tracker.run_yolo(pano_frame);
        // Generate perspective view
        std::vector<PanoViewer::gaze> gazes;
        int person_id = 0;  // Counter for unique window names
        for(const cv::Rect& person : people){
            int yaw = ((person.x + person.width / 2) * 0.125) - 180.0;
            viewer.setYaw(yaw);
            cv::Mat perspective_view = viewer.generatePerspectiveView(pano_frame);
            rotation_output pose = tracker.run(perspective_view);
            PanoViewer::gaze gaze = viewer.addGaze(person_id, pose.yaw, pose.pitch, person);
            gazes.push_back(gaze);
            cv::Point pano_pixel = viewer.rayToPanoPixel(cv::Vec3f(gaze.start.x, gaze.start.y, gaze.start.z),
                                  gaze.direction,
                                  90.0f,
                                  GLOBAL_FRAME_WIDTH,
                                  GLOBAL_FRAME_HEIGHT);

            printf("Pano Pixel: x=%f, y=%f\n", pano_pixel.x, pano_pixel.y);
            cv::circle(pano_frame, pano_pixel, 10, cv::Scalar(255, 0, 0), 2);
            cv::rectangle(pano_frame, person, cv::Scalar(0, 255, 0), 2);  // Green rectangle, 2px thick
            person_id++;
        }
        // Add status overlay
        // Handle keyboard input
        imshow("360° Viewer", pano_frame);
        char key = cv::waitKey(1) & 0xFF;
        if (key == 27) break; // ESC to exit
    }
    
    cv::destroyAllWindows();
    return 0;
}
