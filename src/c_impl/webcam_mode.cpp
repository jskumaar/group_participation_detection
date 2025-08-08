#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include "tracker.h"

class PanoViewer {
private:
    cv::VideoCapture cap;
    cv::Mat pano_frame;
    OPNetTracker tracker;
    
    // View parameters
    float yaw = 0.0f;        // Horizontal rotation (left/right)
    float pitch = 0.0f;      // Vertical rotation (up/down)
    float fov = 90.0f;       // Field of view (zoom level)
    
    // Output dimensions (16:9 aspect ratio)
    int output_width = 1280;
    int output_height = 720;
    
    // Mouse control
    bool mouse_pressed = false;
    int last_mouse_x = 0;
    int last_mouse_y = 0;
    
    // Conversion constants
    const float PI = 3.14159265359f;
    const float DEG_TO_RAD = PI / 180.0f;




public:
    PanoViewer() {
        // Try to open camera input 1 (change to 0 if needed)
        cap.open(1);
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open camera input 1. Trying input 0..." << std::endl;
        }
        
        // Set camera properties for better quality
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 960);
        cap.set(cv::CAP_PROP_FPS, 30);
    }
    
    ~PanoViewer() {
        if (cap.isOpened()) {
            cap.release();
        }
    }
    
    // Convert spherical coordinates to equirectangular coordinates
    cv::Point2f sphericalToEquirectangular(float theta, float phi, int pano_width, int pano_height) {
        float x = (theta / (2 * PI) + 0.5f) * pano_width;
        float y = (phi / PI + 0.5f) * pano_height;
        return cv::Point2f(x, y);
    }
    
    // Generate perspective view from panoramic image
    cv::Mat generatePerspectiveView(const cv::Mat& pano) {
        cv::Mat output(output_height, output_width, CV_8UC3);
        
        float aspect_ratio = (float)output_width / output_height;
        float fov_rad = fov * DEG_TO_RAD;
        float yaw_rad = yaw * DEG_TO_RAD;
        float pitch_rad = pitch * DEG_TO_RAD;
        
        for (int y = 0; y < output_height; y++) {
            for (int x = 0; x < output_width; x++) {
                // Convert screen coordinates to normalized coordinates [-1, 1]
                float screen_x = (2.0f * x / output_width) - 1.0f;
                float screen_y = (2.0f * y / output_height) - 1.0f;
                
                // Apply aspect ratio correction
                screen_x *= aspect_ratio;
                
                // Convert to spherical coordinates
                float theta = atan2(screen_x, 1.0f / tan(fov_rad * 0.5f));
                float phi = atan2(screen_y * cos(theta), 1.0f / tan(fov_rad * 0.5f));
                
                // Apply rotation (yaw and pitch)
                float cos_pitch = cos(pitch_rad);
                float sin_pitch = sin(pitch_rad);
                float cos_yaw = cos(yaw_rad);
                float sin_yaw = sin(yaw_rad);
                
                // Rotate around Y axis (yaw)
                float new_theta = theta + yaw_rad;
                
                // Rotate around X axis (pitch)
                float new_phi = phi + pitch_rad;
                
                // Clamp phi to valid range
                new_phi = std::max(-PI/2.0f, std::min(PI/2.0f, new_phi));
                
                // Convert to equirectangular coordinates
                cv::Point2f eq_coord = sphericalToEquirectangular(new_theta, new_phi, pano.cols, pano.rows);
                
                // Handle wrapping for theta (horizontal)
                while (eq_coord.x < 0) eq_coord.x += pano.cols;
                while (eq_coord.x >= pano.cols) eq_coord.x -= pano.cols;
                
                // Clamp phi (vertical)
                eq_coord.y = std::max(0.0f, std::min((float)pano.rows - 1, eq_coord.y));
                
                // Sample from panoramic image using bilinear interpolation
                if (eq_coord.x >= 0 && eq_coord.x < pano.cols && 
                    eq_coord.y >= 0 && eq_coord.y < pano.rows) {
                    
                    // Bilinear interpolation for smoother results
                    int x1 = (int)floor(eq_coord.x);
                    int y1 = (int)floor(eq_coord.y);
                    int x2 = std::min(x1 + 1, pano.cols - 1);
                    int y2 = std::min(y1 + 1, pano.rows - 1);
                    
                    float fx = eq_coord.x - x1;
                    float fy = eq_coord.y - y1;
                    
                    // Handle horizontal wrapping for x coordinates
                    x1 = x1 % pano.cols;
                    x2 = x2 % pano.cols;
                    if (x1 < 0) x1 += pano.cols;
                    if (x2 < 0) x2 += pano.cols;
                    
                    // Get the four surrounding pixels
                    cv::Vec3b c00 = pano.at<cv::Vec3b>(y1, x1);
                    cv::Vec3b c10 = pano.at<cv::Vec3b>(y1, x2);
                    cv::Vec3b c01 = pano.at<cv::Vec3b>(y2, x1);
                    cv::Vec3b c11 = pano.at<cv::Vec3b>(y2, x2);
                    
                    // Interpolate
                    cv::Vec3b color;
                    for (int ch = 0; ch < 3; ch++) {
                        float top = c00[ch] * (1 - fx) + c10[ch] * fx;
                        float bottom = c01[ch] * (1 - fx) + c11[ch] * fx;
                        color[ch] = (uchar)(top * (1 - fy) + bottom * fy);
                    }
                    
                    output.at<cv::Vec3b>(y, x) = color;
                }
            }
        }
        
        return output;
    }


    
    // Mouse callback function
    // static void onMouse(int event, int x, int y, int flags, void* userdata) {
    //     PanoViewer* viewer = static_cast<PanoViewer*>(userdata);
    //     viewer->handleMouse(event, x, y, flags);
    // }
    
    // void handleMouse(int event, int x, int y, int flags) {
    //     switch (event) {
    //         case cv::EVENT_LBUTTONDOWN:
    //             mouse_pressed = true;
    //             last_mouse_x = x;
    //             last_mouse_y = y;
    //             break;
                
    //         case cv::EVENT_LBUTTONUP:
    //             mouse_pressed = false;
    //             break;
                
    //         case cv::EVENT_MOUSEMOVE:
    //             if (mouse_pressed) {
    //                 float dx = x - last_mouse_x;
    //                 float dy = y - last_mouse_y;
                    
    //                 // Adjust sensitivity
    //                 yaw -= dx * 0.5f;
    //                 pitch += dy * 0.5f;
                    
    //                 // Clamp pitch to prevent flipping
    //                 pitch = std::max(-89.0f, std::min(89.0f, pitch));
                    
    //                 // Normalize yaw
    //                 while (yaw > 180.0f) yaw -= 360.0f;
    //                 while (yaw < -180.0f) yaw += 360.0f;
                    
    //                 last_mouse_x = x;
    //                 last_mouse_y = y;
    //             }
    //             break;
                
    //         case cv::EVENT_MOUSEWHEEL:
    //             if (cv::getMouseWheelDelta(flags) > 0) {
    //                 fov = std::max(10.0f, fov - 5.0f);  // Zoom in
    //             } else {
    //                 fov = std::min(120.0f, fov + 5.0f); // Zoom out
    //             }
    //             break;
    //     }
    // }
    
    // void handleKeyboard(char key) {
    //     float pan_speed = 2.0f;
    //     float zoom_speed = 3.0f;
        
    //     switch (key) {
    //         case 'w': case 'W':
    //             pitch = std::max(-89.0f, pitch - pan_speed);
    //             break;
    //         case 's': case 'S':
    //             pitch = std::min(89.0f, pitch + pan_speed);
    //             break;
    //         case 'a': case 'A':
    //             yaw -= pan_speed;
    //             break;
    //         case 'd': case 'D':
    //             yaw += pan_speed;
    //             break;
    //         case 'q': case 'Q':
    //             fov = std::max(10.0f, fov - zoom_speed);
    //             break;
    //         case 'e': case 'E':
    //             fov = std::min(120.0f, fov + zoom_speed);
    //             break;
    //         case 'r': case 'R':
    //             // Reset view
    //             yaw = 0.0f;
    //             pitch = 0.0f;
    //             fov = 90.0f;
    //             break;
    //     }
        
    //     // Normalize yaw
    //     while (yaw > 180.0f) yaw -= 360.0f;
    //     while (yaw < -180.0f) yaw += 360.0f;
    // }
    
    void run() {
        if (!cap.isOpened()) {
            std::cerr << "Camera not available!" << std::endl;
            return;
        }
        
        // cv::setMouseCallback("360° Viewer", onMouse, this);
        
        while (true) {
            cap >> pano_frame;
            
            if (pano_frame.empty()) {
                std::cerr << "Failed to capture frame!" << std::endl;
                break;
            }
            
            // Ensure we have the right aspect ratio (2:1 for equirectangular)
            if (pano_frame.cols / (float)pano_frame.rows != 2.0f) {
                std::cout << "Warning: Input is not 2:1 aspect ratio. Current: " 
                         << pano_frame.cols << "x" << pano_frame.rows << std::endl;
            }
            
            auto people = tracker.detect_people_in_frame(pano_frame);
            cv::imshow("test", pano_frame);
            // Generate perspective view
            // int person_id = 0;  // Counter for unique window names
            // for(const cv::Rect& person : people){
            //     printf("coord: (%d, %d, %d, %d)\n", person.x, person.y, person.width, person.height);
            //     yaw = ((person.x + person.width / 2) * 0.125) - 180.0;
            //     cv::Mat perspective_view = generatePerspectiveView(pano_frame);
            //     rotation_output pose = tracker.run(perspective_view);
            //     printf("Person %d: ", person_id);
            //     printf("Yaw: %.2f, Pitch: %.2f, Roll: %.2f\n", pose.yaw, pose.pitch, pose.roll);
            //     std::string status = "Yaw: " + std::to_string((int)yaw) + 
            //         "° Pitch: " + std::to_string((int)pitch) + 
            //         "° FOV: " + std::to_string((int)fov) + "°";
            //     cv::putText(perspective_view, status, cv::Point(10, 30), 
            //             cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            //     // Create unique window name for each person
            //     std::string window_name = "Person " + std::to_string(person_id) + " - 360° Viewer";
            //     cv::imshow(window_name, perspective_view);
            //     person_id++;
            // }
            // Add status overlay
            // Handle keyboard input
            char key = cv::waitKey(1) & 0xFF;
            if (key == 27) break; // ESC to exit
            
            // handleKeyboard(key);
        }
        
        cv::destroyAllWindows();
    }
};

int main() {
    PanoViewer viewer;
    viewer.run();
    return 0;
}
