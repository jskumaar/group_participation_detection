#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

class PanoViewer{
    public:
        PanoViewer();
        ~PanoViewer();
        cv::Mat generatePerspectiveView(const cv::Mat& pano);
        void setYaw(float new_yaw);
        void setPitch(float new_pitch);
        void setFOV(float new_fov);
    private:

        //functions
        cv::Point2f sphericalToEquirectangular(float theta, float phi, int pano_width, int pano_height);
        
        // View parameters
        float yaw;        // Horizontal rotation (left/right)
        float pitch;      // Vertical rotation (up/down)
        float fov;       // Field of view (zoom level)

        // Output dimensions (16:9 aspect ratio)
        const int output_width = 1280;
        const int output_height = 720;

        // Conversion constants
        const float PI = 3.14159265359f;
        const float DEG_TO_RAD = PI / 180.0f;

};