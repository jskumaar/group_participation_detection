/**
 * @file 360_image_process.h
 * @brief Header for panoramic image processing and gaze tracking
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <deque>
#include <vector>
#include <fstream>

class PanoViewer{
    public:
        struct gaze{
            cv::Point3f start;
            cv::Vec3f direction;
            int personID;
            cv::Rect2f box;
        };

        PanoViewer();
        ~PanoViewer();
        cv::Mat generatePerspectiveView(const cv::Mat& pano);
        void setYaw(float new_yaw);
        void setPitch(float new_pitch);
        void setFOV(float new_fov);
        float getFOV();
        float getYaw();
        float getPitch();

        int getOutputWidth() const { return output_width; }
        int getOutputHeight() const { return output_height; }

        void setMaxAngle(float angle_deg) { max_angle_threshold = angle_deg; }

        float computeFOVForPersonBox(const cv::Rect& box, int pano_height, int h_star_pixels, float r_head = 0.15f, float deg_min = 20.f, float deg_max = 110.f) const;

        gaze addGaze(float cam_yaw, float cam_pitch, float cam_fov, float pose_yaw, float pose_pitch, cv::Vec3f position);
        cv::Rect convertPerspectiveRectToEquirectangular(const cv::Rect& perspective_box, int pano_width, int pano_height) const;

        void gaze_analysis(std::vector<gaze>& gazes);
        void saveGazeAnalysis(std::ofstream& csv_file, long frame_number, const std::vector<gaze>& gazes);
        bool isLookingAt(PanoViewer::gaze gaze, cv::Point3f facePos, float maxAngleDeg);

    private:
        cv::Point2f sphericalToEquirectangular(float theta, float phi, int pano_width, int pano_height);

        float yaw;
        float pitch;
        float fov;
        bool needs_rebuild;

        float max_angle_threshold = 20.0f;

        const int output_width = 640;
        const int output_height = 480;
        float aspect_ratio = (float)output_width / output_height;

        const float PI = 3.14159265359f;
        const float DEG_TO_RAD = PI / 180.0f;
        const float RAD_TO_DEG = 180.0f / PI;

        cv::Mat map_x, map_y;
        float cached_yaw = -999.0f;
        float cached_pitch = -999.0f;
        float cached_fov = -999.0f;
        int cached_width = -1;
        int cached_height = -1;
        int cached_pano_cols = -1;
        int cached_pano_rows = -1;
        void buildRemapTables(int pano_cols, int pano_rows);
};

