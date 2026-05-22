/**
 * @file 360_image_process.h
 * @brief Header for panoramic image processing and gaze tracking
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/quaternion.hpp>
#include <cmath>
#include <deque>
#include <vector>

namespace vision {

/** Unit direction from yaw/pitch (radians); forward = (0,0,1) at zero. Used by PanoViewer. */
cv::Vec3f dir_from_yaw_pitch(float yaw_rad, float pitch_rad);

/** Yaw/pitch (degrees) from R's forward column (col 2), matching dir_from_yaw_pitch. */
void yaw_pitch_deg_from_rot_mat(const cv::Matx33f &R, float &yaw_deg, float &pitch_deg);

} // namespace vision

class PanoViewer{
    public:
        struct gaze{
            cv::Point3f start;
            cv::Vec3f direction;
            int personID;
            cv::Rect2f box;
            cv::Rect2f box_projected;
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

        float computeFOVForPersonBox(const cv::Rect& box, int pano_height, int h_star_pixels, float r_head = 0.15f, float deg_min = 20.f, float deg_max = 110.f) const;

        gaze addGaze(float cam_yaw, float cam_pitch, float cam_fov, float pose_yaw, float pose_pitch, cv::Vec3f position);
        cv::Rect convertPerspectiveRectToEquirectangular(const cv::Rect& perspective_box, int pano_width, int pano_height) const;

    private:
        cv::Point2f sphericalToEquirectangular(float theta, float phi, int pano_width, int pano_height);

        float yaw;
        float pitch;
        float fov;
        bool needs_rebuild;

        const int output_width = 640;
        const int output_height = 480;
        float aspect_ratio = (float)output_width / output_height;

        const float PI = 3.14159265359f;
        const float DEG_TO_RAD = PI / 180.0f;
        const float RAD_TO_DEG = 180.0f / PI;

        cv::Mat map_x, map_y;
        cv::Mat perspective_output_;
        float cached_yaw = -999.0f;
        float cached_pitch = -999.0f;
        float cached_fov = -999.0f;
        int cached_width = -1;
        int cached_height = -1;
        int cached_pano_cols = -1;
        int cached_pano_rows = -1;
        void buildRemapTables(int pano_cols, int pano_rows);
};

