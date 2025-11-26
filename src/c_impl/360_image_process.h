/**
 * @file 360_image_process.h
 * @brief Header for panoramic image processing and gaze tracking
 *
 * Defines the PanoViewer class for processing 360-degree equirectangular images,
 * generating perspective views, and computing gaze vectors for person tracking.
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <deque>
#include <vector>

class PanoViewer{
    public:
        struct gaze{
            cv::Point3f start;
            cv::Vec3f direction;
            int personID;
        };


        struct gaze_window{
            std::deque<int> sights; // queue of n frames, each with data holding a person_id at who this person is looking at 
            std::vector<int> counts; // an array of num_people size mapping index to num frames the person looks at another person with index id
            int personID;
            int max_window_length; // after hits this, start removing frames from front and add frame to back 
        };
        // window_length in frames, 



        PanoViewer();
        ~PanoViewer();
        cv::Mat generatePerspectiveView(const cv::Mat& pano, bool needs_rebuild);
        void setYaw(float new_yaw);
        void setPitch(float new_pitch);
        void setFOV(float new_fov);
        float getFOV();
		float getYaw();
		float getPitch();

        // New accessors for output dimensions
        int getOutputWidth() const { return output_width; }
        int getOutputHeight() const { return output_height; }

        // Compute a vertical FOV (degrees) so a person's head will be approximately
        // `h_star_pixels` high in the perspective crop. `box` is the person bbox in pano pixels
        // and pano_height is the panorama height in pixels.
        float computeFOVForPersonBox(const cv::Rect& box, int pano_height, int h_star_pixels, float r_head = 0.15f, float deg_min = 20.f, float deg_max = 110.f) const;

        //calc functions
        gaze addGaze(int personID, float cam_yaw, float cam_pitch, float cam_fov, float pose_yaw, float pose_pitch, cv::Vec3f position);

		void gaze_analysis(std::vector<gaze>& gazes);
        
    private:
        //functions
        std::pair<float, float> apply_head_offset_correction(float yaw, float pitch, float fov, cv::Vec3f head_offset);
        cv::Point2f sphericalToEquirectangular(float theta, float phi, int pano_width, int pano_height);

        
        // View parameters
        float yaw;        // Horizontal rotation (left/right)
        float pitch;      // Vertical rotation (up/down)
        float fov;       // Field of view (zoom level)

        // Output dimensions 
        const int output_width = 640;
        const int output_height = 480;
        float aspect_ratio = (float)output_width / output_height;

        // Conversion constants
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
        bool isLookingAt(PanoViewer::gaze gaze, cv::Point3f facePos, float maxAngleDeg);

};