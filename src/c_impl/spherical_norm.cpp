#include <cmath>
#include <opencv2/opencv.hpp>

// Convert bbox center (pixel) to spherical coordinates (longitude, latitude)
std::pair<float, float> bbox_to_spherical_coord(float bbox_center_x, float bbox_center_y, int frame_width, int frame_height) {
    float u = bbox_center_x / frame_width;
    float v = bbox_center_y / frame_height;

    float longitude = (u * 2.0f * CV_PI) - CV_PI;        // [-π, π]
    float latitude = (CV_PI / 2.0f) - (v * CV_PI);      // [π/2, -π/2]

    return {longitude, latitude};
}

// Convert spherical to Cartesian coordinates (radius assumed)
cv::Vec3f spherical_to_cartesian(float longitude, float latitude, float radius) {
    float x = radius * cos(latitude) * cos(longitude);
    float y = radius * cos(latitude) * sin(longitude);
    float z = radius * sin(latitude);
    return cv::Vec3f(x, y, z);
}

// Rotate vector by yaw (around Z) and pitch (around X)
cv::Vec3f rotate_vector(const cv::Vec3f& vector, float yaw, float pitch) {
    // Rotation matrix for yaw (Z axis)
    cv::Matx33f R_yaw(
        cos(yaw), -sin(yaw), 0,
        sin(yaw),  cos(yaw), 0,
        0,        0,         1
    );

    // Rotation matrix for pitch (X axis)
    cv::Matx33f R_pitch(
        1, 0,           0,
        0, cos(pitch), -sin(pitch),
        0, sin(pitch),  cos(pitch)
    );

    // Combine rotations: pitch then yaw (R_yaw * R_pitch)
    cv::Matx33f R = R_yaw * R_pitch;

    cv::Vec3f rotated = R * vector;
    return rotated;
}

// Convert local gaze (yaw, pitch in degrees) and person position (longitude, latitude) 
// to global gaze direction vector
cv::Vec3f local_to_global_gaze(float opentrack_yaw, float opentrack_pitch, float person_longitude, float person_latitude) {
    // Person position on unit sphere
    cv::Vec3f person_pos = spherical_to_cartesian(person_longitude, person_latitude, 1.0f);

    // Base gaze direction is inward (towards origin)
    cv::Vec3f base_gaze_dir = -person_pos / cv::norm(person_pos);

    // Convert degrees to radians
    float yaw_rad = opentrack_yaw * CV_PI / 180.0f;
    float pitch_rad = opentrack_pitch * CV_PI / 180.0f;

    // Rotate gaze vector by yaw then pitch
    return rotate_vector(base_gaze_dir, yaw_rad, pitch_rad);
}

// Draw arrow from start_pt to end_pt (end_pt is a 2D point here)
void draw_arrow_global(cv::Mat& image, const cv::Point& start_pt, const cv::Point& end_pt, const cv::Scalar& color) {
    cv::arrowedLine(image, start_pt, end_pt, color, 2);
}

// Cartesian to spherical (longitude, latitude)
std::pair<float, float> cartesian_to_spherical(float x, float y, float z) {
    float radius = std::sqrt(x*x + y*y + z*z);
    float longitude = std::atan2(y, x);    // [-π, π]
    float latitude = std::asin(z / radius); // [-π/2, π/2]
    return {longitude, latitude};
}

// Spherical (longitude, latitude) to 2D pixel coordinates
cv::Point spherical_to_2d(float longitude, float latitude, int frame_width, int frame_height) {
    // Clamp
    if (longitude < -CV_PI) longitude = -CV_PI;
    else if (longitude > CV_PI) longitude = CV_PI;

    if (latitude < -CV_PI/2) latitude = -CV_PI/2;
    else if (latitude > CV_PI/2) latitude = CV_PI/2;

    float u = (longitude + CV_PI) / (2 * CV_PI);
    float v = (CV_PI / 2 - latitude) / CV_PI;

    int x = std::round(u * frame_width);
    int y = std::round(v * frame_height);

    // Clamp to frame bounds
    x = std::max(0, std::min(frame_width - 1, x));
    y = std::max(0, std::min(frame_height - 1, y));

    return cv::Point(x, y);
}
