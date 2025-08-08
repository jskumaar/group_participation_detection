#include <iostream>
#include <cmath>
#include <Eigen/Dense>



//contains conversions between pixel, spherical, and cartesian coords.
// also has angular distance calculations, bounding box intersections calculations, 
constexpr double PI = 3.14159265358979323846;

// === HELPERS ===

std::pair<double, double> pixel_to_spherical(int x, int y, int W, int H) {
    double lon = (static_cast<double>(x) / W) * 2 * PI - PI;      // [-π, π]
    double lat = PI / 2 - (static_cast<double>(y) / H) * PI;      // [π/2, -π/2]
    return {lon, lat};
}

Vector3d spherical_to_cartesian(double lon, double lat, double radius = 1.0) {
    double x = radius * std::cos(lat) * std::cos(lon);
    double y = radius * std::cos(lat) * std::sin(lon);
    double z = radius * std::sin(lat);
    return Vector3d(x, y, z);
}

std::pair<double, double> cartesian_to_spherical(const Vector3d& vec) {
    double x = vec.x();
    double y = vec.y();
    double z = vec.z();
    double norm = vec.norm();
    double lon = std::atan2(y, x);
    double lat = std::asin(z / norm);
    return {lon, lat};
}

std::pair<int, int> spherical_to_pixel(double lon, double lat, int W, int H) {
    int x = static_cast<int>((lon + PI) / (2 * PI) * W);
    int y = static_cast<int>((PI / 2 - lat) / PI * H);
    return {x, y};
}

Vector3d gaze_vector_from_yaw_pitch(double yaw_rad, double pitch_rad) {
    // Standard camera frame: z = forward, x = right, y = up
    double x = std::sin(yaw_rad) * std::cos(pitch_rad);
    double y = std::sin(pitch_rad);
    double z = std::cos(yaw_rad) * std::cos(pitch_rad);
    return Vector3d(x, y, z);
}

Vector3d local_to_global_gaze(double opentrack_yaw, double opentrack_pitch, double person_longitude, double person_latitude) {
    // 1. Person's position on sphere (radius = 1)
    Vector3d person_pos = spherical_to_cartesian(person_longitude, person_latitude);

    // 2. Person’s forward vector (looking toward sphere center)
    Vector3d forward = -person_pos.normalized();

    // 3. World "up" vector (global Y)
    Vector3d world_up(0, 1, 0);

    // 4. Person’s right vector
    Vector3d right = world_up.cross(forward).normalized();

    // 5. Person’s true up vector
    Vector3d up = forward.cross(right);

    // 6. Rotation matrix from local to world
    Matrix3d R;
    R.col(0) = right;    // local X
    R.col(1) = up;       // local Y
    R.col(2) = forward;  // local Z

    // 7. Local gaze vector from yaw/pitch
    Vector3d local_gaze = gaze_vector_from_yaw_pitch(opentrack_yaw, opentrack_pitch);

    // 8. Transform to world coordinates
    Vector3d global_gaze = R * local_gaze;
    return global_gaze.normalized();
}

double angular_distance(const Vector3d& vec1, const Vector3d& vec2) {
    double dot = vec1.dot(vec2);
    dot = std::clamp(dot, -1.0, 1.0);
    return std::acos(dot);  // radians
}














int main() {

    return 0;
}
