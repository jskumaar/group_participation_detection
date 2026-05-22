#include "vision/360_image_process.h"

#include "vision/sort_tracker.h"

#define M_PI 3.14159265358979323846

PanoViewer::PanoViewer() {
    yaw = 0.0f;
    pitch = 0.0f;
    fov = 90.0f;
}

PanoViewer::~PanoViewer() {
}

cv::Point2f PanoViewer::sphericalToEquirectangular(float theta, float phi, int pano_width, int pano_height) {
    float x = (theta / (2 * M_PI) + 0.5f) * pano_width;
    float y = (phi / M_PI + 0.5f) * pano_height;
    return cv::Point2f(x, y);
}

void PanoViewer::buildRemapTables(int pano_cols, int pano_rows) {
    map_x.create(output_height, output_width, CV_32FC1);
    map_y.create(output_height, output_width, CV_32FC1);
    float aspect_ratio = (float)output_width / output_height;
    float fov_rad = fov * DEG_TO_RAD;
    float yaw_rad = yaw * DEG_TO_RAD;
    float pitch_rad = pitch * DEG_TO_RAD;
    float fov_factor = 1.0f / tan(fov_rad * 0.5f);

    float cy = cosf(yaw_rad),   sy = sinf(yaw_rad);
    float cp = cosf(pitch_rad), sp = sinf(pitch_rad);

    for (int y = 0; y < output_height; y++) {
        float* map_x_row = map_x.ptr<float>(y);
        float* map_y_row = map_y.ptr<float>(y);
        float screen_y = 1.0f - (2.0f * (y + 0.5f) / (float)output_height);
        for (int x = 0; x < output_width; x++) {
            float screen_x = (2.0f * (x + 0.5f) / (float)output_width) - 1.0f;
            screen_x *= aspect_ratio;

            float vx = screen_x;
            float vy = screen_y;
            float vz = fov_factor;

            float inv_len = 1.0f / sqrtf(vx*vx + vy*vy + vz*vz);
            vx *= inv_len; vy *= inv_len; vz *= inv_len;

            float y1 =  cp * vy + sp * vz;
            float z1 = -sp * vy + cp * vz;
            float x1 =  vx;

            float x2 =  cy * x1 + sy * z1;
            float z2 = -sy * x1 + cy * z1;
            float y2 =  y1;

            float lon = atan2f(x2, z2);
            float lat = asinf(std::max(-1.0f, std::min(1.0f, y2)));

            float u = (lon / (2.0f * (float)M_PI) + 0.5f) * pano_cols;
            float v = (0.5f - (lat / (float)M_PI)) * pano_rows;

            u = fmodf(u, (float)pano_cols);
            if (u < 0) u += pano_cols;

            if (v < 0.0f) v = 0.0f;
            if (v > (float)pano_rows - 1.0f) v = (float)pano_rows - 1.0f;

            map_x_row[x] = u;
            map_y_row[x] = v;
        }
    }

    cached_yaw = yaw;
    cached_pitch = pitch;
    cached_fov = fov;
    cached_width = output_width;
    cached_height = output_height;
    cached_pano_cols = pano_cols;
    cached_pano_rows = pano_rows;
}

cv::Mat PanoViewer::generatePerspectiveView(const cv::Mat& pano) {
    if (needs_rebuild || map_x.empty() || map_y.empty()) {
        buildRemapTables(pano.cols, pano.rows);
        needs_rebuild = false;
    }
    if (perspective_output_.rows != output_height || perspective_output_.cols != output_width ||
        perspective_output_.type() != pano.type())
    {
        perspective_output_.create(output_height, output_width, pano.type());
    }
    cv::remap(pano, perspective_output_, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_WRAP);
    return perspective_output_;
}

namespace vision {

namespace {

cv::Vec3f unit(const cv::Vec3f& v) {
    const float n = static_cast<float>(cv::norm(v));
    return (n > 0.f) ? (v / n) : cv::Vec3f(0, 0, 0);
}

} // namespace

cv::Vec3f dir_from_yaw_pitch(float yaw, float pitch) {
    const float cp = std::cos(pitch);
    const float sp = std::sin(pitch);
    const float cy = std::cos(yaw);
    const float sy = std::sin(yaw);
    return unit(cv::Vec3f(sy * cp, sp, cy * cp));
}

void yaw_pitch_deg_from_rot_mat(const cv::Matx33f &R, float &yaw_deg, float &pitch_deg)
{
    const auto fwd = R.col(2);
    const float yaw = std::atan2(fwd(0), fwd(2));
    const float pitch = std::asin(std::max(-1.f, std::min(1.f, fwd(1))));
    constexpr float kRadToDeg = 180.f / static_cast<float>(M_PI);
    yaw_deg = yaw * kRadToDeg;
    pitch_deg = pitch * kRadToDeg;
}

} // namespace vision

namespace {

cv::Vec3f unit_vec3f(const cv::Vec3f& v) {
    const float n = static_cast<float>(cv::norm(v));
    return (n > 0.f) ? (v / n) : cv::Vec3f(0, 0, 0);
}

} // namespace

cv::Vec3f global_gaze_from_panorama(float yaw, float pitch,
                                    cv::Vec3f cam_to_person,
                                    cv::Vec3f world_up = cv::Vec3f(0.f,1.f,0.f)) {

    cv::Vec3f forward = unit_vec3f(-cam_to_person);

    if (std::abs(forward.dot(world_up)) > 0.999f) {
        world_up = cv::Vec3f(0.f, 0.f, 1.f);
        if (std::abs(forward.dot(world_up)) > 0.999f)
            world_up = cv::Vec3f(1.f, 0.f, 0.f);
    }
    cv::Vec3f right = unit_vec3f(world_up.cross(forward));
    cv::Vec3f up    = unit_vec3f(forward.cross(right));

    cv::Vec3f local = vision::dir_from_yaw_pitch(yaw, pitch);

    cv::Vec3f global = right * local[0] + up * local[1] + forward * local[2];
    return unit_vec3f(global);
}

PanoViewer::gaze PanoViewer::addGaze(float cam_yaw, float cam_pitch, float cam_fov, float yaw, float pitch, cv::Vec3f position) {
    gaze new_gaze;

    float cy = std::cos(cam_yaw*DEG_TO_RAD);
    float sy = std::sin(cam_yaw * DEG_TO_RAD);
    float cp = std::cos(cam_pitch*DEG_TO_RAD);
    float sp = std::sin(cam_pitch * DEG_TO_RAD);

    float x = position[0];
    float y = position[1];
    float z = position[2];

    float v1x = x;
    float v1y = cp * y - sp * z;
    float v1z = sp * y + cp * z;

    cv::Vec3f worldPos;
    worldPos[0] = cy * v1x + sy * v1z;
    worldPos[1] = v1y;
    worldPos[2] = -sy * v1x + cy * v1z;

    cv::Vec3f start_direction = vision::dir_from_yaw_pitch(cam_yaw * DEG_TO_RAD, cam_pitch * DEG_TO_RAD);

    new_gaze.direction = global_gaze_from_panorama(yaw * DEG_TO_RAD, pitch * DEG_TO_RAD, start_direction);
    new_gaze.start = cv::Point3f(worldPos[0], worldPos[1], worldPos[2]);
    return new_gaze;
}

void PanoViewer::setYaw(float new_yaw) {
    yaw = new_yaw;
    needs_rebuild = true;
}

void PanoViewer::setPitch(float new_pitch) {
    pitch = new_pitch;
    needs_rebuild = true;
}

void PanoViewer::setFOV(float new_fov) {
    fov = new_fov;
    needs_rebuild = true;
}

float PanoViewer::getFOV() {
    return fov;
}
float PanoViewer::getYaw() {
    return yaw;
}
float PanoViewer::getPitch() {
    return pitch;
}

float PanoViewer::computeFOVForPersonBox(const cv::Rect& box, int pano_height, int h_star_pixels, float r_head, float deg_min, float deg_max) const {
    const double person_h = static_cast<double>(box.height);
    const double H = static_cast<double>(pano_height);
    const double alpha_person = (person_h / H) * M_PI;
    const double alpha_head = r_head * alpha_person;

    const double N = static_cast<double>(output_height);

    double denom = std::max(1e-6, static_cast<double>(h_star_pixels));
    double tan_half_theta = (N * std::tan(alpha_head * 0.5)) / denom;

    const double tan_min = std::tan((deg_min * M_PI/180.0) * 0.5);
    const double tan_max = std::tan((deg_max * M_PI/180.0) * 0.5);
    if (!std::isfinite(tan_half_theta) || tan_half_theta <= 0) tan_half_theta = tan_max;
    tan_half_theta = std::min(std::max(tan_half_theta, tan_min), tan_max);

    double theta = 2.0 * std::atan(tan_half_theta);
    float theta_deg = static_cast<float>(theta * 180.0 / M_PI);
    return theta_deg;
}

cv::Rect PanoViewer::convertPerspectiveRectToEquirectangular(const cv::Rect& perspective_box, int pano_width, int pano_height) const {

    std::vector<cv::Point2f> corners;
    corners.push_back(cv::Point2f(perspective_box.x, perspective_box.y));
    corners.push_back(cv::Point2f(perspective_box.x + perspective_box.width, perspective_box.y));
    corners.push_back(cv::Point2f(perspective_box.x + perspective_box.width, perspective_box.y + perspective_box.height));
    corners.push_back(cv::Point2f(perspective_box.x, perspective_box.y + perspective_box.height));

    float aspect_ratio = (float)output_width / output_height;
    float fov_rad = fov * DEG_TO_RAD;
    float yaw_rad = yaw * DEG_TO_RAD;
    float pitch_rad = pitch * DEG_TO_RAD;
    float fov_factor = 1.0f / tan(fov_rad * 0.5f);

    float cy = cosf(yaw_rad), sy = sinf(yaw_rad);
    float cp = cosf(pitch_rad), sp = sinf(pitch_rad);

    float min_u = 1e9, max_u = -1e9;
    float min_v = 1e9, max_v = -1e9;

    for (const auto& pt : corners) {
        float screen_y = 1.0f - (2.0f * (pt.y + 0.5f) / (float)output_height);
        float screen_x = (2.0f * (pt.x + 0.5f) / (float)output_width) - 1.0f;
        screen_x *= aspect_ratio;

        float vx = screen_x;
        float vy = screen_y;
        float vz = fov_factor;

        float inv_len = 1.0f / sqrtf(vx*vx + vy*vy + vz*vz);
        vx *= inv_len; vy *= inv_len; vz *= inv_len;

        float y1 =  cp * vy + sp * vz;
        float z1 = -sp * vy + cp * vz;
        float x1 =  vx;

        float x2 =  cy * x1 + sy * z1;
        float z2 = -sy * x1 + cy * z1;
        float y2 =  y1;

        float lon = atan2f(x2, z2);
        float lat = asinf(std::max(-1.0f, std::min(1.0f, y2)));

        float u = (lon / (2.0f * (float)M_PI) + 0.5f) * pano_width;
        float v = (0.5f - (lat / (float)M_PI)) * pano_height;

        u = fmodf(u, (float)pano_width);
        if (u < 0) u += pano_width;

        if (v < 0.0f) v = 0.0f;
        if (v > (float)pano_height - 1.0f) v = (float)pano_height - 1.0f;

        if (u < min_u) min_u = u;
        if (u > max_u) max_u = u;
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }

    if ((max_u - min_u) > (pano_width / 2.0f)) {
        // seam spanning case: left as-is
    }

    return cv::Rect((int)min_u, (int)min_v, (int)(max_u - min_u), (int)(max_v - min_v));
}

