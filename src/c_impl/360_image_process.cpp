#include "360_image_process.h"
#include "SORT.h"

#define M_PI 3.14159265358979323846


PanoViewer::PanoViewer() {
    // Try to open camera input 1 (change to 0 if needed)
    yaw = 0.0f;
    pitch = 0.0f;
    fov = 90.0f;
}

PanoViewer::~PanoViewer() {
}
    
    // Convert spherical coordinates to equirectangular coordinates
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

            // 1) Build camera ray (not yet normalized is OK, but we normalize for stability)
            float vx = screen_x;
            float vy = screen_y;
            float vz = fov_factor;

            float inv_len = 1.0f / sqrtf(vx*vx + vy*vy + vz*vz);
            vx *= inv_len; vy *= inv_len; vz *= inv_len;

            // 2) Apply pitch rotation about X axis first
            // y' = cos(pitch)*y + sin(pitch)*z
            // z' = sin(pitch)*y - cos(pitch)*z
            float y1 =  cp * vy + sp * vz;
            float z1 = -sp * vy + cp * vz;
            float x1 =  vx;

            // 3) Apply yaw rotation about Y axis
            // x'' =  cos(yaw)*x + sin(yaw)*z
            // z'' = -sin(yaw)*x + cos(yaw)*z
            float x2 =  cy * x1 + sy * z1;
            float z2 = -sy * x1 + cy * z1;
            float y2 =  y1;

            // 4) Convert direction vector -> lon/lat
            float lon = atan2f(x2, z2); // [-pi, pi]
            float lat = asinf(std::max(-1.0f, std::min(1.0f, y2))); // [-pi/2, pi/2]

            // 5) Map lon/lat -> equirectangular pixel coords
            // lon: -pi..pi maps to 0..pano_cols
            float u = (lon / (2.0f * (float)M_PI) + 0.5f) * pano_cols;

            // lat: +pi/2 (top) to -pi/2 (bottom)
            // If your pano uses the opposite convention, flip this sign.
            float v = (0.5f - (lat / (float)M_PI)) * pano_rows;

            // 6) Wrap horizontally, clamp vertically
            u = fmodf(u, (float)pano_cols);
            if (u < 0) u += pano_cols;

            if (v < 0.0f) v = 0.0f;
            if (v > (float)pano_rows - 1.0f) v = (float)pano_rows - 1.0f;

            map_x_row[x] = u;
            map_y_row[x] = v;
        }
    }
    // Update cache
    cached_yaw = yaw;
    cached_pitch = pitch;
    cached_fov = fov;
    cached_width = output_width;
    cached_height = output_height;
    cached_pano_cols = pano_cols;
    cached_pano_rows = pano_rows;
}

cv::Mat PanoViewer::generatePerspectiveView(const cv::Mat& pano) {
    // Rebuild lookup tables only if parameters changed significantly

    if (needs_rebuild || map_x.empty() || map_y.empty()) {
        buildRemapTables(pano.cols, pano.rows);
        needs_rebuild = false;
    }
    cv::Mat output;
    // Use OpenCV's optimized remap function with bilinear interpolation
    cv::remap(pano, output, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_WRAP);
    return output;
}

static inline cv::Vec3f unit(const cv::Vec3f& v) {
    float n = cv::norm(v);
    return (n > 0.f) ? (v / n) : cv::Vec3f(0,0,0);
}


// Local camera-style convention: z=forward, x=right, y=up
static inline cv::Vec3f dir_from_yaw_pitch(float yaw, float pitch) {
    float cp = std::cos(pitch), sp = std::sin(pitch);
    float cy = std::cos(yaw),   sy = std::sin(yaw);
    return unit(cv::Vec3f(sy * cp,  // x (right)
                          sp,       // y (up)
                          cy * cp   // z (forward)
    ));
}



/**
 * Convert local yaw/pitch (relative to person-facing-camera) to a global/world gaze direction.
 * - yaw, pitch in radians
 * - (u,v) is the person's pixel in a 2:1 equirect panorama of size W×H
 * - world_up is typically (0,1,0)
 */
cv::Vec3f global_gaze_from_panorama(float yaw, float pitch,
                                    cv::Vec3f cam_to_person,
                                    cv::Vec3f world_up = cv::Vec3f(0.f,1.f,0.f)) {


    // Zero-gaze (yaw=0,pitch=0) means looking at the camera plane
    cv::Vec3f forward = unit(-cam_to_person);  // person -> camera

    // Build a stable local basis {right, up, forward}
    if (std::abs(forward.dot(world_up)) > 0.999f) {
        world_up = cv::Vec3f(0.f, 0.f, 1.f);
        if (std::abs(forward.dot(world_up)) > 0.999f)
            world_up = cv::Vec3f(1.f, 0.f, 0.f);
    }
    cv::Vec3f right = unit(world_up.cross(forward));
    cv::Vec3f up    = unit(forward.cross(right));

    // Local gaze direction from yaw/pitch (unit)
    cv::Vec3f local = dir_from_yaw_pitch(yaw, pitch);

    // Rotate local -> world
    cv::Vec3f global = right * local[0] + up * local[1] + forward * local[2];
    return unit(global);
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
    worldPos[0] = cy * v1x + sy * v1z;   // X_world
    worldPos[1] = v1y;                   // Y_world (unchanged by yaw)
    worldPos[2] = -sy * v1x + cy * v1z;   // Z_world

    //get initial virtual camera ray
	cv::Vec3f start_direction = dir_from_yaw_pitch(cam_yaw * DEG_TO_RAD, cam_pitch * DEG_TO_RAD);



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



bool PanoViewer::isLookingAt(PanoViewer::gaze gaze, cv::Point3f facePos, float maxAngleDeg)
{
    cv::Vec3f v = facePos - gaze.start;
    float dist = std::sqrt(v.dot(v));

    cv::Vec3f vdir = v / dist;  // normalize
    float cosTheta = vdir.dot(gaze.direction);

    float maxAngleRad = maxAngleDeg * static_cast<float>(M_PI / 180.0);
    float cosMax = std::cos(maxAngleRad);

    return cosTheta >= cosMax;
}

void PanoViewer::gaze_analysis(std::vector<gaze>& gazes) {
    for (size_t i = 0; i < gazes.size(); ++i) {
        for (size_t j = 0; j < gazes.size(); ++j) {
            if (i == j) continue; // don't compare to self

            const auto& gi = gazes[i];
            const auto& gj = gazes[j];


            if (isLookingAt(gi, gj.start, 20)) {
                std::cout << "Person " << gi.personID
                    << " is looking at person " << gj.personID
                    << std::endl;
            }
        }
    }
}

float PanoViewer::computeFOVForPersonBox(const cv::Rect& box, int pano_height, int h_star_pixels, float r_head, float deg_min, float deg_max) const {
    // Person angular height (radians) ~ (h / H) * pi
    const double person_h = static_cast<double>(box.height);
    const double H = static_cast<double>(pano_height);
    const double alpha_person = (person_h / H) * M_PI;
    const double alpha_head = r_head * alpha_person; // head angular size

    // Output crop height (pixels)
    const double N = static_cast<double>(output_height);

    // Solve for theta: tan(theta/2) = N * tan(alpha_head/2) / h_star
    double denom = std::max(1e-6, static_cast<double>(h_star_pixels));
    double tan_half_theta = (N * std::tan(alpha_head * 0.5)) / denom;

    // clamp using deg_min/deg_max
    const double tan_min = std::tan((deg_min * M_PI/180.0) * 0.5);
    const double tan_max = std::tan((deg_max * M_PI/180.0) * 0.5);
    if (!std::isfinite(tan_half_theta) || tan_half_theta <= 0) tan_half_theta = tan_max;
    tan_half_theta = std::min(std::max(tan_half_theta, tan_min), tan_max);

    double theta = 2.0 * std::atan(tan_half_theta);
    float theta_deg = static_cast<float>(theta * 180.0 / M_PI);
    return theta_deg;
}

void PanoViewer::saveGazeAnalysis(std::ofstream& csv_file, long frame_number, const std::vector<gaze>& gazes) {
    if (!csv_file.is_open()) return;

    for (const auto& g : gazes) {
        std::string looking_at_ids = "";
        for (const auto& other : gazes) {
            if (g.personID == other.personID) continue;
            if (isLookingAt(g, other.start, 20)) {
                if (!looking_at_ids.empty()) looking_at_ids += ";";
                looking_at_ids += std::to_string(other.personID);
            }
        }

        csv_file << frame_number << ","
                 << g.personID << ","
                 << (g.box.x + g.box.width / 2.0f) << "," << (g.box.y + g.box.height / 2.0f) << ","
                 << g.start.x << "," << g.start.y << "," << g.start.z << ","
                 << g.direction[0] << "," << g.direction[1] << "," << g.direction[2] << ","
                 << "\"" << looking_at_ids << "\"\n";
    }
}

// Convert a bounding box from a specific perspective view back to the full equirectangular panorama
cv::Rect PanoViewer::convertPerspectiveRectToEquirectangular(const cv::Rect& perspective_box, int pano_width, int pano_height) const {
    
    // We will project the 4 corners of the perspective box
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
        // 1) Normalize to screen coords [-1, 1]
        // Note: Code mirrors buildRemapTables logic
        float screen_y = 1.0f - (2.0f * (pt.y + 0.5f) / (float)output_height);
        float screen_x = (2.0f * (pt.x + 0.5f) / (float)output_width) - 1.0f;
        screen_x *= aspect_ratio;

        // 2) Build camera ray
        float vx = screen_x;
        float vy = screen_y;
        float vz = fov_factor;

        float inv_len = 1.0f / sqrtf(vx*vx + vy*vy + vz*vz);
        vx *= inv_len; vy *= inv_len; vz *= inv_len;

        // 3) Apply pitch rotation (about X)
        float y1 =  cp * vy + sp * vz;
        float z1 = -sp * vy + cp * vz;
        float x1 =  vx;

        // 4) Apply yaw rotation (about Y)
        float x2 =  cy * x1 + sy * z1;
        float z2 = -sy * x1 + cy * z1;
        float y2 =  y1;

        // 5) Convert to spherical (lon, lat)
        float lon = atan2f(x2, z2); // [-pi, pi]
        float lat = asinf(std::max(-1.0f, std::min(1.0f, y2))); // [-pi/2, pi/2]

        // 6) Map to equirectangular pixel coords
        // lon: -pi..pi -> 0..pano_width
        float u = (lon / (2.0f * (float)M_PI) + 0.5f) * pano_width;
        
        // lat: +pi/2..-pi/2 -> 0..pano_height
        // Note: buildRemapTables used (0.5 - lat/pi) which maps +pi/2 to 0 (top)
        float v = (0.5f - (lat / (float)M_PI)) * pano_height;

        // 7) Wrap/Clamp
        u = fmodf(u, (float)pano_width);
        if (u < 0) u += pano_width;

        if (v < 0.0f) v = 0.0f;
        if (v > (float)pano_height - 1.0f) v = (float)pano_height - 1.0f;

        if (u < min_u) min_u = u;
        if (u > max_u) max_u = u;
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }

    // Handle wrapping case: if the box spans across the seam (e.g., wrap around 360/0)
    // Heuristic: if width > half the panorama, assume it wrapped.
    if ((max_u - min_u) > (pano_width / 2.0f)) {
        // Naive handling: usually implies the object is on the seam.
        // For purposes of returning a single Rect, this is tricky. 
        // We might return the union of the two sides or just one side.
        // Given this is for visualization/tracking, keeping the larger 'wrapped' box might be confusing visually 
        // as it would span the entire image.
        // A better approach for a rect might be to ensure min_u < max_u by adding pano_width to negative values 
        // before min/max finding, but 'u' is already normalized 0..width.
        // Let's leave as is for now unless seam artifacts become critical.
    }

    return cv::Rect((int)min_u, (int)min_v, (int)(max_u - min_u), (int)(max_v - min_v));
}


