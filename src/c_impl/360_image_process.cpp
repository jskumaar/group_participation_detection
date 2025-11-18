#include "360_image_process.h"
#include "SORT.h"


#define num_people 3;
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
    float x = (theta / (2 * PI) + 0.5f) * pano_width;
    float y = (phi / PI + 0.5f) * pano_height;
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
    for (int y = 0; y < output_height; y++) {
        float* map_x_row = map_x.ptr<float>(y);
        float* map_y_row = map_y.ptr<float>(y);
        float screen_y = (2.0f * y / output_height) - 1.0f;
        for (int x = 0; x < output_width; x++) {
            float screen_x = ((2.0f * x / output_width) - 1.0f) * aspect_ratio;
            float theta = atan2(screen_x, fov_factor);
            float phi = atan2(screen_y * cos(theta), fov_factor);
            float new_theta = theta + yaw_rad;
            float new_phi = std::max(-PI / 2.0f, std::min(PI / 2.0f, phi + pitch_rad));
            cv::Point2f eq_coord = sphericalToEquirectangular(new_theta, new_phi, pano_cols, pano_rows);
            // Handle horizontal wrapping
            while (eq_coord.x < 0) eq_coord.x += pano_cols;
            while (eq_coord.x >= pano_cols) eq_coord.x -= pano_cols;
            // Clamp vertical
            eq_coord.y = std::max(0.0f, std::min((float)pano_rows - 1, eq_coord.y));
            map_x_row[x] = eq_coord.x;
            map_y_row[x] = eq_coord.y;
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

cv::Mat PanoViewer::generatePerspectiveView(const cv::Mat& pano, bool needs_rebuild) {
    // Rebuild lookup tables only if parameters changed significantly

    if (needs_rebuild) {
        buildRemapTables(pano.cols, pano.rows);
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

// 2:1 equirectangular mapping -> direction FROM camera
// u ∈ [0, W), v ∈ [0, H)
// yaw (λ) ∈ (-π, π], pitch (φ) ∈ [-π/2, π/2]
static inline cv::Vec3f pano_pixel_to_direction(int u, int v, int W, int H) {
    float yaw   = ( (float)u / (float)W ) * 2.0f * (float)M_PI - (float)M_PI;        // -π..π
    float pitch = (0.5f - (float)v / (float)H) * (float)M_PI;                        // -π/2..π/2
    return dir_from_yaw_pitch(yaw, pitch); // matches our (z-forward, x-right, y-up) convention
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

std::pair<float, float> PanoViewer::apply_head_offset_correction(float yaw, float pitch, float fov, cv::Vec3f head_offset) {
    float nx = head_offset[0] / (output_width/ 2);   // ranges roughly -1..1
    float ny = head_offset[1] / (output_height/ 2);   // ranges roughly -1..1

    nx *= aspect_ratio;

    float theta = atan2(nx, 1.0f / tan(fov * 0.5f));
    float phi = atan2(ny * cos(theta), 1.0f / tan(fov * 0.5f));

	return { (yaw + theta) * RAD_TO_DEG, (pitch + phi) * RAD_TO_DEG };
}

cv::Vec3f compute_scaled_person_vector(cv::Vec3f scaled_cam, cv::Vec3f person_vector) {
    float dot_pu = scaled_cam.dot(person_vector);
    float dot_pp = scaled_cam.dot(scaled_cam);


    float t = dot_pp / dot_pu;   // scale for u
    cv::Vec3f x = person_vector * t;         // intersection point
    return x;
}

PanoViewer::gaze PanoViewer::addGaze(int personID, float cam_yaw, float cam_pitch, float cam_fov, float yaw, float pitch, cv::Vec3f position) {
    gaze new_gaze;
    new_gaze.personID = personID;


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
}

void PanoViewer::setPitch(float new_pitch) {
    pitch = new_pitch;
}

void PanoViewer::setFOV(float new_fov) {
    fov = new_fov;
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

PanoViewer::gaze_window& PanoViewer::add_frame_to_window(PanoViewer::gaze_window& gaze_win, int person_id, int intersect)
{
    if (gaze_win.sights.size() == gaze_win.max_window_length){
        int first_value = gaze_win.sights[0];
        gaze_win.sights.pop_front();
        if (first_value ==-1){
            gaze_win.counts[person_id]--;
        }
        else{
            gaze_win.counts[first_value] --;
        }
    } // only decrement counts and remove first val if size of sights has reached max size 

    gaze_win.sights.push_back(intersect);
    if (intersect==-1){ // not looking at andybody, add to count index of person_id to keep track 
        gaze_win.counts[person_id] ++;
    }

    else{
        gaze_win.counts[intersect]++;
    }

    return gaze_win;
}

void PanoViewer::print_gaze_window(PanoViewer::gaze_window& gaze_win, cv::Mat frame, cv::Rect bbox)
{
    if (gaze_win.sights.size() < gaze_win.max_window_length)
    {
        cv::putText(frame, "not enought data",  cv::Point(bbox.x-100, bbox.y-300), cv::FONT_HERSHEY_SIMPLEX,  2.0, cv::Scalar (255, 255, 255), 2);
        return;
    }

    int y_offset = 400;

    cv::putText(frame, "Over last  " + std::to_string(gaze_win.max_window_length) + " frames: ",  
                cv::Point(bbox.x, bbox.y-y_offset), cv::FONT_HERSHEY_SIMPLEX,  0.8, cv::Scalar (0, 0, 0), 2);



    y_offset-=100;
    for(int i = 0; i<gaze_win.counts.size(); i++){
        if (i== gaze_win.personID)
        {
            double pct_elsewhere = 100.0 * (double)(gaze_win.counts[gaze_win.personID]) / (double)(gaze_win.max_window_length);
            cv::putText(frame, "% looking elswhere: " + std::to_string(pct_elsewhere),  
                        cv::Point(bbox.x, bbox.y-y_offset), cv::FONT_HERSHEY_SIMPLEX,  0.8, cv::Scalar (0, 0, 0), 2);
        } 
        else
        {
            double pct = 100.0 * (double)(gaze_win.counts[i]) / (double)(gaze_win.max_window_length);
            cv::putText(frame, "%looking at person " + std::to_string(i) + " : " + std::to_string(pct), 
                        cv::Point(bbox.x, bbox.y-y_offset), cv::FONT_HERSHEY_SIMPLEX,  0.8, cv::Scalar (0, 0, 0), 2);
        }
        y_offset-=100;
    }

}

// New helper: compute FOV so head maps to h_star_pixels in output
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