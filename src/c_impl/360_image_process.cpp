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
    
// Generate perspective view from panoramic image
cv::Mat PanoViewer::generatePerspectiveView(const cv::Mat& pano) {
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

    // Zero-gaze (yaw=0,pitch=0) means looking at the camera
    cv::Vec3f forward = unit(-cam_to_person);  // person -> camera

    // Build a stable local basis {right, up, forward}
    if (std::abs(forward.dot(world_up)) > 0.999f) {
        printf("weird stuff going on\n");
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

cv::Point PanoViewer::rayToPanoPixel(const cv::Vec3f& p, const cv::Vec3f& d_unit,
                       float R, int W, int H)
{
    cv::Point2f out = {-1, -1};
    // Quadratic coefficients for |p + t d|^2 = R^2  with |d|=1
    float b = p.dot(d_unit);                  // half of the b term in quadratic
    float c = p.dot(p) - R*R;
    float disc = b*b - c;                     // discriminant

    if (disc < 0.f) return out;               // no intersection

    float s = std::sqrt(std::max(0.f, disc));
    float t1 = -b - s;                        // near intersection
    float t2 = -b + s;                        // far intersection

    // choose the farthest intersection that's forward along the ray
    float t_hit = -std::numeric_limits<float>::infinity();
    if (t1 >= 0.f) t_hit = t1;
    if (t2 >= 0.f && t2 > t_hit) t_hit = t2;
    if (!std::isfinite(t_hit) || t_hit < 0.f) return out; // no forward hit

    // Intersection point
    cv::Vec3f P = p + t_hit * d_unit;
    // Project exactly onto sphere to avoid FP error
    P *= (R / cv::norm(P));

    // Convert to pano pixel coordinates (equirectangular 2:1)
    float yaw   = std::atan2(P[0], P[2]);      // -π to π
    float pitch = std::asin(P[1] / R);         // -π/2 to π/2

    float u = (yaw + static_cast<float>(M_PI)) / (2.f * static_cast<float>(M_PI)) * W;
    float v = (0.5f - pitch / static_cast<float>(M_PI)) * H;

    // Wrap horizontally
    if (u < 0.f) u += W * std::ceil(-u / W);
    if (u >= W)  u -= W * std::floor(u / W);

    return cv::Point(u, v);
}

PanoViewer::gaze PanoViewer::addGaze(int personID, float yaw, float pitch, cv::Rect bounding_box) {
    gaze new_gaze;
    new_gaze.personID = personID;
    int centerX = bounding_box.x + bounding_box.width / 2;
    int centerY = bounding_box.y + bounding_box.height / 2;
    cv::Vec3f start_direction = pano_pixel_to_direction(centerX, centerY, GLOBAL_FRAME_WIDTH, GLOBAL_FRAME_HEIGHT);
    new_gaze.direction = global_gaze_from_panorama(yaw * DEG_TO_RAD, pitch * DEG_TO_RAD, start_direction);
    start_direction *= radius;
    new_gaze.start = cv::Point3f(start_direction[0], start_direction[1], start_direction[2]);
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



inline cv::Point2d pano_to_spherical(const cv::Point& pixel) {
    // Normalize [0, W] → [0, 1], then scale to [-π, π]
    double lon = (static_cast<double>(pixel.x) /PanoViewer::getGlobalWidth()) * 2.0 * M_PI - 1.0 * M_PI;

    // Normalize [0, H] → [0, 1], then scale to [-π/2, π/2]
    double lat = (0.5 -static_cast<double>(pixel.y) / PanoViewer::getGlobalHeight())* M_PI;

    return {lon, lat};  // (longitude, latitude)
}






// Returns -1 if gaze not intersecting any bounding box
int PanoViewer::bbox_intersections(const cv::Point& gaze_pt, const cv::Point& start_pt,
                                   const std::vector<Sort::Track>& bounding_boxes)
{
    // Convert gaze point to spherical coordinates
    cv::Point2d gaze_sph = pano_to_spherical(gaze_pt);

    
    double scale = 1.2; // scale bounding box by 20%
    for (const Sort::Track& t : bounding_boxes) {
        // Skip self. -1 since track ids start at 1
        if (cv::Point(t.box.x, t.box.y) == start_pt){
            continue;
        }
        cv::Rect bbox = t.box;
        // Compute scaled bounding box
        cv::Point center(bbox.x + bbox.width/2, bbox.y + bbox.height/2);
        int scaled_w = static_cast<int>(bbox.width * scale);
        int scaled_h = static_cast<int>(bbox.height * scale);
        cv::Point top_left(center.x - scaled_w/2, center.y - scaled_h/2);
        cv::Point bot_right(center.x + scaled_w/2, center.y + scaled_h/2);

        // Convert corners to spherical coordinates
        cv::Point2d top_left_sph  = pano_to_spherical(top_left);
        cv::Point2d bot_right_sph = pano_to_spherical(bot_right);

        double min_lon = std::min(top_left_sph.x, bot_right_sph.x);
        double max_lon = std::max(top_left_sph.x, bot_right_sph.x);
        double min_lat = std::min(top_left_sph.y, bot_right_sph.y);
        double max_lat = std::max(top_left_sph.y, bot_right_sph.y);

        // Check if gaze lies inside spherical box
        if (gaze_sph.x >= min_lon && gaze_sph.x <= max_lon && 
            gaze_sph.y >= min_lat && gaze_sph.y <= max_lat) {
            return t.id-1;
        }
    }

    return -1;
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