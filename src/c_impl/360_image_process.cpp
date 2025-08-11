#include "360_image_process.h"

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

void PanoViewer::setYaw(float new_yaw) {
    yaw = new_yaw;
}

void PanoViewer::setPitch(float new_pitch) {
    pitch = new_pitch;
}

void PanoViewer::setFOV(float new_fov) {
    fov = new_fov;
}


