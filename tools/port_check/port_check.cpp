// Runs the real project code on the inputs in cases.txt and prints the results,
// so tools/port_check/compare.js can compare them with assets/js/model.js.
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <opencv2/core/quaternion.hpp>

#include "vision/360_image_process.h"
#include "vision/gaze_bayes_mapper.h"

namespace {

// Copied verbatim from vision/opnet_tracker.cpp, which needs ONNX Runtime to link.
cv::Quatf rotation_from_two_vectors(const cv::Vec3f& a, const cv::Vec3f& b) {
    const cv::Vec3f axis = a.cross(b);
    const float dot = a.dot(b);
    const float len = static_cast<float>(cv::norm(axis));
    cv::Vec3f normed_axis = axis / len;
    float angle = std::atan2(len, dot);
    if (!(std::isfinite(normed_axis[0]) && std::isfinite(normed_axis[1]) && std::isfinite(normed_axis[2]))) {
        angle = 0.f;
        normed_axis = cv::Vec3f{1.f, 0.f, 0.f};
    }
    return cv::Quatf::createFromAngleAxis(angle, normed_axis);
}

// The combination block from OPNetTracker::detect.
void combine(float head_yaw_deg, float head_pitch_deg, float gaze_yaw, float gaze_pitch, float confidence) {
    constexpr float kPi = 3.14159265358979323846f;
    constexpr float kAlpha = 0.4f;
    const float compensated_gaze_yaw_deg = gaze_yaw - (kAlpha * head_yaw_deg);
    const float compensated_gaze_pitch_deg = gaze_pitch - (kAlpha * head_pitch_deg);
    constexpr float kDegToRad = kPi / 180.f;
    const float l2cs_yaw_rad = compensated_gaze_yaw_deg * kDegToRad;
    const float l2cs_pitch_rad = compensated_gaze_pitch_deg * kDegToRad;
    cv::Vec3f head_dir = vision::dir_from_yaw_pitch(head_yaw_deg * kDegToRad, head_pitch_deg * kDegToRad);
    cv::Quatf q_head_standard = rotation_from_two_vectors(cv::Vec3f(0.f, 0.f, 1.f), head_dir);
    cv::Vec3f eye_dir = vision::dir_from_yaw_pitch(l2cs_yaw_rad, l2cs_pitch_rad);
    cv::Quatf q_eye_standard = rotation_from_two_vectors(cv::Vec3f(0.f, 0.f, 1.f), eye_dir);
    constexpr float kL2csConfMinPct = 20.f;
    constexpr float kL2csConfMaxPct = 40.f;
    const float conf_pct = confidence * 100.f;
    const float eye_weight = std::clamp((conf_pct - kL2csConfMinPct) / (kL2csConfMaxPct - kL2csConfMinPct), 0.f, 1.f);
    const cv::Quatf q_identity{1.f, 0.f, 0.f, 0.f};
    const cv::Quatf q_eye_weighted = cv::Quatf::nlerp(q_identity, q_eye_standard, eye_weight, cv::QUAT_ASSUME_UNIT);
    cv::Quatf q_gaze_standard = q_head_standard * q_eye_weighted;
    float combined_yaw_deg = 0.f, combined_pitch_deg = 0.f;
    vision::yaw_pitch_deg_from_rot_mat(q_gaze_standard.toRotMat3x3(cv::QUAT_ASSUME_UNIT), combined_yaw_deg, combined_pitch_deg);
    std::cout << "combine " << eye_weight << " " << combined_yaw_deg << " " << combined_pitch_deg << "\n";
}

PanoViewer::gaze readGaze(std::istream& in) {
    PanoViewer::gaze g;
    in >> g.personID >> g.start.x >> g.start.y >> g.start.z
       >> g.direction[0] >> g.direction[1] >> g.direction[2]
       >> g.box_projected.x >> g.box_projected.y >> g.box_projected.width >> g.box_projected.height;
    return g;
}

} // namespace

int main(int argc, char** argv) {
    std::ifstream in(argv[1]);
    std::cout << std::setprecision(9);
    std::string kind;
    while (in >> kind) {
        if (kind == "combine") {
            float a, b, c, d, e;
            in >> a >> b >> c >> d >> e;
            combine(a, b, c, d, e);
        } else if (kind == "addgaze") {
            float cy, cp, fov, y, p, x0, x1, x2;
            in >> cy >> cp >> fov >> y >> p >> x0 >> x1 >> x2;
            PanoViewer v;
            auto g = v.addGaze(cy, cp, fov, y, p, cv::Vec3f(x0, x1, x2));
            std::cout << "addgaze " << g.start.x << " " << g.start.y << " " << g.start.z << " "
                      << g.direction[0] << " " << g.direction[1] << " " << g.direction[2] << "\n";
        } else if (kind == "rect") {
            float yaw, pitch, fov;
            int x, y, w, h, pw, ph;
            in >> yaw >> pitch >> fov >> x >> y >> w >> h >> pw >> ph;
            PanoViewer v;
            v.setYaw(yaw);
            v.setPitch(pitch);
            v.setFOV(fov);
            cv::Rect r = v.convertPerspectiveRectToEquirectangular(cv::Rect(x, y, w, h), pw, ph);
            std::cout << "rect " << r.x << " " << r.y << " " << r.width << " " << r.height << "\n";
        } else if (kind == "fov") {
            int bh, ph, hs;
            in >> bh >> ph >> hs;
            PanoViewer v;
            std::cout << "fov " << v.computeFOVForPersonBox(cv::Rect(0, 0, 10, bh), ph, hs) << "\n";
        } else if (kind == "bayes") {
            int n;
            vision::GazeBayesConfig cfg;
            in >> n >> cfg.sigma_x_deg >> cfg.sigma_y_deg >> cfg.omega_base >> cfg.omega_motion >> cfg.omega_null >> cfg.likelihood_null_baseline;
            std::vector<PanoViewer::gaze> f1, f2;
            for (int i = 0; i < n; ++i) f1.push_back(readGaze(in));
            for (int i = 0; i < n; ++i) f2.push_back(readGaze(in));
            vision::GazeBayesMapper mapper;
            mapper.infer(f1, cfg);
            auto edges = mapper.infer(f2, cfg);
            std::cout << "bayes";
            for (const auto& e : edges)
                std::cout << " " << e.from_person_id << " " << e.to_person_id << " " << e.angle_deg << " " << (e.is_looking ? 1 : 0) << " " << e.probability;
            std::cout << "\n";
        }
    }
    return 0;
}
