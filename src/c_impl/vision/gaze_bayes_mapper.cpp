#include "vision/gaze_bayes_mapper.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace vision {

namespace {

constexpr float kPi = 3.14159265359f;
constexpr float kDegToRad = kPi / 180.f;

cv::Vec3f unitVec(const cv::Vec3f& v) {
    const float n = static_cast<float>(cv::norm(v));
    if (n <= 1e-6f) return cv::Vec3f(0.f, 0.f, 0.f);
    return v / n;
}

cv::Point2f rectCenter(const cv::Rect2f& r) {
    return cv::Point2f(r.x + 0.5f * r.width, r.y + 0.5f * r.height);
}

float rectDiagonal(const cv::Rect2f& r) {
    return std::sqrt(r.width * r.width + r.height * r.height);
}

float clamp01(float v) {
    return std::max(0.f, std::min(1.f, v));
}

float motionFromHeadRect(const cv::Rect2f& rect, std::unordered_map<int, cv::Rect2f>& cache, int personId) {
    if (rect.width <= 0.f || rect.height <= 0.f) return 0.f;

    const cv::Point2f center = rectCenter(rect);
    const float scale = std::max(rectDiagonal(rect), 1.f);

    auto it = cache.find(personId);
    if (it == cache.end()) {
        cache[personId] = rect;
        return 0.f;
    }

    const cv::Point2f prevCenter = rectCenter(it->second);
    const float delta = static_cast<float>(cv::norm(center - prevCenter));
    it->second = rect;
    return clamp01(delta / scale);
}

void buildHeadFrame(const cv::Vec3f& forwardIn, cv::Vec3f& right, cv::Vec3f& up) {
    const cv::Vec3f forward = unitVec(forwardIn);
    if (cv::norm(forward) <= 1e-6f) {
        right = cv::Vec3f(1.f, 0.f, 0.f);
        up = cv::Vec3f(0.f, 1.f, 0.f);
        return;
    }

    cv::Vec3f worldUp(0.f, 1.f, 0.f);
    if (std::abs(forward.dot(worldUp)) > 0.999f) {
        worldUp = cv::Vec3f(0.f, 0.f, 1.f);
    }
    right = unitVec(worldUp.cross(forward));
    up = unitVec(forward.cross(right));
}

float anisotropicLikelihood(
    const cv::Vec3f& forward,
    const cv::Vec3f& right,
    const cv::Vec3f& up,
    const cv::Point3f& origin,
    const cv::Point3f& target,
    float sigmaXRad,
    float sigmaYRad) {
    cv::Vec3f toTarget(
        target.x - origin.x,
        target.y - origin.y,
        target.z - origin.z);
    toTarget = unitVec(toTarget);
    if (cv::norm(toTarget) <= 1e-6f) {
        return 0.f;
    }

    const float lx = toTarget.dot(right);
    const float ly = toTarget.dot(up);
    const float lz = toTarget.dot(forward);
    if (lz <= 1e-4f) {
        return 0.f;
    }

    const float thetaX = std::atan2(lx, lz);
    const float thetaY = std::atan2(ly, lz);
    const float exponent = (thetaX * thetaX) / (2.f * sigmaXRad * sigmaXRad)
        + (thetaY * thetaY) / (2.f * sigmaYRad * sigmaYRad);
    return std::exp(-exponent);
}

float angleBetweenDeg(const cv::Vec3f& gazeDirection, const cv::Point3f& origin, const cv::Point3f& target) {
    cv::Vec3f toTarget(target.x - origin.x, target.y - origin.y, target.z - origin.z);
    const cv::Vec3f gazeDir = unitVec(gazeDirection);
    toTarget = unitVec(toTarget);
    if (cv::norm(gazeDir) <= 1e-6f || cv::norm(toTarget) <= 1e-6f) {
        return 180.f;
    }

    float cosine = gazeDir.dot(toTarget);
    cosine = std::max(-1.f, std::min(1.f, cosine));
    return std::acos(cosine) * (180.f / kPi);
}

void softmaxInPlace(std::vector<float>& values) {
    if (values.empty()) return;
    const float maxV = *std::max_element(values.begin(), values.end());
    float sum = 0.f;
    for (float& v : values) {
        v = std::exp(v - maxV);
        sum += v;
    }
    if (sum <= 1e-12f) {
        const float uniform = 1.f / static_cast<float>(values.size());
        for (float& v : values) v = uniform;
        return;
    }
    for (float& v : values) v /= sum;
}

} // namespace

void GazeBayesMapper::reset() {
    prev_box_projected_.clear();
}

std::vector<domain::InteractionPair> GazeBayesMapper::infer(
    const std::vector<PanoViewer::gaze>& gazes,
    const GazeBayesConfig& cfg) {
    std::vector<domain::InteractionPair> interactions;
    if (gazes.size() < 2) return interactions;

    const float sigmaXRad = cfg.sigma_x_deg * kDegToRad;
    const float sigmaYRad = cfg.sigma_y_deg * kDegToRad;

    std::unordered_set<int> activeIds;
    activeIds.reserve(gazes.size());
    for (const auto& g : gazes) activeIds.insert(g.personID);

    for (auto it = prev_box_projected_.begin(); it != prev_box_projected_.end();) {
        if (activeIds.count(it->first) == 0) {
            it = prev_box_projected_.erase(it);
        } else {
            ++it;
        }
    }

    std::unordered_map<int, float> motionByPerson;
    motionByPerson.reserve(gazes.size());
    for (const auto& g : gazes) {
        motionByPerson[g.personID] = motionFromHeadRect(g.box_projected, prev_box_projected_, g.personID);
    }

    interactions.reserve(gazes.size() * (gazes.size() - 1));

    for (size_t i = 0; i < gazes.size(); ++i) {
        const auto& observer = gazes[i];

        cv::Vec3f right, up;
        buildHeadFrame(observer.direction, right, up);
        const cv::Vec3f forward = unitVec(observer.direction);

        struct TargetSlot {
            int personId = -1;
            size_t gazeIndex = 0;
        };

        std::vector<TargetSlot> targets;
        targets.reserve(gazes.size());
        // null target
        targets.push_back({-1, 0});

        for (size_t j = 0; j < gazes.size(); ++j) {
            if (j == i) continue;
            targets.push_back({gazes[j].personID, j});
        }

        const size_t numTargets = targets.size();
        std::vector<float> rawScores(numTargets, 0.f);
        std::vector<float> likelihoods(numTargets, 0.f);

        //null target score and likelihood
        rawScores[0] = cfg.omega_null;
        likelihoods[0] = std::max(cfg.likelihood_null_baseline, 1e-6f);

        for (size_t t = 1; t < numTargets; ++t) {
            const auto& targetGaze = gazes[targets[t].gazeIndex];
            auto motionIt = motionByPerson.find(targets[t].personId);
            const float motion = (motionIt != motionByPerson.end()) ? motionIt->second : 0.f;
            rawScores[t] = cfg.omega_base + cfg.omega_motion * motion;

            float likelihood = anisotropicLikelihood(
                forward, right, up,
                observer.start, targetGaze.start,
                sigmaXRad, sigmaYRad);
            likelihoods[t] = std::max(likelihood, 1e-6f);
        }

        std::vector<float> priors = rawScores;
        softmaxInPlace(priors);

        std::vector<float> unnormalized(numTargets, 0.f);
        for (size_t t = 0; t < numTargets; ++t) {
            unnormalized[t] = likelihoods[t] * priors[t];
        }

        float posteriorSum = 0.f;
        for (float v : unnormalized) posteriorSum += v;
        if (posteriorSum <= 1e-12f) posteriorSum = 1.f;

        int bestTargetId = -1;
        float bestPosterior = -1.f;
        for (size_t t = 0; t < numTargets; ++t) {
            const float posterior = unnormalized[t] / posteriorSum;
            if (posterior > bestPosterior) {
                bestPosterior = posterior;
                bestTargetId = targets[t].personId;
            }
        }

        for (size_t j = 0; j < gazes.size(); ++j) {
            if (j == i) continue;

            interactions.push_back(domain::InteractionPair{
                observer.personID,
                gazes[j].personID,
                angleBetweenDeg(observer.direction, observer.start, gazes[j].start),
                bestTargetId == gazes[j].personID
            });
        }
    }

    return interactions;
}

} // namespace vision
