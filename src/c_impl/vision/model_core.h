#include <array>
#include <vector>
#include <string>
#include <optional>
#include <opencv2/core/quaternion.hpp>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

float sigmoid(float x);

class Localizer
{
    public:
        Localizer(Ort::MemoryInfo &allocator_info,
                    Ort::Session &&session);

        std::pair<float, cv::Rect2f> run(
            const cv::Mat &frame);

        double last_inference_time_millis() const;
    private:
        inline static constexpr int INPUT_IMG_WIDTH = 288;
        inline static constexpr int INPUT_IMG_HEIGHT = 224;
        Ort::Session session_{nullptr};
        cv::Mat scaled_frame_, input_mat_;
        Ort::Value input_val_{nullptr}, output_val_{nullptr};
        std::array<float, 5> results_;
        double last_inference_time_ = 0;
};

class PoseEstimator
{
    public:
        struct Face
        {
            cv::Quatf rotation;
            cv::Matx33f rotaxis_cov_tril;
            cv::Rect2f box;
            cv::Point2f center;
            float size;
            cv::Matx33f center_size_cov_tril;
        };

        PoseEstimator(Ort::MemoryInfo &allocator_info,
                        Ort::Session &&session);
        std::optional<Face> run(const cv::Mat &frame, const cv::Rect &box);
        cv::Mat last_network_input() const;
        double last_inference_time_millis() const;
        bool has_uncertainty() const { return has_uncertainty_; }

    private:
        std::string get_network_input_name(size_t i) const;
        std::string get_network_output_name(size_t i) const;
        int64_t model_version_ = 0;
        Ort::Session session_{nullptr};
        mutable Ort::Allocator allocator_;
        cv::Mat scaled_frame_{}, input_mat_{};
        std::vector<Ort::Value> input_val_;
        std::vector<std::string> input_names_;
        std::vector<const char *> input_c_names_;
        cv::Vec<float, 3> output_coord_{};
        cv::Vec<float, 4> output_quat_{};
        cv::Vec<float, 4> output_box_{};
        cv::Matx33f output_rotaxis_scales_tril_{};
        cv::Matx33f output_coord_scales_tril_{};
        cv::Vec3f output_coord_scales_std_{};
        std::vector<Ort::Value> output_val_;
        std::vector<std::string> output_names_;
        std::vector<const char *> output_c_names_;
        double last_inference_time_ = 0;
        bool has_uncertainty_ = false;
        bool pos_scale_uncertainty_is_matrix_ = false;
};

class L2CSEstimator
{
    public:
        struct HeadPose
        {
            float yaw_deg = 0.f;
            float pitch_deg = 0.f;
            float confidence = 0.f;
        };

        L2CSEstimator(Ort::MemoryInfo &allocator_info, Ort::Session &&session);
        std::optional<HeadPose> run(const cv::Mat &frame, const cv::Rect2f &box);

    private:
        std::optional<cv::Rect> clamp_roi_to_frame(const cv::Rect2f &box, const cv::Size &frame_size) const;
        static float decode_angle_from_logits(const float *logits, size_t count, float *max_prob_out);
        void fill_input_nchw_from_rgb(const cv::Mat &rgb_uint8);

        inline static constexpr int INPUT_IMG_WIDTH = 448;
        inline static constexpr int INPUT_IMG_HEIGHT = 448;
        Ort::Session session_{nullptr};
        std::string input_name_;
        std::array<std::string, 2> output_names_;
        std::array<const char *, 2> output_c_names_{};
        std::array<const char *, 1> input_c_names_{};
        cv::Mat resized_rgb_;
        cv::Mat roi_work_;
        cv::Mat float_rgb_;
        std::array<cv::Mat, 3> ch_planes_{};
        std::vector<float> input_tensor_data_;
        Ort::Value input_val_{nullptr};
};

class Reframer
{
    public:
        struct DetectedPeople {
            cv::Rect bbox;
            float confidence;
        };
        Reframer(Ort::MemoryInfo &allocator_info,
                   Ort::Session &&session, float confidence, float iou);
        std::vector<DetectedPeople> run(const cv::Mat &frame);

        void setThresholds(float confidence, float iou) {
            rectConfidenceThreshold = confidence;
            iouThreshold = iou;
        }
    private:
        void extract_detections(std::vector<Reframer::DetectedPeople>& oResult);
        Ort::Session session_{nullptr};
        cv::Mat scaled_frame_{}, input_mat_{};
        cv::Mat output_mat_{};
        std::vector<int64_t> inputNodeDims{};
        std::vector<int64_t> mOutputDims{};
        inline static constexpr int INPUT_IMG_WIDTH = 640;
        inline static constexpr int INPUT_IMG_HEIGHT = 640;
        std::string input_node_name_;
        std::string output_node_name_;
        const char* inputNodeNames[1];
        const char* outputNodeNames[1];
        Ort::Value input_val_{nullptr}, output_val_{nullptr};
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        std::vector<int> nmsResult;
        float iouThreshold;
        float rectConfidenceThreshold;
};

