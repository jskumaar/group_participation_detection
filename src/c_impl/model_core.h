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

        // Returns bounding wrt image coordinate of the input image
        // The preceeding float is the score for being a face normalized to [0,1].
        std::pair<float, cv::Rect2f> run(
            const cv::Mat &frame);

        double last_inference_time_millis() const;
    private:
        inline static constexpr int INPUT_IMG_WIDTH = 288;
        inline static constexpr int INPUT_IMG_HEIGHT = 224;
        Ort::Session session_{nullptr};
        // Inputs / outputs
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
            cv::Matx33f rotaxis_cov_tril; // Lower triangular factor of Cholesky decomposition
            cv::Rect2f box;
            cv::Point2f center;
            float size;
            cv::Matx33f center_size_cov_tril; // Lower triangular factor of Cholesky decomposition
        };

        PoseEstimator(Ort::MemoryInfo &allocator_info,
                        Ort::Session &&session);
        /** Inference
        *
        * Coordinates are defined wrt. the image space of the input `frame`.
        * X goes right, Z (depth) into the image, Y points down (like pixel coordinates values increase from top to bottom)
        */
        std::optional<Face> run(const cv::Mat &frame, const cv::Rect &box);
        // Returns an image compatible with the 'frame' image for displaying.
        cv::Mat last_network_input() const;
        double last_inference_time_millis() const;
        bool has_uncertainty() const { return has_uncertainty_; }

    private:
        std::string get_network_input_name(size_t i) const;
        std::string get_network_output_name(size_t i) const;
        int64_t model_version_ = 0;  // Queried meta data from the ONNX file
        Ort::Session session_{nullptr};  // ONNX's runtime context for running the model
        mutable Ort::Allocator allocator_;   // Memory allocator for tensors
        // Inputs
        cv::Mat scaled_frame_{}, input_mat_{};  // Input. One is the original crop, the other is rescaled (?)
        std::vector<Ort::Value> input_val_;    // Tensors to put into the model
        std::vector<std::string> input_names_; // Refers to the names in the onnx model.
        std::vector<const char *> input_c_names_; // Refers to the C names in the onnx model.
        // Outputs
        cv::Vec<float, 3> output_coord_{};  // 2d Coordinate and head size output.
        cv::Vec<float, 4> output_quat_{};   //  Quaternion output
        cv::Vec<float, 4> output_box_{};    // Bounding box output
        cv::Matx33f output_rotaxis_scales_tril_{}; // Lower triangular matrix of LLT factorization of covariance of rotation vector as offset from output quaternion
        cv::Matx33f output_coord_scales_tril_{}; // Lower triangular factor
        cv::Vec3f output_coord_scales_std_{}; // Depending on the model, alternatively a 3d vector with standard deviations.
        std::vector<Ort::Value> output_val_; // Tensors to put the model outputs in.
        std::vector<std::string> output_names_; // Refers to the names in the onnx model.
        std::vector<const char *> output_c_names_; // Refers to the C names in the onnx model.
        // More bookkeeping
        double last_inference_time_ = 0;
        bool has_uncertainty_ = false;
        bool pos_scale_uncertainty_is_matrix_ = false;
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


