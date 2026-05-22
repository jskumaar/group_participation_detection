#include "vision/model_core.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <unordered_map>

cv::Rect2f unnormalize(const cv::Rect2f &r, int h, int w)
{
    auto unnorm = [](float x) -> float { return 0.5f*(x+1); };
    auto tl = r.tl();
    auto br = r.br();
    auto x0 = unnorm(tl.x)*w;
    auto y0 = unnorm(tl.y)*h;
    auto x1 = unnorm(br.x)*w;
    auto y1 = unnorm(br.y)*h;
    return {
        x0, y0, x1-x0, y1-y0
    };
}

float sigmoid(float x)
{
    return 1.f/(1.f + std::exp(-x));
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i < vec.size() - 1) os << ", ";
    }
    os << "]";
    return os;
}

Ort::Value create_tensor(const Ort::TypeInfo& info, Ort::Allocator& alloc)
{
    const auto shape = info.GetTensorTypeAndShapeInfo().GetShape();
    auto t = Ort::Value::CreateTensor<float>(
        alloc, shape.data(), shape.size());
    memset(t.GetTensorMutableData<float>(), 0, sizeof(float)*info.GetTensorTypeAndShapeInfo().GetElementCount());
    return t;
}

int find_input_intensity_quantile(const cv::Mat& frame, float percentage)
{
    const int channels[] = { 0 };
    const int hist_size[] = { 256 };
    float range[] = { 0, 256 };
    const float* ranges[] = { range };
    cv::Mat hist;
    cv::calcHist(&frame, 1,  channels, cv::Mat(), hist, 1, hist_size, ranges, true, false);
    int gray_level = 0;
    const int num_pixels_quantile = frame.total()*percentage*0.01f;
    int num_pixels_accum = 0;
    for (int i=0; i<hist_size[0]; ++i)
    {
        num_pixels_accum += hist.at<float>(i);
        if (num_pixels_accum > num_pixels_quantile)
        {
            gray_level = i;
            break;
        }
    }
    return gray_level;
}

void normalize_brightness(const cv::Mat& frame, cv::Mat& out)
{
    const float pct = 90;

    const int brightness = find_input_intensity_quantile(frame, pct);

    const double alpha = brightness<127 ? (pct/100.f*0.5f/std::max(5,brightness)) : 1./255;
    const double beta = -0.5;

    frame.convertTo(out, CV_32F, alpha, beta);
}

std::string PoseEstimator::get_network_input_name(size_t i) const
{
#if ORT_API_VERSION >= 12
    return std::string(&*session_.GetInputNameAllocated(i, allocator_));
#else
    return std::string(session_.GetInputName(i, allocator_));
#endif
}

std::string PoseEstimator::get_network_output_name(size_t i) const
{
#if ORT_API_VERSION >= 12
    return std::string(&*session_.GetOutputNameAllocated(i, allocator_));
#else
    return std::string(session_.GetOutputName(i, allocator_));
#endif
}

Localizer::Localizer(Ort::MemoryInfo &allocator_info, Ort::Session &&session) :
    session_{std::move(session)},
    scaled_frame_(INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, CV_8U),
    input_mat_(INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, CV_32F)
{
    const std::int64_t input_shape[4] = { 1, 1, INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH };
    input_val_ = Ort::Value::CreateTensor<float>(allocator_info, input_mat_.ptr<float>(0), input_mat_.total(), input_shape, 4);

    const std::int64_t output_shape[2] = { 1, 5 };
    output_val_ = Ort::Value::CreateTensor<float>(allocator_info, results_.data(), results_.size(), output_shape, 2);
}

std::pair<float, cv::Rect2f> Localizer::run(
    const cv::Mat &frame)
{
    auto p = input_mat_.ptr(0);
    cv::resize(frame, scaled_frame_, { INPUT_IMG_WIDTH, INPUT_IMG_HEIGHT }, 0, 0, cv::INTER_AREA);
    scaled_frame_.convertTo(input_mat_, CV_32F, 1.0/255.0, -0.5);
    assert (input_mat_.ptr(0) == p);
    assert (!input_mat_.empty() && input_mat_.isContinuous());
    assert (input_mat_.cols == INPUT_IMG_WIDTH && input_mat_.rows == INPUT_IMG_HEIGHT);

    const char* input_names[] = {"x"};
    const char* output_names[] = {"logit_box"};

    session_.Run(Ort::RunOptions{nullptr}, input_names, &input_val_, 1, output_names, &output_val_, 1);

    const cv::Rect2f roi = unnormalize(cv::Rect2f{
        results_[1],
        results_[2],
        results_[3]-results_[1],
        results_[4]-results_[2]
    }, frame.rows, frame.cols);
    const float score = sigmoid(results_[0]);

    return { score, roi };
}

cv::Quatf image_to_world(cv::Quatf q);

cv::Size get_input_image_shape(const Ort::Session &session)
{
    if (session.GetInputCount() < 1)
        throw std::invalid_argument("Model must take at least one input tensor");
    const std::vector<std::int64_t> shape =
        session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 4)
        throw std::invalid_argument("Model takes the input tensor in the wrong shape");
    return { static_cast<int>(shape[3]), static_cast<int>(shape[2]) };
}

PoseEstimator::PoseEstimator(Ort::MemoryInfo &allocator_info, Ort::Session &&session)
    : model_version_{session.GetModelMetadata().GetVersion()}
    , session_{std::move(session)}
    , allocator_{session_, allocator_info}
{
    using namespace std::literals::string_literals;

    if (session_.GetOutputCount() < 2)
        throw std::runtime_error("Invalid Model: must have at least two outputs");

    if (model_version_ <= 0 || model_version_ > 4)
        model_version_ = 1;

    const cv::Size input_image_shape = get_input_image_shape(session_);

    scaled_frame_ = cv::Mat(input_image_shape, CV_8U, cv::Scalar(0));
    input_mat_ = cv::Mat(input_image_shape, CV_32F, cv::Scalar(0.f));

    {
        const std::int64_t input_shape[4] = { 1, 1, input_image_shape.height, input_image_shape.width };
        input_val_.push_back(
            Ort::Value::CreateTensor<float>(allocator_info, input_mat_.ptr<float>(0), input_mat_.total(), input_shape, 4));
    }

    struct TensorSpec
    {
        std::vector<int64_t> shape;
        float* buffer = nullptr;
        size_t element_count = 0;
        bool element_count_is_opencv_rows = false;
        bool available = false;
    };

    std::unordered_map<std::string, TensorSpec> understood_outputs = {
        { "pos_size", TensorSpec{ { 1, 3 }, &output_coord_[0], (size_t)output_coord_.rows } },
        { "quat", TensorSpec{ { 1, 4},  &output_quat_[0], (size_t)output_quat_.rows } },
        { "box", TensorSpec{ { 1, 4}, &output_box_[0], (size_t)output_box_.rows } },
        { "rotaxis_scales_tril", TensorSpec{ {1, 3, 3}, output_rotaxis_scales_tril_.val, 9 }},
        { "rotaxis_std", TensorSpec{ {1, 3, 3}, output_rotaxis_scales_tril_.val, 9 }},
        { "pos_size_std", TensorSpec{ {1, 3}, output_coord_scales_std_.val, (size_t)output_coord_scales_std_.rows}},
        { "pos_size_scales", TensorSpec{ {1, 3}, output_coord_scales_std_.val, (size_t)output_coord_scales_std_.rows}},
        { "pos_size_scales_tril", TensorSpec{ {1, 3, 3}, output_coord_scales_tril_.val, 9}}
    };

    output_names_.resize(session_.GetOutputCount());
    output_c_names_.resize(session_.GetOutputCount());
    for (size_t i=0; i<session_.GetOutputCount(); ++i)
    {
        const std::string name = get_network_output_name(i);
        const auto& output_info = session_.GetOutputTypeInfo(i);
        const auto& onnx_tensor_spec = output_info.GetTensorTypeAndShapeInfo();
        auto my_tensor_spec_it = understood_outputs.find(name);

        if (my_tensor_spec_it != understood_outputs.end())
        {
            TensorSpec& t = my_tensor_spec_it->second;
            if (onnx_tensor_spec.GetShape() != t.shape ||
                onnx_tensor_spec.GetElementType() != Ort::TypeToTensorType<float>::type)
                throw std::runtime_error("Invalid output tensor spec for "s + name);
            output_val_.push_back(Ort::Value::CreateTensor<float>(
                allocator_info, t.buffer, t.element_count, t.shape.data(), t.shape.size()));
            t.available = true;
        }
        else
        {
            output_val_.push_back(create_tensor(output_info, allocator_));
        }
        output_names_[i] = name;
        output_c_names_[i] = output_names_[i].c_str();
    }

    has_uncertainty_ = understood_outputs.at("rotaxis_scales_tril").available ||
                       understood_outputs.at("rotaxis_std").available;
    has_uncertainty_ &= understood_outputs.at("pos_size_std").available ||
                        understood_outputs.at("pos_size_scales").available ||
                        understood_outputs.at("pos_size_scales_tril").available;
    pos_scale_uncertainty_is_matrix_ = understood_outputs.at("pos_size_scales_tril").available;

    input_names_.resize(session_.GetInputCount());
    input_c_names_.resize(session_.GetInputCount());
    for (size_t i = 0; i < session_.GetInputCount(); ++i)
    {
        input_names_[i] = get_network_input_name(i);
        input_c_names_[i] = input_names_[i].c_str();
    }

    assert (input_names_.size() == input_val_.size());
    assert (output_names_.size() == output_val_.size());
}

std::optional<PoseEstimator::Face> PoseEstimator::run(
    const cv::Mat &frame, const cv::Rect &box)
{
    cv::Mat cropped;

    const int patch_size = std::max(box.width, box.height)*1.05;
    const cv::Point2f patch_center = {
        std::clamp<float>(box.x + 0.5f*box.width, 0.f, frame.cols),
        std::clamp<float>(box.y + 0.5f*box.height, 0.f, frame.rows)
    };
    cv::getRectSubPix(frame, {patch_size, patch_size}, patch_center, cropped);

    if (cropped.rows != patch_size || cropped.cols != patch_size)
        return {};

    [[maybe_unused]] auto* p = input_mat_.ptr(0);

    cv::resize(cropped, scaled_frame_, scaled_frame_.size(), 0, 0, cv::INTER_AREA);

    normalize_brightness(scaled_frame_, input_mat_);

    assert (input_mat_.ptr(0) == p);
    assert (!input_mat_.empty() && input_mat_.isContinuous());

    try
    {
        session_.Run(
            Ort::RunOptions{ nullptr },
            input_c_names_.data(),
            input_val_.data(),
            input_val_.size(),
            output_c_names_.data(),
            output_val_.data(),
            output_val_.size());
    }
    catch (const Ort::Exception &e)
    {
        std::cout << "Failed to run the model: " << e.what();
        return {};
    }

    cv::Matx33f center_size_cov_tril = {};
    if (has_uncertainty_)
    {
        if (pos_scale_uncertainty_is_matrix_)
        {
            center_size_cov_tril = output_coord_scales_tril_;
        }
        else
        {
            center_size_cov_tril(0,0) = output_coord_scales_std_[0];
            center_size_cov_tril(1,1) = output_coord_scales_std_[1];
            center_size_cov_tril(2,2) = output_coord_scales_std_[2];
        }
        center_size_cov_tril *= patch_size*0.5f;
    }

    const cv::Point2f center = patch_center +
        (0.5f*patch_size)*cv::Point2f{output_coord_[0], output_coord_[1]};
    const float size = patch_size*0.5f*output_coord_[2];

    cv::Quatf rotation = {
        output_quat_[3],
        output_quat_[0],
        output_quat_[1],
        output_quat_[2] };

    assert(output_rotaxis_scales_tril_(0, 1) == 0);
    assert(output_rotaxis_scales_tril_(0, 2) == 0);
    assert(output_rotaxis_scales_tril_(1, 2) == 0);
    assert(center_size_cov_tril(0, 1) == 0);
    assert(center_size_cov_tril(0, 2) == 0);
    assert(center_size_cov_tril(1, 2) == 0);

    cv::Matx33f rotaxis_scales_tril = output_rotaxis_scales_tril_;

    if (model_version_ < 2)
    {
        rotation = image_to_world(rotation);
    }

    const cv::Rect2f outbox = {
        patch_center.x + (0.5f*patch_size)*output_box_[0],
        patch_center.y + (0.5f*patch_size)*output_box_[1],
        0.5f*patch_size*(output_box_[2]-output_box_[0]),
        0.5f*patch_size*(output_box_[3]-output_box_[1])
    };

    return std::optional<Face>({
        rotation, rotaxis_scales_tril, outbox, center, size, center_size_cov_tril
    });
}

L2CSEstimator::L2CSEstimator(Ort::MemoryInfo &allocator_info, Ort::Session &&session)
    : session_{std::move(session)}
    , resized_rgb_(INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, CV_8UC3, cv::Scalar(0))
    , input_tensor_data_(3 * INPUT_IMG_HEIGHT * INPUT_IMG_WIDTH, 0.f)
{
    if (session_.GetInputCount() < 1)
        throw std::runtime_error("Invalid L2CS model: missing input tensor");
    if (session_.GetOutputCount() < 2)
        throw std::runtime_error("Invalid L2CS model: expected at least two output tensors");

    Ort::AllocatorWithDefaultOptions allocator;
#if ORT_API_VERSION >= 12
    input_name_ = std::string(session_.GetInputNameAllocated(0, allocator).get());
    output_names_[0] = std::string(session_.GetOutputNameAllocated(0, allocator).get());
    output_names_[1] = std::string(session_.GetOutputNameAllocated(1, allocator).get());
#else
    input_name_ = std::string(session_.GetInputName(0, allocator));
    output_names_[0] = std::string(session_.GetOutputName(0, allocator));
    output_names_[1] = std::string(session_.GetOutputName(1, allocator));
#endif
    input_c_names_[0] = input_name_.c_str();
    output_c_names_[0] = output_names_[0].c_str();
    output_c_names_[1] = output_names_[1].c_str();

    const std::int64_t input_shape[4] = {1, 3, INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH};
    input_val_ = Ort::Value::CreateTensor<float>(
        allocator_info,
        input_tensor_data_.data(),
        input_tensor_data_.size(),
        input_shape,
        4);
}

std::optional<cv::Rect> L2CSEstimator::clamp_roi_to_frame(const cv::Rect2f &box, const cv::Size &frame_size) const
{
    if (frame_size.width <= 0 || frame_size.height <= 0)
        return std::nullopt;

    const int x0 = std::clamp(static_cast<int>(std::floor(box.x)), 0, frame_size.width - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(box.y)), 0, frame_size.height - 1);
    const int x1 = std::clamp(static_cast<int>(std::ceil(box.x + box.width)), 0, frame_size.width);
    const int y1 = std::clamp(static_cast<int>(std::ceil(box.y + box.height)), 0, frame_size.height);

    if (x1 <= x0 || y1 <= y0)
        return std::nullopt;

    return cv::Rect(x0, y0, x1 - x0, y1 - y0);
}

float L2CSEstimator::decode_angle_from_logits(const std::vector<float> &logits, float *max_prob_out)
{
    if (logits.empty())
    {
        if (max_prob_out)
            *max_prob_out = 0.f;
        return 0.f;
    }

    const float max_logit = *std::max_element(logits.begin(), logits.end());
    float exp_sum = 0.f;
    for (const float logit : logits)
        exp_sum += std::exp(logit - max_logit);

    if (exp_sum <= std::numeric_limits<float>::epsilon())
    {
        if (max_prob_out)
            *max_prob_out = 0.f;
        return 0.f;
    }

    float expected_bin = 0.f;
    float max_prob = 0.f;
    for (size_t i = 0; i < logits.size(); ++i)
    {
        const float prob = std::exp(logits[i] - max_logit) / exp_sum;
        expected_bin += prob * static_cast<float>(i);
        max_prob = std::max(max_prob, prob);
    }

    if (max_prob_out)
        *max_prob_out = max_prob;
    return expected_bin * 4.f - 180.f;
}

std::optional<L2CSEstimator::HeadPose> L2CSEstimator::run(const cv::Mat &frame, const cv::Rect2f &box)
{
    if (frame.empty())
        return std::nullopt;

    const auto roi = clamp_roi_to_frame(box, frame.size());
    if (!roi.has_value())
        return std::nullopt;

    cv::Mat cropped = frame(*roi).clone();
    if (cropped.empty())
        return std::nullopt;

    if (cropped.channels() == 1)
        cv::cvtColor(cropped, resized_rgb_, cv::COLOR_GRAY2RGB);
    else if (cropped.channels() == 3)
        cv::cvtColor(cropped, resized_rgb_, cv::COLOR_BGR2RGB);
    else
        return std::nullopt;

    cv::resize(resized_rgb_, resized_rgb_, {INPUT_IMG_WIDTH, INPUT_IMG_HEIGHT}, 0, 0, cv::INTER_AREA);

    constexpr std::array<float, 3> mean = {0.485f, 0.456f, 0.406f};
    constexpr std::array<float, 3> std = {0.229f, 0.224f, 0.225f};
    const size_t plane_size = static_cast<size_t>(INPUT_IMG_WIDTH) * INPUT_IMG_HEIGHT;
    for (int y = 0; y < INPUT_IMG_HEIGHT; ++y)
    {
        const auto *row = resized_rgb_.ptr<cv::Vec3b>(y);
        for (int x = 0; x < INPUT_IMG_WIDTH; ++x)
        {
            for (int c = 0; c < 3; ++c)
            {
                const float v = static_cast<float>(row[x][c]) / 255.f;
                input_tensor_data_[c * plane_size + static_cast<size_t>(y) * INPUT_IMG_WIDTH + x] = (v - mean[c]) / std[c];
            }
        }
    }

    std::vector<Ort::Value> outputs;
    try
    {
        outputs = session_.Run(
            Ort::RunOptions{nullptr},
            input_c_names_.data(),
            &input_val_,
            1,
            output_c_names_.data(),
            output_c_names_.size());
    }
    catch (const Ort::Exception &)
    {
        return std::nullopt;
    }

    if (outputs.size() < 2 || !outputs[0].IsTensor() || !outputs[1].IsTensor())
        return std::nullopt;

    auto flatten_logits = [](const Ort::Value &tensor) -> std::vector<float> {
        const auto info = tensor.GetTensorTypeAndShapeInfo();
        const size_t count = info.GetElementCount();
        if (count == 0 || info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            return {};
        const float *data = tensor.GetTensorData<float>();
        return std::vector<float>(data, data + count);
    };

    const std::vector<float> yaw_logits = flatten_logits(outputs[0]);
    const std::vector<float> pitch_logits = flatten_logits(outputs[1]);
    if (yaw_logits.empty() || pitch_logits.empty())
        return std::nullopt;

    float yaw_conf = 0.f;
    float pitch_conf = 0.f;
    const float yaw_deg = decode_angle_from_logits(yaw_logits, &yaw_conf);
    const float pitch_deg = decode_angle_from_logits(pitch_logits, &pitch_conf);

    return HeadPose{
        yaw_deg,
        pitch_deg,
        0.5f * (yaw_conf + pitch_conf)};
}

void Reframer::extract_detections(std::vector<Reframer::DetectedPeople>& oResult){
    cv::Mat transposed;
    cv::transpose(output_mat_, transposed);
    float* data = (float*)transposed.data;
    for (int i = 0; i < mOutputDims[2]; ++i)
    {
        float* classesScores = data + 4;
        cv::Mat scores(1, mOutputDims[1]-4, CV_32FC1, classesScores);
        cv::Point class_id;
        double maxClassScore;
        cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
        if (maxClassScore > rectConfidenceThreshold && class_id.x == 0)
        {
            confidences.push_back((float)maxClassScore);
            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];
            int left = int(x - 0.5 * w);
            int top = int(y - 0.5 * h);
            boxes.push_back(cv::Rect(left, top, (int)w, (int)h));
        }
        data += mOutputDims[1];
    }
    cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsResult);
    for (int i = 0; i < (int)nmsResult.size(); ++i)
    {
        int idx = nmsResult[i];
        Reframer::DetectedPeople result;
        result.confidence = confidences[idx];
        result.bbox = boxes[idx];
        oResult.push_back(result);
    }
    boxes.clear();
    confidences.clear();
    nmsResult.clear();
}

Reframer::Reframer(Ort::MemoryInfo &allocator_info,
                   Ort::Session &&session, float confidence, float iou)
    : session_(std::move(session))
    , scaled_frame_(INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, CV_8UC3)
    , input_mat_(std::vector<int>{1, 3, INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH}, CV_32F, cv::Scalar(0.f))
{
    rectConfidenceThreshold = confidence;
    iouThreshold = iou;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr input_name = session_.GetInputNameAllocated(0, allocator);
    input_node_name_ = std::string(input_name.get());
    inputNodeNames[0] = input_node_name_.c_str();
    inputNodeDims = { 1, 3, INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH };

    Ort::AllocatedStringPtr output_name = session_.GetOutputNameAllocated(0, allocator);
    output_node_name_ = std::string(output_name.get());
    outputNodeNames[0] = output_node_name_.c_str();
    mOutputDims = session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    output_mat_ = cv::Mat((int)mOutputDims[1], (int)mOutputDims[2], CV_32F);

    input_val_ = Ort::Value::CreateTensor<float>(allocator_info, input_mat_.ptr<float>(),
        input_mat_.total(), inputNodeDims.data(), inputNodeDims.size());
    output_val_ = Ort::Value::CreateTensor<float>(allocator_info, output_mat_.ptr<float>(0),
        output_mat_.total(), mOutputDims.data(), mOutputDims.size());
}

std::vector<Reframer::DetectedPeople> Reframer::run(const cv::Mat &frame)
{
    auto p = input_mat_.ptr(0);
    cv::Mat temp_blob;
    cv::dnn::blobFromImage(frame, temp_blob, 1/255.0, cv::Size(INPUT_IMG_WIDTH, INPUT_IMG_HEIGHT), cv::Scalar(), false, false);
    temp_blob.copyTo(input_mat_);
    assert (input_mat_.ptr(0) == p);
    p = output_mat_.ptr(0);
    session_.Run(Ort::RunOptions{nullptr},
        inputNodeNames, &input_val_, 1,
        outputNodeNames, &output_val_, 1);
    assert (output_mat_.ptr(0) == p);
    std::vector<Reframer::DetectedPeople> detectedPeople;
    extract_detections(detectedPeople);
    return detectedPeople;
}

