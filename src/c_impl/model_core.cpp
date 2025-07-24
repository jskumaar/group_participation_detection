#include "pre_process.h"


Localizer::Localizer(Ort::MemoryInfo &allocator_info, Ort::Session &&session) :
    session_{std::move(session)},
    scaled_frame_(INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, CV_8U),
    input_mat_(INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, CV_32F)
{
    // Only works when input_mat does not reallocated memory ...which it should not.
    // Non-owning memory reference to input_mat?
    // Note: shape = (bach x channels x h x w)
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
    scaled_frame_.convertTo(input_mat_, CV_32F, 1./255., -0.5);

    assert (input_mat_.ptr(0) == p);
    assert (!input_mat_.empty() && input_mat_.isContinuous());
    assert (input_mat_.cols == INPUT_IMG_WIDTH && input_mat_.rows == INPUT_IMG_HEIGHT);

    const char* input_names[] = {"x"};
    const char* output_names[] = {"logit_box"};

    Timer t; t.start();

    session_.Run(Ort::RunOptions{nullptr}, input_names, &input_val_, 1, output_names, &output_val_, 1);

    last_inference_time_ = t.elapsed_ms();

    const cv::Rect2f roi = unnormalize(cv::Rect2f{
        results_[1],
        results_[2],
        results_[3]-results_[1], // Width
        results_[4]-results_[2] // Height
    }, frame.rows, frame.cols);
    const float score = sigmoid(results_[0]);

    return { score, roi };
}