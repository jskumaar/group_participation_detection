#pragma once

#include <string>

#include <opencv2/opencv.hpp>

namespace media {

class VideoSource {
public:
    VideoSource() = default;
    ~VideoSource();

    bool open(const std::string& path);
    void close();

    bool isOpen() const;

    void setPreferredFrameSize(int width, int height);
    bool read(cv::Mat& outFrame);             // advances by one frame
    bool seekFrame(long frameIndex);          // random access

    long totalFrames() const;
    long currentFrame() const;                // OpenCV's current frame index

private:
    cv::VideoCapture cap_;
    int preferredWidth_ = 0;
    int preferredHeight_ = 0;
};

} // namespace media

