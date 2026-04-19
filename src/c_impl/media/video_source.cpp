#include "video_source.h"

namespace media {

VideoSource::~VideoSource() {
    close();
}

bool VideoSource::open(const std::string& path) {
    close();
    cap_.open(path);
    if (!cap_.isOpened()) return false;

    if (preferredWidth_ > 0) cap_.set(cv::CAP_PROP_FRAME_WIDTH, preferredWidth_);
    if (preferredHeight_ > 0) cap_.set(cv::CAP_PROP_FRAME_HEIGHT, preferredHeight_);
    return true;
}

void VideoSource::close() {
    if (cap_.isOpened()) cap_.release();
}

bool VideoSource::isOpen() const {
    return cap_.isOpened();
}

void VideoSource::setPreferredFrameSize(int width, int height) {
    preferredWidth_ = width;
    preferredHeight_ = height;
    if (cap_.isOpened()) {
        if (preferredWidth_ > 0) cap_.set(cv::CAP_PROP_FRAME_WIDTH, preferredWidth_);
        if (preferredHeight_ > 0) cap_.set(cv::CAP_PROP_FRAME_HEIGHT, preferredHeight_);
    }
}

bool VideoSource::read(cv::Mat& outFrame) {
    if (!cap_.isOpened()) return false;
    cap_ >> outFrame;
    return !outFrame.empty();
}

bool VideoSource::seekFrame(long frameIndex) {
    if (!cap_.isOpened()) return false;
    return cap_.set(cv::CAP_PROP_POS_FRAMES, static_cast<double>(frameIndex));
}

long VideoSource::totalFrames() const {
    if (!cap_.isOpened()) return 0;
    return static_cast<long>(cap_.get(cv::CAP_PROP_FRAME_COUNT));
}

long VideoSource::currentFrame() const {
    if (!cap_.isOpened()) return 0;
    return static_cast<long>(cap_.get(cv::CAP_PROP_POS_FRAMES));
}

std::uint64_t VideoSource::currentTimestampNs() const {
    if (!cap_.isOpened()) return 0;
    const double posMs = cap_.get(cv::CAP_PROP_POS_MSEC);
    if (!(posMs > 0.0)) return 0;
    return static_cast<std::uint64_t>(posMs * 1'000'000.0);
}

} // namespace media

