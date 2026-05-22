#include "app/app_controller.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <string>

#include "io/grpc_pipeline_publisher.h"

namespace {

std::unique_ptr<io::GrpcPipelinePublisher> createGrpcStream() {
    const char* target = std::getenv("RAPPORT_GRPC_TARGET");
    const std::string endpoint = (target != nullptr && *target != '\0')
        ? std::string(target)
        : std::string("127.0.0.1:50051");

    const char* maxQueuedEnv = std::getenv("RAPPORT_GRPC_MAX_QUEUED");
    std::size_t maxQueued = 256;
    if (maxQueuedEnv != nullptr) {
        try {
            maxQueued = static_cast<std::size_t>(std::stoul(maxQueuedEnv));
        } catch (...) {
            maxQueued = 256;
        }
    }

    return std::make_unique<io::GrpcPipelinePublisher>(endpoint, maxQueued);
}

} // namespace

AppController::AppController(QObject* parent)
    : QObject(parent) {

    tickTimer_.setInterval(25);
    connect(&tickTimer_, &QTimer::timeout, this, &AppController::onTick);

    fpsTimer_.start();

    csvWriter_.open("gaze_analysis.csv");
    vision_.setConfig(vision_.getConfig());
    pipelinePublisher_ = createGrpcStream();
    pipelinePublisher_->start();

    emit statusTextChanged("Ready");
    emit fpsTextChanged("FPS: 0");
    emit yoloStatusChanged(false);
    emit playButtonTextChanged("Play");
}

AppController::~AppController() {
    tickTimer_.stop();
    videoSource_.close();
    if (pipelinePublisher_) {
        pipelinePublisher_->stop();
    }
    csvWriter_.close();
}

void AppController::applyTrackerConfig(const TrackerConfig& cfg) {
    vision_.setConfig(cfg);
}

void AppController::openVideoPath(const QString& path) {
    if (path.isEmpty()) return;

    videoSource_.close();
    videoSource_.setPreferredFrameSize(2880, 1440);

    if (!videoSource_.open(path.toStdString())) {
        emit statusTextChanged("Could not open video file.");
        return;
    }

    totalFrames_ = videoSource_.totalFrames();
    emit sliderRangeChanged(0, totalFrames_);

    cv::Mat firstFrame;
    videoSource_.read(firstFrame);
    if (firstFrame.empty()) {
        emit statusTextChanged("Failed to load first frame.");
        return;
    }

    lastPanoRows_ = firstFrame.rows;
    lastPanoCols_ = firstFrame.cols;

    enterSelectionMode(
        firstFrame,
        QString("Loaded: %1. Please select exactly %2 people to track.")
            .arg(path)
            .arg(vision_.getConfig().num_people));
}

void AppController::togglePlayPause() {
    if (!videoSource_.isOpen()) return;

    //convert selectionorder_ (list of indices) to list of respective bounding boxes
    if (isSelecting_) {
        std::vector<cv::Rect> selected;
        for (int idx : selectionOrder_) {
            if (idx >= 0 &&
                (size_t)idx < selectionCandidates_.size() &&
                (size_t)idx < selectionMask_.size() &&
                selectionMask_[idx]) {
                selected.push_back(selectionCandidates_[idx]);
            }
        }

        if ((int)selected.size() != vision_.getConfig().num_people) {
            emit statusTextChanged(QString("Select exactly %1 people (currently %2).")
                .arg(vision_.getConfig().num_people)
                .arg(selected.size()));
            return;
        }

        vision_.setExpectedPeople(vision_.getConfig().num_people);
        vision_.setSelectedDetections(selected, lastPanoRows_, lastPanoCols_);

        isSelecting_ = false;

        isPlaying_ = true;
        tickTimer_.start();
        emit playButtonTextChanged("Pause");
        emit statusTextChanged("Tracking Started");
        return;
    }

    isPlaying_ = !isPlaying_;
    if (isPlaying_) {
        tickTimer_.start();
        emit playButtonTextChanged("Pause");
    } else {
        tickTimer_.stop();
        emit playButtonTextChanged("Play");
    }
}

void AppController::seekFrame(int frameIndex) {
    if (!videoSource_.isOpen()) return;
    videoSource_.seekFrame(frameIndex);
    justSeeked_ = true;
    if (!isPlaying_) onTick();
}

void AppController::onBoxClicked(int index) {
    if (!isSelecting_) return;
    if (index < 0 || (size_t)index >= selectionMask_.size()) return;

    const int need = vision_.getConfig().num_people;

    if (selectionMask_[index]) {
        selectionMask_[index] = false;
        auto it = std::find(selectionOrder_.begin(), selectionOrder_.end(), index);
        if (it != selectionOrder_.end()) selectionOrder_.erase(it);
    } else {
        if ((int)selectionOrder_.size() >= need) {
            emit statusTextChanged(
                QString("Select exactly %1 people — click a green box to deselect one first.").arg(need));
            return;
        }
        selectionMask_[index] = true;
        selectionOrder_.push_back(index);
    }
    updateSelectionOverlays();
}

void AppController::onTick() {
    if (!videoSource_.isOpen()) return;

    cv::Mat frame;
    videoSource_.read(frame);
    if (frame.empty()) {
        tickTimer_.stop();
        isPlaying_ = false;
        emit playButtonTextChanged("Play");
        emit statusTextChanged("End of video");
        return;
    }

    lastPanoRows_ = frame.rows;
    lastPanoCols_ = frame.cols;

    long currentFrame = videoSource_.currentFrame();
    emit sliderValueChanged(currentFrame);
    emit playbackPositionChanged(
        currentFrame,
        totalFrames_,
        static_cast<qint64>(videoSource_.currentTimestampNs()));

    frameCount_++;
    if (fpsTimer_.elapsed() > 1000) {
        emit fpsTextChanged(QString("FPS: %1").arg(frameCount_));
        frameCount_ = 0;
        fpsTimer_.restart();
    }

    if (isSelecting_) {
        // Selection overlay stays driven by click events; just update frame.
        if (showVideo_) emit frameReady(frame);
        return;
    }

    processTrackingFrame(frame);
}

void AppController::enterSelectionMode(const cv::Mat& frame, const QString& reason) {
    isSelecting_ = true;

    selectionOrder_.clear();

    auto selection = vision_.prepareSelection(frame);
    selectionCandidates_ = selection.yoloDetections;

    selectionMask_.assign(selectionCandidates_.size(), false);
    updateSelectionOverlays();

    emit statusTextChanged(reason);
    emit playButtonTextChanged("Start Tracking");

    if (showVideo_) emit frameReady(frame);
}

void AppController::updateSelectionOverlays() {
    std::vector<VideoWidget::PersonBox> overlays;
    overlays.reserve(selectionCandidates_.size());

    //set id 0 if not selected, 1-based index if selected since 
    //label 0 does not get displayed in VideoWidget
    for (size_t i = 0; i < selectionCandidates_.size(); ++i) {
        Qt::GlobalColor color = (i < selectionMask_.size() && selectionMask_[i]) ? Qt::green : Qt::red;
        int displayId = 0;
        if (i < selectionMask_.size() && selectionMask_[i]) {
            auto it = std::find(selectionOrder_.begin(), selectionOrder_.end(), (int)i);
            if (it != selectionOrder_.end()) displayId = (int)(it - selectionOrder_.begin()) + 1;
        }
        overlays.push_back({selectionCandidates_[i], displayId, color});
    }
    emit overlaysReady(overlays);
}

void AppController::processTrackingFrame(const cv::Mat& frame) {
    auto result = vision_.processFrame(frame, justSeeked_);
    if (justSeeked_) justSeeked_ = false;

    if (result.requestSelection) {
        // reselect required
        isPlaying_ = false;
        tickTimer_.stop();
        emit playButtonTextChanged("Resume Tracking");

        enterSelectionMode(frame, "Tracking Lost: Please re-select people.");
        return;
    }

    QString interactionText = "Interactions: ";
    bool found = false;
    for (const auto& edge : result.interactions) {
        if (!edge.is_looking) continue;
        if (found) interactionText += " | ";
        interactionText += QString("Person %1 -> Person %2")
            .arg(edge.from_person_id)
            .arg(edge.to_person_id);
        found = true;
    }
    if (!found) interactionText += "None";

    if (isPlaying_) {
        csvWriter_.writeFrame(videoSource_.currentFrame(), result.gazes, result.interactions);
        csvWriter_.flush();
    }

    emit interactionTextChanged(interactionText);
    emit yoloStatusChanged(result.yoloActive);

    // Overlays
    //yolo detections only contain yolo bpunding boxes of people we are tracking
    std::vector<VideoWidget::PersonBox> overlays;
    overlays.reserve(result.gazes.size() + result.yoloDetections.size());
    for (const auto& g : result.gazes) overlays.push_back({g.box, g.personID, Qt::green});
    for (const auto& r : result.yoloDetections) overlays.push_back({r, -1, Qt::red});
    emit overlaysReady(overlays);

    if (pipelinePublisher_) {
        domain::PipelineFrameContext publishContext;
        publishContext.frame_index = static_cast<std::uint64_t>(videoSource_.currentFrame());
        publishContext.playback_timestamp_ns = videoSource_.currentTimestampNs();
        publishContext.interactions = result.interactions;
        pipelinePublisher_->publish(result, publishContext);
    }

    if (showVTK_) emit gazesReady(result.gazes);
    if (showVideo_) emit frameReady(frame);
}

