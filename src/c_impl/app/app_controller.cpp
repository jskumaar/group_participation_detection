#include "app/app_controller.h"

#include <algorithm>

AppController::AppController(QObject* parent)
    : QObject(parent) {

    tickTimer_.setInterval(33);
    connect(&tickTimer_, &QTimer::timeout, this, &AppController::onTick);

    fpsTimer_.start();

    csvWriter_.open("gaze_analysis.csv");
    vision_.setConfig(vision_.getConfig());

    emit statusTextChanged("Ready");
    emit fpsTextChanged("FPS: 0");
    emit yoloStatusChanged(false);
    emit playButtonTextChanged("Play");
}

AppController::~AppController() {
    tickTimer_.stop();
    videoSource_.close();
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

    enterSelectionMode(firstFrame, QString("Loaded: %1. Please select people to track.").arg(path));
}

void AppController::togglePlayPause() {
    if (!videoSource_.isOpen()) return;

    if (isSelecting_) {
        std::vector<cv::Rect> selected;
        for (int idx : selectionOrder_) {
            if (idx >= 0 &&
                (size_t)idx < lastYoloDetections_.size() &&
                (size_t)idx < selectionMask_.size() &&
                selectionMask_[idx]) {
                selected.push_back(lastYoloDetections_[idx]);
            }
        }

        if (selected.size() < (size_t)vision_.getConfig().num_people) {
            emit statusTextChanged(QString("Please select at least %1 people (Selected: %2).")
                .arg(vision_.getConfig().num_people)
                .arg(selected.size()));
            return;
        }

        vision_.setExpectedPeople(vision_.getConfig().num_people);
        vision_.setSelectedDetections(selected, lastPanoRows_, lastPanoCols_);
        lastYoloDetections_ = selected;

        isSelecting_ = false;
        emit selectingChanged(false);

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

    selectionMask_[index] = !selectionMask_[index];
    if (selectionMask_[index]) {
        selectionOrder_.push_back(index);
    } else {
        auto it = std::find(selectionOrder_.begin(), selectionOrder_.end(), index);
        if (it != selectionOrder_.end()) selectionOrder_.erase(it);
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
    emit selectingChanged(true);

    selectionOrder_.clear();

    auto selection = vision_.prepareSelection(frame);
    lastYoloDetections_ = selection.yoloDetections;

    selectionMask_.assign(lastYoloDetections_.size(), false);
    updateSelectionOverlays();

    emit statusTextChanged(reason);
    emit playButtonTextChanged("Start Tracking");

    if (showVideo_) emit frameReady(frame);
}

void AppController::updateSelectionOverlays() {
    std::vector<VideoWidget::PersonBox> overlays;
    overlays.reserve(lastYoloDetections_.size());

    for (size_t i = 0; i < lastYoloDetections_.size(); ++i) {
        Qt::GlobalColor color = (i < selectionMask_.size() && selectionMask_[i]) ? Qt::green : Qt::red;
        int displayId = 0;
        if (i < selectionMask_.size() && selectionMask_[i]) {
            auto it = std::find(selectionOrder_.begin(), selectionOrder_.end(), (int)i);
            if (it != selectionOrder_.end()) displayId = (int)(it - selectionOrder_.begin()) + 1;
        }
        overlays.push_back({lastYoloDetections_[i], displayId, color});
    }
    emit overlaysReady(overlays);
}

void AppController::processTrackingFrame(const cv::Mat& frame) {
    auto result = vision_.processFrame(frame, justSeeked_, /*enableTracking*/true);
    if (justSeeked_) justSeeked_ = false;

    if (result.requestSelection) {
        // reselect required
        isPlaying_ = false;
        tickTimer_.stop();
        emit playButtonTextChanged("Resume Tracking");

        lastYoloDetections_ = result.yoloDetections;
        selectionMask_.assign(lastYoloDetections_.size(), false);
        selectionOrder_.clear();

        enterSelectionMode(frame, "Tracking Lost: Please re-select people.");
        return;
    }

    if (isPlaying_) {
        csvWriter_.writeFrame(videoSource_.currentFrame(), result.gazes);
        csvWriter_.flush();
    }

    // Interaction label
    QString interactionText = "Interactions: ";
    bool found = false;
    for (size_t i = 0; i < result.gazes.size(); ++i) {
        for (size_t j = 0; j < result.gazes.size(); ++j) {
            if (i == j) continue;
            if (vision_.panoViewer().isLookingAt(result.gazes[i], result.gazes[j].start, vision_.getConfig().max_angle_deg)) {
                if (found) interactionText += " | ";
                interactionText += QString("Person %1 -> Person %2").arg(result.gazes[i].personID).arg(result.gazes[j].personID);
                found = true;
            }
        }
    }
    if (!found) interactionText += "None";

    emit interactionTextChanged(interactionText);
    emit yoloStatusChanged(result.yoloActive);

    // Overlays
    std::vector<VideoWidget::PersonBox> overlays;
    overlays.reserve(result.gazes.size() + result.yoloDetections.size());
    for (const auto& g : result.gazes) overlays.push_back({g.box, g.personID, Qt::green});
    for (const auto& r : result.yoloDetections) overlays.push_back({r, -1, Qt::red});
    emit overlaysReady(overlays);

    if (showVTK_) emit gazesReady(result.gazes);
    if (showVideo_) emit frameReady(frame);
}

