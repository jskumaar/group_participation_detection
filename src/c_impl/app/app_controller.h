#pragma once

#include <QObject>
#include <QElapsedTimer>
#include <QTimer>
#include <QString>
#include <memory>

#include <opencv2/opencv.hpp>

#include "app/videowidget.h"
#include "io/csv_gaze_writer.h"
#include "io/grpc_pipeline_publisher.h"
#include "media/video_source.h"
#include "pipelines/vision_pipeline.h"

class AppController : public QObject {
    Q_OBJECT

public:
    explicit AppController(QObject* parent = nullptr);
    ~AppController() override;

    // UI-driven configuration
    void setShowVideo(bool show) { showVideo_ = show; }
    void setShowVTK(bool show) { showVTK_ = show; }

    bool isPlaying() const { return isPlaying_; }
    bool isSelecting() const { return isSelecting_; }

    TrackerConfig getTrackerConfig() const { return vision_.getConfig(); }
    void applyTrackerConfig(const TrackerConfig& cfg);

signals:
    void statusTextChanged(const QString& text);
    void fpsTextChanged(const QString& text);
    void yoloStatusChanged(bool active);
    void playButtonTextChanged(const QString& text);
    void sliderRangeChanged(long min, long max);
    void sliderValueChanged(long value);
    void playbackPositionChanged(long currentFrame, long totalFrames, qint64 timestampNs);

    void frameReady(const cv::Mat& frame);
    void overlaysReady(const std::vector<VideoWidget::PersonBox>& overlays);
    void gazesReady(const std::vector<PanoViewer::gaze>& gazes);
    void interactionTextChanged(const QString& text);

    void selectingChanged(bool selecting);

public slots:
    void openVideoPath(const QString& path);
    void togglePlayPause();
    void seekFrame(int frameIndex);
    void onBoxClicked(int index);

private slots:
    void onTick();

private:
    void enterSelectionMode(const cv::Mat& frame, const QString& reason);
    void updateSelectionOverlays();
    void processTrackingFrame(const cv::Mat& frame);

    QTimer tickTimer_;
    QElapsedTimer fpsTimer_;
    int frameCount_ = 0;

    media::VideoSource videoSource_;
    pipelines::VisionPipeline vision_;
    io::CsvGazeWriter csvWriter_;
    std::unique_ptr<io::GrpcPipelinePublisher> pipelinePublisher_;

    bool showVideo_ = true;
    bool showVTK_ = true;

    bool isPlaying_ = false;
    bool isSelecting_ = false;
    bool justSeeked_ = false;

    long totalFrames_ = 0;
    int lastPanoRows_ = 0;
    int lastPanoCols_ = 0;

    /** Raw YOLO boxes on the current frame while the user is picking exactly num_people. */
    std::vector<cv::Rect> selectionCandidates_;
    std::vector<bool> selectionMask_;
    std::vector<int> selectionOrder_;
};

