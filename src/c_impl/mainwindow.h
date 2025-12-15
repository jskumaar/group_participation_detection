#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QElapsedTimer>
#include <QSlider>
#include <QPushButton>
#include <QCheckBox>
#include <QLabel>
#include <opencv2/opencv.hpp>

// Include original project headers
#include "360_image_process.h"
#include "tracker.h"
#include "SORT.h" 
#include "videowidget.h"

#include "gazevisualizer.h"



class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void openVideo();
    void togglePlayPause();
    void updateFrame();
    void seekVideo(int value);
    void toggleVisuals(int state);
    void onVideoToggle(int state);
    void onVTKToggle(int state);
    void showSettings();

private:
    void setupUI();
    void processFrame(cv::Mat& frame);

    // UI Components
    VideoWidget *videoWidget;
    QSlider *timeSlider;
    QPushButton *playButton;
    QCheckBox *visualsCheckBox;
    QCheckBox *videoToggleBtn;
    QCheckBox *vtkToggleBtn;
    QLabel *statusLabel;
    QLabel *fpsLabel;
    QLabel *yoloLabel;

    // Logic
    QTimer *timer;
    QElapsedTimer fpsTimer;
    int frameCount = 0;
    cv::VideoCapture cap;
    bool isPlaying = false;
    long totalFrames = 0;
    bool showVideo = true;
    bool showVTK = true;

    // Tracking Members (Moved from main.cpp globals)
    OPNetTracker tracker;
    Sort object_tracker;
    PanoViewer pano_viewer;
    
    std::vector<Sort::Track> tracks;
    float localizer_confidence = 0;
    bool yolo_update = false;
    
    // Config
    const int GLOBAL_FRAME_WIDTH = 2880;
    const int GLOBAL_FRAME_HEIGHT = 1440;
    float localizer_threshold = 0.5f;
    int panorama_offset = 0;
    
    // Gaze Analysis
    std::ofstream csv_file;
    float eyeBoost(float yaw_deg);


    
    GazeVisualizer *gazeVisualizer;
};

#endif // MAINWINDOW_H
