#include "mainwindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QDockWidget>
#include <QStatusBar>
#include <QDebug>
#include <QDialog>
#include <QFormLayout>
#include <QDoubleSpinBox>
#include <QDialogButtonBox>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    setupUI();
    
    // Initialize Tracking Objects
    // Note: Constructors for tracker/object_tracker/pano_viewer called implicitly or match default
    
    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &MainWindow::updateFrame);

    // Start FPS timer
    fpsTimer.start();


    // Open CSV for logging    
    csv_file.open("gaze_analysis.csv");
    if (csv_file.is_open()) {
        csv_file << "Frame,PersonID,BoxCenterX,BoxCenterY,GazeStartX,GazeStartY,GazeStartZ,GazeDirX,GazeDirY,GazeDirZ,LookingAtIDs\n";
    }
}

MainWindow::~MainWindow() {
    if (cap.isOpened()) cap.release();
    if (csv_file.is_open()) csv_file.close();
}

void MainWindow::setupUI() {
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

    // Content Area (Video + 3D Graph)
    QHBoxLayout *contentLayout = new QHBoxLayout();
    mainLayout->addLayout(contentLayout, 1);

    // Video Area
    videoWidget = new VideoWidget(this);
    videoWidget->setMinimumSize(800, 400); // 2:1 Aspect Ratio
    contentLayout->addWidget(videoWidget, 2);

    // 3D Visualizer
    gazeVisualizer = new GazeVisualizer(this);
    gazeVisualizer->setMinimumWidth(300);
    contentLayout->addWidget(gazeVisualizer, 1);

    // Controls
    QHBoxLayout *controlsLayout = new QHBoxLayout();
    
    QPushButton *openBtn = new QPushButton("Open Video", this);
    connect(openBtn, &QPushButton::clicked, this, &MainWindow::openVideo);
    controlsLayout->addWidget(openBtn);

    QPushButton *settingsBtn = new QPushButton("Settings", this);
    connect(settingsBtn, &QPushButton::clicked, this, &MainWindow::showSettings);
    controlsLayout->addWidget(settingsBtn);

    playButton = new QPushButton("Play", this);
    playButton->setEnabled(false);
    connect(playButton, &QPushButton::clicked, this, &MainWindow::togglePlayPause);
    controlsLayout->addWidget(playButton);

    timeSlider = new QSlider(Qt::Horizontal, this);
    timeSlider->setEnabled(false);
    connect(timeSlider, &QSlider::sliderMoved, this, &MainWindow::seekVideo);
    // Note: for smoother scrubbing, might want sliderPressed/Released logic
    controlsLayout->addWidget(timeSlider);

    visualsCheckBox = new QCheckBox("Draw Overlays", this);
    visualsCheckBox->setChecked(true);
    connect(visualsCheckBox, &QCheckBox::toggled, this, &MainWindow::toggleVisuals);
    controlsLayout->addWidget(visualsCheckBox);

    videoToggleBtn = new QCheckBox("Show Video", this);
    videoToggleBtn->setChecked(true);
    connect(videoToggleBtn, &QCheckBox::toggled, this, &MainWindow::onVideoToggle);
    controlsLayout->addWidget(videoToggleBtn);

    vtkToggleBtn = new QCheckBox("Show 3D", this);
    vtkToggleBtn->setChecked(true);
    connect(vtkToggleBtn, &QCheckBox::toggled, this, &MainWindow::onVTKToggle);

    controlsLayout->addWidget(vtkToggleBtn);

    mainLayout->addLayout(controlsLayout);



    statusLabel = new QLabel("Ready", this);
    statusBar()->addWidget(statusLabel);

    fpsLabel = new QLabel("FPS: 0", this);
    statusBar()->addPermanentWidget(fpsLabel);

    yoloLabel = new QLabel("YOLO: -", this);
    statusBar()->addPermanentWidget(yoloLabel);
}

void MainWindow::openVideo() {
    QString fileName = QFileDialog::getOpenFileName(this, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv)");
    if (fileName.isEmpty()) return;

    if (cap.isOpened()) cap.release();
    
    cap.open(fileName.toStdString());
    if (!cap.isOpened()) {
        QMessageBox::critical(this, "Error", "Could not open video file.");
        return;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, GLOBAL_FRAME_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, GLOBAL_FRAME_HEIGHT);
    
    totalFrames = (long)cap.get(cv::CAP_PROP_FRAME_COUNT);
    timeSlider->setRange(0, totalFrames);
    timeSlider->setEnabled(true);
    playButton->setEnabled(true);
    
    isPlaying = true;
    playButton->setText("Pause");
    timer->start(33); // ~30 FPS
    
    statusLabel->setText(QString("Loaded: %1").arg(fileName));
}

void MainWindow::togglePlayPause() {
    if (!cap.isOpened()) return;
    
    isPlaying = !isPlaying;
    if (isPlaying) {
        timer->start(33);
        playButton->setText("Pause");
    } else {
        timer->stop();
        playButton->setText("Play");
    }
}

void MainWindow::seekVideo(int value) {
    if (!cap.isOpened()) return;
    cap.set(cv::CAP_PROP_POS_FRAMES, value);
    // If paused, update single frame
    if (!isPlaying) {
        updateFrame();
    }
}

void MainWindow::updateFrame() {
    if (!cap.isOpened()) return;
    
    // If playing, standard grab. If seeking (paused), we might just read once.
    // Logic here assumes playing or single-shot update
    
    cv::Mat pano_frame;
    cap >> pano_frame;
    
    if (pano_frame.empty()) {
        timer->stop();
        isPlaying = false;
        playButton->setText("Play");
        statusLabel->setText("End of video");
        return;
    }
    
    // Update Slider UI
    long currentFrame = (long)cap.get(cv::CAP_PROP_POS_FRAMES);
    QSignalBlocker blocker(timeSlider); // Prevent seek loop
    timeSlider->setValue(currentFrame);

    // FPS Calculation
    frameCount++;
    yoloFrameCount++;
    if (fpsTimer.elapsed() > 1000) {
        // qDebug() << "FPS:" << frameCount;
        fpsLabel->setText(QString("FPS: %1").arg(frameCount));
        frameCount = 0;
        fpsTimer.restart();
    }

    processFrame(pano_frame);
    // videoWidget->updateFrame(pano_frame); // Moved to processFrame to respect toggle
}

void MainWindow::processFrame(cv::Mat& pano_frame) {
    if (pano_frame.cols != pano_frame.rows * 2) {
         printf("Panorama frame is not a valid size.\n");
         return;
    }

    // Apply Circular Shift if offset is set
    if (panorama_offset != 0) {
        int w = pano_frame.cols;
        int h = pano_frame.rows;
        int shift = panorama_offset % w;
        if (shift < 0) shift += w; // Handle negative
        
        if (shift != 0) {
            // Split and concatenate
            // Left part becomes right part of new image, Right part becomes left part.
            // Wait, shifting right by 'shift':
            // New image starts with the last 'shift' pixels (which wrap around from right)
            // Followed by the rest.
            
            // Use a temporary buffer to avoid in-place writing over source data
            cv::Mat result;
            cv::Mat left_part = pano_frame(cv::Rect(w - shift, 0, shift, h));
            cv::Mat right_part = pano_frame(cv::Rect(0, 0, w - shift, h));
            cv::hconcat(left_part, right_part, result);
            pano_frame = result;
        }
    }
    
    // Tracking Logic (Adapted from main.cpp)
    std::vector<VideoWidget::PersonBox> currentPeople;
    std::vector<cv::Rect> people;
    bool interval_check = (yoloFrameCount % tracker.getConfig().yolo_check_interval == 0);
    yoloFrameCount = interval_check ? 0 : yoloFrameCount;
    if (tracker.need_yolo_update() || (tracks.size() < num_people && interval_check)) {
        people = tracker.run_yolo(pano_frame);
        lastYoloDetections = people;
        tracks = object_tracker.inject(people, pano_frame.rows, pano_frame.cols, tracker.getConfig().head_height_ratio);
        printf("yolo updated\n");
        tracker.yolo_updated();
    }

    cv::Mat perspective_view;
    std::vector<PanoViewer::gaze> gazes;
    
    // Clear old data points if too many
    // For simplicity, we just append. In a real app, we'd scroll/cull.
    for (const auto& track : tracks) {
         
         perspective_view = track.viewer->generatePerspectiveView(pano_frame);
         auto pose = tracker.run(perspective_view, static_cast<int>(track.viewer->getFOV()));
        if (!pose.has_value()){
            continue;
        }
        float yaw_boost = eyeBoost(pose->yaw);
         
        PanoViewer::gaze g = track.viewer->addGaze(track.viewer->getYaw(),  
                                                  track.viewer->getPitch(), 
                                                  track.viewer->getFOV(), 
                                                  pose->yaw * yaw_boost, 
                                                  -pose->pitch, 
                                                  cv::Vec3f(pose->x, pose->y, pose->z));
        g.box = track.viewer->convertPerspectiveRectToEquirectangular(pose->rect, pano_frame.cols, pano_frame.rows);
        gazes.emplace_back(g);
        perspective_view.release();
    }

    gazes = object_tracker.update(gazes);
    // Save to CSV if playing
    if (isPlaying) {
         pano_viewer.saveGazeAnalysis(csv_file, (long)cap.get(cv::CAP_PROP_POS_FRAMES), gazes);
         csv_file.flush(); 
    }

    // Update 3D Visualizer
    if (showVTK) {
        gazeVisualizer->updateData(gazes);
    }
    
    // Update Visuals
    if (showVideo) {
        for(const auto& g : gazes){
            currentPeople.push_back({g.box, g.personID, Qt::green});
        }
        for(const auto& r : lastYoloDetections){
            currentPeople.push_back({r, -1, Qt::red});
        }
        videoWidget->updateOverlays(currentPeople);
        videoWidget->updateFrame(pano_frame);
    }

    // Update YOLO Status
    if (tracker.need_yolo_update()) {
        yoloLabel->setText("YOLO: ACTIVE");
        yoloLabel->setStyleSheet("QLabel { color : green; font-weight: bold; }");
    } else {
        yoloLabel->setText("YOLO: -");
        yoloLabel->setStyleSheet("QLabel { color : gray; }");
    }
}

void MainWindow::toggleVisuals(bool checked) {
    videoWidget->setShowVisuals(checked);
}

void MainWindow::onVideoToggle(bool checked) {
    showVideo = checked;
    if (!showVideo) {
         // Optionally clear video widget or leave last frame
    }
}

void MainWindow::onVTKToggle(bool checked) {
    showVTK = checked;
}

float MainWindow::eyeBoost(float yaw_deg) {
    TrackerConfig config = tracker.getConfig();
    float y = std::abs(yaw_deg);
    float max_boost = config.eye_boost_max;
    float knee = config.eye_boost_knee;
    float steepness = config.eye_boost_steepness;
    float extra = max_boost * (1.0f - 1.0f / (1.0f + std::exp((y - knee) * steepness)));
    return 1.0f + extra;
}

void MainWindow::showSettings() {
    QDialog dialog(this);
    dialog.setWindowTitle("Settings");
    QFormLayout form(&dialog);

    // Get current config
    TrackerConfig config = tracker.getConfig();

    QDoubleSpinBox *headSizeBox = new QDoubleSpinBox(&dialog);
    headSizeBox->setRange(100, 500);
    headSizeBox->setValue(config.head_size_mm);
    form.addRow("Head Size (mm):", headSizeBox);

    QDoubleSpinBox *nmsBox = new QDoubleSpinBox(&dialog);
    nmsBox->setRange(0, 1);
    nmsBox->setSingleStep(0.01);
    nmsBox->setValue(config.nms_threshold);
    form.addRow("NMS Threshold:", nmsBox);

    QDoubleSpinBox *confBox = new QDoubleSpinBox(&dialog);
    confBox->setRange(0, 1);
    confBox->setSingleStep(0.01);
    confBox->setValue(config.confidence_threshold);
    form.addRow("Confidence Threshold:", confBox);

    QDoubleSpinBox *locBox = new QDoubleSpinBox(&dialog);
    locBox->setRange(0, 1);
    locBox->setSingleStep(0.01);
    locBox->setValue(config.localizer_threshold);
    form.addRow("Localizer Threshold:", locBox);

    QDoubleSpinBox *decayBox = new QDoubleSpinBox(&dialog);
    decayBox->setRange(0, 1);
    decayBox->setSingleStep(0.05);
    decayBox->setValue(config.velocity_decay);
    form.addRow("Velocity Decay:", decayBox);

    QDoubleSpinBox *iouBox = new QDoubleSpinBox(&dialog);
    iouBox->setRange(0, 1);
    iouBox->setSingleStep(0.01);
    iouBox->setValue(config.iou_threshold);
    form.addRow("IOU Threshold:", iouBox);

    QSpinBox *numPeopleBox = new QSpinBox(&dialog);
    numPeopleBox->setRange(1, 20);
    numPeopleBox->setValue(config.num_people);
    form.addRow("Num People:", numPeopleBox);

    QDoubleSpinBox *maxAngleBox = new QDoubleSpinBox(&dialog);
    maxAngleBox->setRange(1, 180);
    maxAngleBox->setValue(config.max_angle_deg);
    form.addRow("Max Angle (deg):", maxAngleBox);

    QDoubleSpinBox *kneeBox = new QDoubleSpinBox(&dialog);
    kneeBox->setRange(0, 50);
    kneeBox->setValue(config.eye_boost_knee);
    form.addRow("Eye Boost Knee:", kneeBox);

    QDoubleSpinBox *steepnessBox = new QDoubleSpinBox(&dialog);
    steepnessBox->setRange(0, 1);
    steepnessBox->setSingleStep(0.01);
    steepnessBox->setValue(config.eye_boost_steepness);
    form.addRow("Eye Boost Steepness:", steepnessBox);

    QDoubleSpinBox *maxBoostBox = new QDoubleSpinBox(&dialog);
    maxBoostBox->setRange(0, 5);
    maxBoostBox->setSingleStep(0.1);
    maxBoostBox->setValue(config.eye_boost_max);
    form.addRow("Eye Boost Max:", maxBoostBox);

    QSpinBox *yoloIntervalBox = new QSpinBox(&dialog);
    yoloIntervalBox->setRange(1, 100);
    yoloIntervalBox->setValue(config.yolo_check_interval);
    form.addRow("YOLO Interval (frames):", yoloIntervalBox);

    QSpinBox *offsetBox = new QSpinBox(&dialog);
    offsetBox->setRange(-GLOBAL_FRAME_WIDTH, GLOBAL_FRAME_WIDTH);
    offsetBox->setValue(panorama_offset);
    form.addRow("Panorama Offset (px):", offsetBox);

    QDoubleSpinBox *headRatioBox = new QDoubleSpinBox(&dialog);
    headRatioBox->setRange(0.01, 1.0);
    headRatioBox->setSingleStep(0.01);
    headRatioBox->setValue(config.head_height_ratio);
    form.addRow("Head Height Ratio:", headRatioBox);

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, Qt::Horizontal, &dialog);
    form.addRow(&buttonBox);
    connect(&buttonBox, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(&buttonBox, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);

    if (dialog.exec() == QDialog::Accepted) {
        config.head_size_mm = headSizeBox->value();
        config.nms_threshold = nmsBox->value();
        config.confidence_threshold = confBox->value();
        config.localizer_threshold = locBox->value();
        config.velocity_decay = decayBox->value();
        config.iou_threshold = iouBox->value();
        config.num_people = numPeopleBox->value();
        config.max_angle_deg = maxAngleBox->value();
        
        config.eye_boost_knee = kneeBox->value();
        config.eye_boost_steepness = steepnessBox->value();
        config.eye_boost_max = maxBoostBox->value();
        config.yolo_check_interval = yoloIntervalBox->value();
        config.head_height_ratio = headRatioBox->value();
        
        tracker.setConfig(config);
        
        // Update Objects
        object_tracker.setIOUThreshold(config.iou_threshold);
        KalmanTracker::decay_velocity_factor = config.velocity_decay;
        this->num_people = config.num_people;
        pano_viewer.setMaxAngle(config.max_angle_deg);
        
        panorama_offset = offsetBox->value();
    }
}
