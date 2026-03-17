#include "app/mainwindow.h"

#include <algorithm>

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QStatusBar>
#include <QDialog>
#include <QFormLayout>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QDialogButtonBox>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    setupUI();

    controller_ = new AppController(this);

    connect(controller_, &AppController::statusTextChanged, statusLabel, &QLabel::setText);
    connect(controller_, &AppController::fpsTextChanged, fpsLabel, &QLabel::setText);
    connect(controller_, &AppController::playButtonTextChanged, playButton, &QPushButton::setText);
    connect(controller_, &AppController::interactionTextChanged, gazeInfoLabel, &QLabel::setText);

    connect(controller_, &AppController::yoloStatusChanged, this, [this](bool active){
        if (active) {
            yoloLabel->setText("YOLO: ACTIVE");
            yoloLabel->setStyleSheet("QLabel { color : green; font-weight: bold; }");
        } else {
            yoloLabel->setText("YOLO: -");
            yoloLabel->setStyleSheet("QLabel { color : gray; }");
        }
    });

    connect(controller_, &AppController::sliderRangeChanged, this, [this](long min, long max){
        timeSlider->setRange((int)min, (int)max);
        timeSlider->setEnabled(true);
        playButton->setEnabled(true);
    });
    connect(controller_, &AppController::sliderValueChanged, this, [this](long value){
        QSignalBlocker blocker(timeSlider);
        timeSlider->setValue((int)value);
    });

    connect(controller_, &AppController::frameReady, this, [this](const cv::Mat& frame){
        if (showVideo) videoWidget->updateFrame(frame);
    });
    connect(controller_, &AppController::overlaysReady, this, [this](const std::vector<VideoWidget::PersonBox>& overlays){
        videoWidget->updateOverlays(overlays);
    });
    connect(controller_, &AppController::gazesReady, this, [this](const std::vector<PanoViewer::gaze>& gazes){
        if (showVTK) gazeVisualizer->updateData(gazes);
    });

    controller_->setShowVideo(showVideo);
    controller_->setShowVTK(showVTK);
}

MainWindow::~MainWindow() {
}

void MainWindow::setupUI() {
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

    QHBoxLayout *contentLayout = new QHBoxLayout();
    mainLayout->addLayout(contentLayout, 1);

    videoWidget = new VideoWidget(this);
    videoWidget->setMinimumSize(800, 400);
    connect(videoWidget, &VideoWidget::boxClicked, this, &MainWindow::onBoxClicked);
    contentLayout->addWidget(videoWidget, 2);

    gazeVisualizer = new GazeVisualizer(this);
    gazeVisualizer->setMinimumWidth(300);
    contentLayout->addWidget(gazeVisualizer, 1);

    gazeInfoLabel = new QLabel("Interactions: None", this);
    gazeInfoLabel->setAlignment(Qt::AlignCenter);
    gazeInfoLabel->setStyleSheet("font-size: 14px; font-weight: bold; color: yellow;");
    mainLayout->addWidget(gazeInfoLabel);

    QHBoxLayout *controlsLayout = new QHBoxLayout();

    QPushButton *openBtn = new QPushButton("Open Video", this);
    connect(openBtn, &QPushButton::clicked, this, &MainWindow::openVideo);
    controlsLayout->addWidget(openBtn);

    QPushButton *settingsBtn = new QPushButton("Settings", this);
    connect(settingsBtn, &QPushButton::clicked, this, &MainWindow::showSettings);
    controlsLayout->addWidget(settingsBtn);

    playButton = new QPushButton("Play", this);
    playButton->setEnabled(false);
    connect(playButton, &QPushButton::clicked, this, [this](){
        controller_->togglePlayPause();
    });
    controlsLayout->addWidget(playButton);

    timeSlider = new QSlider(Qt::Horizontal, this);
    timeSlider->setEnabled(false);
    connect(timeSlider, &QSlider::sliderMoved, this, &MainWindow::seekVideo);
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
    controller_->openVideoPath(fileName);
}

void MainWindow::seekVideo(int value) {
    controller_->seekFrame(value);
}

void MainWindow::toggleVisuals(bool checked) {
    videoWidget->setShowVisuals(checked);
}

void MainWindow::onVideoToggle(bool checked) {
    showVideo = checked;
    controller_->setShowVideo(checked);
}

void MainWindow::onVTKToggle(bool checked) {
    showVTK = checked;
    controller_->setShowVTK(checked);
}

void MainWindow::showSettings() {
    QDialog dialog(this);
    dialog.setWindowTitle("Settings");
    QFormLayout form(&dialog);

    TrackerConfig config = controller_->getTrackerConfig();

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
    // Keep this wide; controller applies modulo pano width.
    offsetBox->setRange(-50000, 50000);
    offsetBox->setValue(controller_->panoramaOffsetPx());
    form.addRow("Panorama Offset (px):", offsetBox);

    QDoubleSpinBox *rerunThreshBox = new QDoubleSpinBox(&dialog);
    rerunThreshBox->setRange(0.01, 1.0);
    rerunThreshBox->setSingleStep(0.05);
    rerunThreshBox->setValue(config.yolorerun_threshold);
    form.addRow("YOLO Rerun IoU Threshold:", rerunThreshBox);

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
        config.yolorerun_threshold = rerunThreshBox->value();

        controller_->applyTrackerConfig(config);
        controller_->setPanoramaOffsetPx(offsetBox->value());
    }
}

void MainWindow::onBoxClicked(int index) {
    controller_->onBoxClicked(index);
}

