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
#include <QPushButton>
#include <QSettings>
#include <QCoreApplication>
#include <QDir>

namespace {

QString defaultsIniPath() {
    return QDir(QCoreApplication::applicationDirPath()).filePath("tracker_settings.ini");
}

QSettings openDefaultsStore() {
    return QSettings(defaultsIniPath(), QSettings::IniFormat);
}

void saveDefaults(const TrackerConfig& cfg) {
    //file IO to ini config file
    QSettings s = openDefaultsStore();
    s.beginGroup("tracker");
    s.setValue("nms_threshold", cfg.nms_threshold);
    s.setValue("confidence_threshold", cfg.confidence_threshold);
    s.setValue("localizer_threshold", cfg.localizer_threshold);
    s.setValue("localizer_iou_threshold", cfg.localizer_iou_threshold);
    s.setValue("roi_zoom", cfg.roi_zoom);
    s.setValue("iou_threshold", cfg.iou_threshold);
    s.setValue("num_people", cfg.num_people);
    s.setValue("max_angle_deg", cfg.max_angle_deg);
    s.setValue("yolo_check_interval", cfg.yolo_check_interval);
    s.setValue("head_height_ratio", cfg.head_height_ratio);
    s.setValue("yolorerun_threshold", cfg.yolorerun_threshold);
    s.setValue("panorama_offset_px", cfg.panorama_offset_px);
    s.endGroup();
    s.sync();
}

bool loadDefaultsInto(TrackerConfig& cfg) {
    const QString path = defaultsIniPath();
    if (!QFile::exists(path)) return false;
    QSettings s(path, QSettings::IniFormat);
    s.beginGroup("tracker");
    //syntax: value(key, default_value)
    cfg.nms_threshold = s.value("nms_threshold", cfg.nms_threshold).toFloat();
    cfg.confidence_threshold = s.value("confidence_threshold", cfg.confidence_threshold).toFloat();
    cfg.localizer_threshold = s.value("localizer_threshold", cfg.localizer_threshold).toFloat();
    cfg.localizer_iou_threshold = s.value("localizer_iou_threshold", cfg.localizer_iou_threshold).toFloat();
    cfg.roi_zoom = s.value("roi_zoom", cfg.roi_zoom).toFloat();
    cfg.iou_threshold = s.value("iou_threshold", cfg.iou_threshold).toFloat();
    cfg.num_people = s.value("num_people", cfg.num_people).toInt();
    cfg.max_angle_deg = s.value("max_angle_deg", cfg.max_angle_deg).toFloat();
    cfg.yolo_check_interval = s.value("yolo_check_interval", cfg.yolo_check_interval).toInt();
    cfg.head_height_ratio = s.value("head_height_ratio", cfg.head_height_ratio).toFloat();
    cfg.yolorerun_threshold = s.value("yolorerun_threshold", cfg.yolorerun_threshold).toFloat();
    cfg.panorama_offset_px = s.value("panorama_offset_px", cfg.panorama_offset_px).toInt();
    s.endGroup();
    return true;
}

} // namespace

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
        //block signal from timeSlider to prevent infinite loop
        QSignalBlocker blocker(timeSlider);
        timeSlider->setValue((int)value);
    });

    connect(controller_, &AppController::playbackPositionChanged, this,
        [this](long currentFrame, long totalFrames, qint64 timestampNs){
            const double seconds = static_cast<double>(timestampNs) / 1e9;
            const int totalMs = static_cast<int>(seconds * 1000.0 + 0.5);
            const int hours = totalMs / 3600000;
            const int minutes = (totalMs / 60000) % 60;
            const int secs = (totalMs / 1000) % 60;
            const int millis = totalMs % 1000;
            QString timeText = (hours > 0)
                ? QString("%1:%2:%3.%4")
                    .arg(hours)
                    .arg(minutes, 2, 10, QChar('0'))
                    .arg(secs, 2, 10, QChar('0'))
                    .arg(millis, 3, 10, QChar('0'))
                : QString("%1:%2.%3")
                    .arg(minutes, 2, 10, QChar('0'))
                    .arg(secs, 2, 10, QChar('0'))
                    .arg(millis, 3, 10, QChar('0'));
            positionLabel->setText(QString("Frame: %1 / %2  |  Time: %3")
                .arg(currentFrame)
                .arg(totalFrames)
                .arg(timeText));
        });

    connect(controller_, &AppController::frameReady, this, [this](const cv::Mat& frame){
        videoWidget->updateFrame(frame);
    });
    connect(controller_, &AppController::overlaysReady, this, [this](const std::vector<VideoWidget::PersonBox>& overlays){
        videoWidget->updateOverlays(overlays);
    });
    connect(controller_, &AppController::gazesReady, this, [this](const std::vector<PanoViewer::gaze>& gazes){
        gazeVisualizer->updateData(gazes);
    });

    controller_->setShowVideo(true);
    controller_->setShowVTK(true);

    {
        TrackerConfig savedConfig = controller_->getTrackerConfig();
        if (loadDefaultsInto(savedConfig)) {
            controller_->applyTrackerConfig(savedConfig);
        }
    }
}

MainWindow::~MainWindow() {
}

void MainWindow::setupUI() {
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

    //automatically becomes child of mainLayout
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

    positionLabel = new QLabel("Frame: 0 / 0  |  Time: 00:00.000", this);
    statusBar()->addPermanentWidget(positionLabel);

    fpsLabel = new QLabel("FPS: 0", this);
    statusBar()->addPermanentWidget(fpsLabel);

    yoloLabel = new QLabel("YOLO: -", this);
    statusBar()->addPermanentWidget(yoloLabel);
}

void MainWindow::openVideo() {
    //trigger file selection popup
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
    controller_->setShowVideo(checked);
}

void MainWindow::onVTKToggle(bool checked) {
    controller_->setShowVTK(checked);
}

void MainWindow::showSettings() {
    QDialog dialog(this);
    dialog.setWindowTitle("Settings");
    QFormLayout form(&dialog);

    TrackerConfig config = controller_->getTrackerConfig();

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

    QDoubleSpinBox *locIouBox = new QDoubleSpinBox(&dialog);
    locIouBox->setRange(0, 1);
    locIouBox->setSingleStep(0.01);
    locIouBox->setValue(config.localizer_iou_threshold);
    form.addRow("Localizer IoU Threshold:", locIouBox);

    QDoubleSpinBox *roiZoomBox = new QDoubleSpinBox(&dialog);
    roiZoomBox->setRange(1, 5);
    roiZoomBox->setSingleStep(0.05);
    roiZoomBox->setValue(config.roi_zoom);
    form.addRow("ROI Zoom:", roiZoomBox);

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
    maxAngleBox->setSingleStep(0.5);
    maxAngleBox->setValue(config.max_angle_deg);
    form.addRow("Looking Angle Threshold (deg):", maxAngleBox);

    QSpinBox *yoloIntervalBox = new QSpinBox(&dialog);
    yoloIntervalBox->setRange(1, 100);
    yoloIntervalBox->setValue(config.yolo_check_interval);
    form.addRow("YOLO Interval (frames):", yoloIntervalBox);

    QSpinBox *offsetBox = new QSpinBox(&dialog);
    // Keep this wide; controller applies modulo pano width.
    offsetBox->setRange(-50000, 50000);
    offsetBox->setValue(config.panorama_offset_px);
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
    QPushButton *setDefaultBtn = buttonBox.addButton("Set as Default", QDialogButtonBox::ActionRole);
    form.addRow(&buttonBox);
    connect(&buttonBox, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(&buttonBox, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);

    auto collectConfig = [&]() {
        TrackerConfig cfg = config;
        cfg.nms_threshold = nmsBox->value();
        cfg.confidence_threshold = confBox->value();
        cfg.localizer_threshold = locBox->value();
        cfg.localizer_iou_threshold = locIouBox->value();
        cfg.roi_zoom = roiZoomBox->value();
        cfg.iou_threshold = iouBox->value();
        cfg.num_people = numPeopleBox->value();
        cfg.max_angle_deg = maxAngleBox->value();
        cfg.yolo_check_interval = yoloIntervalBox->value();
        cfg.head_height_ratio = headRatioBox->value();
        cfg.yolorerun_threshold = rerunThreshBox->value();
        cfg.panorama_offset_px = offsetBox->value();
        return cfg;
    };

    connect(setDefaultBtn, &QPushButton::clicked, &dialog, [&]() {
        TrackerConfig cfg = collectConfig();
        controller_->applyTrackerConfig(cfg);
        saveDefaults(cfg);
        QMessageBox::information(&dialog, "Settings",
            QString("Saved as default to:\n%1").arg(defaultsIniPath()));
    });

    if (dialog.exec() == QDialog::Accepted) {
        config = collectConfig();
        controller_->applyTrackerConfig(config);
    }

    //if rejected, do nothing
}

void MainWindow::onBoxClicked(int index) {
    controller_->onBoxClicked(index);
}

