#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QSlider>
#include <QPushButton>
#include <QCheckBox>
#include <QLabel>

#include "app/videowidget.h"
#include "app/gazevisualizer.h"

#include "app/app_controller.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void openVideo();
    void seekVideo(int value);
    void toggleVisuals(bool checked);
    void onVideoToggle(bool checked);
    void onVTKToggle(bool checked);
    void showSettings();
    void onBoxClicked(int index);

private:
    void setupUI();

    // UI Components
    VideoWidget *videoWidget = nullptr;
    QSlider *timeSlider = nullptr;
    QPushButton *playButton = nullptr;
    QCheckBox *visualsCheckBox = nullptr;
    QCheckBox *videoToggleBtn = nullptr;
    QCheckBox *vtkToggleBtn = nullptr;
    QLabel *statusLabel = nullptr;
    QLabel *fpsLabel = nullptr;
    QLabel *yoloLabel = nullptr;
    QLabel *positionLabel = nullptr;
    QLabel *gazeInfoLabel = nullptr;
    GazeVisualizer *gazeVisualizer = nullptr;

    AppController* controller_ = nullptr;
};

#endif // MAINWINDOW_H

