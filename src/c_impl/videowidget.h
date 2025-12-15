#ifndef VIDEOWIDGET_H
#define VIDEOWIDGET_H

#include <QWidget>
#include <QImage>
#include <QPainter>
#include <vector>
#include <opencv2/opencv.hpp>
#include "360_image_process.h" // For PanoViewer::gaze

class VideoWidget : public QWidget {
    Q_OBJECT
public:
    struct PersonBox {
        cv::Rect box;
        int id;
    };

    explicit VideoWidget(QWidget *parent = nullptr);
    void updateFrame(const cv::Mat& frame);
    void updateOverlays(const std::vector<PersonBox>& people);
    void setShowVisuals(bool show);

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    QImage m_currentImage;
    std::vector<PersonBox> m_people;
    bool m_showVisuals = true;

    QImage matToQImage(const cv::Mat& mat);
};

#endif // VIDEOWIDGET_H
