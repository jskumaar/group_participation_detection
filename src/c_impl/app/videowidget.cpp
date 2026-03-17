#include "app/videowidget.h"

#include <QDebug>
#include <QMouseEvent>

VideoWidget::VideoWidget(QWidget *parent) : QWidget(parent) {
    setAttribute(Qt::WA_OpaquePaintEvent);
}

void VideoWidget::updateFrame(const cv::Mat& frame) {
    if (frame.empty()) return;
    m_currentImage = matToQImage(frame);
    update();
}

void VideoWidget::updateOverlays(const std::vector<PersonBox>& people) {
    m_people = people;
    if (m_showVisuals) update();
}

void VideoWidget::setShowVisuals(bool show) {
    m_showVisuals = show;
    update();
}

void VideoWidget::paintEvent(QPaintEvent *event) {
    QPainter painter(this);

    if (m_currentImage.isNull()) {
        painter.fillRect(rect(), Qt::black);
        return;
    }

    painter.drawImage(rect(), m_currentImage);

    if (!m_showVisuals) return;

    float scaleX = (float)width() / m_currentImage.width();
    float scaleY = (float)height() / m_currentImage.height();

    painter.setFont(QFont("Arial", 12, QFont::Bold));

    for (const auto& person : m_people) {
        painter.setPen(QPen(person.color, 2));

        QRect qRect(
            person.box.x * scaleX,
            person.box.y * scaleY,
            person.box.width * scaleX,
            person.box.height * scaleY
        );
        painter.drawRect(qRect);

        if (person.id > 0) {
            QString idText = QString::number(person.id);
            int textWidth = painter.fontMetrics().horizontalAdvance(idText);
            int textHeight = painter.fontMetrics().height();
            QRect textBgRect(qRect.left(), qRect.top() - textHeight - 2, textWidth + 4, textHeight);
            painter.fillRect(textBgRect, Qt::black);
            painter.setPen(Qt::white);
            painter.drawText(qRect.left() + 2, qRect.top() - 4, idText);
        }
    }
}

void VideoWidget::mousePressEvent(QMouseEvent *event) {
    if (!m_showVisuals || m_people.empty() || m_currentImage.isNull()) {
        QWidget::mousePressEvent(event);
        return;
    }

    float scaleX = (float)width() / m_currentImage.width();
    float scaleY = (float)height() / m_currentImage.height();

    for (int i = 0; i < (int)m_people.size(); ++i) {
        const auto& person = m_people[i];
        QRect qRect(
            person.box.x * scaleX,
            person.box.y * scaleY,
            person.box.width * scaleX,
            person.box.height * scaleY
        );

        if (qRect.contains(event->pos())) {
            emit boxClicked(i);
            return;
        }
    }

    QWidget::mousePressEvent(event);
}

QImage VideoWidget::matToQImage(const cv::Mat& mat) {
    if (mat.type() == CV_8UC3) {
        const uchar *qImageBuffer = (const uchar*)mat.data;
        QImage img(qImageBuffer, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return img.rgbSwapped();
    }
    return QImage();
}

