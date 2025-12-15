#include "videowidget.h"
#include <QDebug>

VideoWidget::VideoWidget(QWidget *parent) : QWidget(parent) {
    setAttribute(Qt::WA_OpaquePaintEvent); // Optimize painting
}

void VideoWidget::updateFrame(const cv::Mat& frame) {
    if (frame.empty()) return;
    m_currentImage = matToQImage(frame);
    update(); // Trigger repaint
}

void VideoWidget::updateOverlays(const std::vector<PersonBox>& people) {
    m_people = people;
    if (m_showVisuals) {
        update();
    }
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

    // Draw video frame scaled to window size
    painter.drawImage(rect(), m_currentImage);

    if (m_showVisuals) {
        // Calculate scale factors
        float scaleX = (float)width() / m_currentImage.width();
        float scaleY = (float)height() / m_currentImage.height();

        // Draw Bounding Boxes
        painter.setPen(QPen(Qt::green, 2));
        painter.setFont(QFont("Arial", 12, QFont::Bold));
        
        for (const auto& person : m_people) {
            QRect qRect(
                person.box.x * scaleX, 
                person.box.y * scaleY, 
                person.box.width * scaleX, 
                person.box.height * scaleY
            );
            painter.drawRect(qRect);
            
            // Draw ID above the box
            QString idText = QString::number(person.id);
            int textWidth = painter.fontMetrics().horizontalAdvance(idText);
            int textHeight = painter.fontMetrics().height();
            
            // Background for text for better visibility
            QRect textBgRect(qRect.left(), qRect.top() - textHeight - 2, textWidth + 4, textHeight);
            painter.fillRect(textBgRect, Qt::black);
            
            painter.setPen(Qt::white);
            painter.drawText(qRect.left() + 2, qRect.top() - 4, idText);
            
            // Reset pen for next box
            painter.setPen(QPen(Qt::green, 2));
        }
    }
}

QImage VideoWidget::matToQImage(const cv::Mat& mat) {
    if (mat.type() == CV_8UC3) {
        // Copy input Mat
        const uchar *qImageBuffer = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage img(qImageBuffer, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return img.rgbSwapped(); // BGR to RGB
    }
    return QImage();
}
