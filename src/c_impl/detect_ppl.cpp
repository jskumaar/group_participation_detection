//use detect_people_in_frames to get vector of bounding boxes for given frame 
// compilation command for me for reference:
//g++ "crop&save.cpp" -o crop_save \
    -I$(brew --prefix opencv)/include/opencv4 \
    -L$(brew --prefix opencv)/lib \
    -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_dnn -lopencv_imgcodecs -lopencv_videoio


#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::dnn;

// Constants
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.75;

// struct Detection {
//     Rect box;
//     float confidence;
//     int class_id;
// };

// struct PersonTrack {
//     int id;
//     Rect box;
//     VideoWriter writer;
    
// };

// Preprocess frame.
vector<Mat> pre_process(const Mat& input_image, Net& net) {
    Mat blob;
    blobFromImage(input_image, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);
    net.setInput(blob);

    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    return outputs;
}

// Postprocess outputs — no class name vector needed
vector<Detection> post_process(const Mat& input_image, vector<Mat>& outputs) {
    vector<Detection> results;
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    float* data = (float*)outputs[0].data;

    const int dimensions = 85;  // x, y, w, h, obj_conf, 80 class scores
    const int rows = 25200;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {
            float* classes_scores = data + 5;
            Point class_id;
            double max_class_score;
            minMaxLoc(Mat(1, 80, CV_32FC1, classes_scores), 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD) {
                int id = class_id.x;
                if (id == 0) {  // Only keep people
                    float cx = data[0];
                    float cy = data[1];
                    float w = data[2];
                    float h = data[3];
                    int left = int((cx - 0.5 * w) * x_factor);
                    int top = int((cy - 0.5 * h) * y_factor);
                    int width = int(w * x_factor);
                    int height = int(h * y_factor);

                    results.push_back({ Rect(left, top, width, height), confidence, id});
                }
            }
        }
        data += dimensions;
    }

    return results;
}



vector<Rect> detect_people_in_frame(const Mat &frame, Net &net) {
    vector<Mat> detections = pre_process(frame, net);
    vector<Detection> people = post_process(frame, detections);

    // Prepare for NMS
    vector<Rect> boxes;
    vector<float> confidences;
    for (const auto &det : people) {
        boxes.push_back(det.box);
        confidences.push_back(det.confidence);
    }

    // Apply NMS
    vector<int> indices;
    NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

    // Gather final boxes
    vector<Rect> final_boxes;
    for (int idx : indices) {
        final_boxes.push_back(boxes[idx]);
    }

    cout << "Detected " << final_boxes.size() << " people." << endl;
    for (size_t i = 0; i < final_boxes.size(); i++) {
        cout << "Person " << i << ": " << boxes[i] << endl;
        // rectangle(frame, boxes[i], Scalar(0, 255, 0), 2);
    }

    return final_boxes;
}



// for testing purposes
int main()
{
    cout << "main called dawg\n";

    // Load YOLOv5 ONNX model
    Net net = readNet("/Users/atind/real_time_gaze_detection/group_participation_detection/yolov5/yolov5s.onnx");

    Mat img = imread("/Users/atind/Desktop/test_img.png");

    vector<Rect> returned_boxes = detect_people_in_frame(img, net);
    return 0;
}