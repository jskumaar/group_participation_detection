// before running this file, create a conda enviornment and use python version 3.10 and run following command:    
// pip install torch==1.11.0 torchvision==0.12.0


// then run 
//python export.py --weights yolov5.pt --include onnx --opset 12

// resulting onnx version of model is what to use. 



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

struct Detection {
    Rect box;
    float confidence;
    int class_id;
};

struct PersonTrack {
    int id;
    Rect box;
    VideoWriter writer;
    
};

// Preprocess frame.
vector<Mat> pre_process(Mat& input_image, Net& net) {
    Mat blob;
    blobFromImage(input_image, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);
    net.setInput(blob);

    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    return outputs;
}

// Postprocess outputs — no class name vector needed
vector<Detection> post_process(Mat& input_image, vector<Mat>& outputs) {
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

int main() {
    std::cout << "main called dawg \n";
    Net net = readNet("yolov5/yolov5s.onnx");

    string vid_path = "/Applications/demo_1_3_min.mp4"
    VideoCapture cap(vid_path);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open video file." << endl;
        return -1;
    }

    double fps = cap.get(CAP_PROP_FPS);
    int width = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);

    Mat frame;
    vector<PersonTrack> people_tracks;
    int person_id = 0;

    // Detect people in first frame
    cap.read(frame);
    vector<Mat> detections = pre_process(frame, net);
    vector<Detection> people = post_process(frame, detections);

    vector<Rect> boxes;
    vector<float> confidences;
    for (const auto& det : people) {
            boxes.push_back(det.box);
            confidences.push_back(det.confidence);
        }


    // nms to remove boxes that are for same person
    vector<int> indices;
    NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

    vector<Detection> nms_people;
    for (int idx : indices) {
        nms_people.push_back(people[idx]);
    }


    // open writer for people in first frame 
    for (auto& person : nms_people) {
        
        Rect box = person.box & Rect(0, 0, frame.cols, frame.rows);
        if (box.width <= 0 || box.height <= 0) continue;
    
        string filename = "person_" + to_string(person_id) + "_3_min_c++.mp4";
        VideoWriter writer(filename, VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(640, 480));
        if (!writer.isOpened()) {
            cerr << "Failed to open writer for: " << filename << endl;
            continue;
        }
        people_tracks.push_back({person_id++, box, writer });
        
    }
   

    // Process every frame using the fixed boxes from frame 1
    while (cap.read(frame)) {
        for (auto& track : people_tracks) {
             Rect safe_box = track.box & Rect(0, 0, frame.cols, frame.rows);
            if (safe_box.width > 0 && safe_box.height > 0) {
                Mat cropped = frame(safe_box).clone();
                resize(cropped, cropped, Size(640, 480));
                track.writer.write(cropped);
            }
        }
    }

    for (auto& track : people_tracks)
        track.writer.release();

    cap.release();
    destroyAllWindows();
    std::cout << "Done!" << endl;
    return 0;
}
