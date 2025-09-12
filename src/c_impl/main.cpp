#include "360_image_process.h"
#include "tracker.h"

#include "SORT.h"
#include <iostream>
#include <vector>
#include <cstdio>

#define M_PI 3.14159265358979323846

PanoViewer viewer;
OPNetTracker tracker;

#define GLOBAL_FRAME_WIDTH 2880
#define GLOBAL_FRAME_HEIGHT 1440
#define num_people 3


#define localizer_confidence 0.0

/*
Run yolo
run SORT
    */

void visualize(FILE* gp, int R, std::vector<PanoViewer::gaze> gazes){

    // Draw loop (could be in a while(true) for realtime)
    fprintf(gp, "splot '-' with points pt 7 lc rgb 'black' title 'Sphere',"
                "'-' with points pt 7 ps 1.5 lc rgb 'red' title 'People',"
                "'-' with vectors nohead lc rgb 'blue' title 'Gaze'\n");

    // Sphere points (mesh)
    int n_theta = 40, n_phi = 20;
    for (int i = 0; i <= n_phi; ++i) {
        double phi = M_PI * i / n_phi;
        for (int j = 0; j <= n_theta; ++j) {
            double theta = 2 * M_PI * j / n_theta;
            double x = R * std::sin(phi) * std::cos(theta);
            double y = R * std::cos(phi);
            double z = R * std::sin(phi) * std::sin(theta);
            fprintf(gp, "%f %f %f\n", x, y, z);
        }
    }
    fprintf(gp, "e\n");

    // People positions
    for (auto& p : gazes)
        fprintf(gp, "%f %f %f\n", p.start.x, p.start.y, p.start.z);
    fprintf(gp, "e\n");

    for (auto& p : gazes) {
        double gx = p.start.x + p.direction[0] * 100;
        double gy = p.start.y + p.direction[1] * 100; 
        double gz = p.start.z + p.direction[2] * 100;
        fprintf(gp, "%f %f %f %f %f %f\n", p.start.x, p.start.y, p.start.z, gx - p.start.x, gy - p.start.y, gz - p.start.z);
    }
    fprintf(gp, "e\n"); 
    fflush(gp);
}

//void initialize_sequence(&std::vector<cv::Mat> frames){
//
// }

int main() {

    // change based on where ur vid is located
    char* vid_path = "demo_1_stitched.mp4";
    cv::VideoCapture cap(vid_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open camera input 1. Trying input 0..." << std::endl;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, GLOBAL_FRAME_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, GLOBAL_FRAME_HEIGHT);

    cv::Mat pano_frame;

    cv::namedWindow("360° Viewer", cv::WINDOW_NORMAL);  // Changed from WINDOW_AUTOSIZE
    cv::resizeWindow("360° Viewer", 640, 480);



    // FILE* gp = popen("gnuplot -persistent", "w");
    // USE ONE BELOW FOR WINDOWS 
    FILE* gp = _popen("gnuplot -persistent", "w");

    if (!gp) { std::cerr << "Failed to start gnuplot\n"; return 1; }

    // Set up 3D plot
    fprintf(gp, "set term wxt size 800,600\n");
    fprintf(gp, "set view equal xyz\n");
    fprintf(gp, "set view 60, 30\n");
    fprintf(gp, "set xrange [-%d:%d]\n", 100, 100);
    fprintf(gp, "set yrange [-%d:%d]\n", 100, 100);
    fprintf(gp, "set zrange [-%d:%d]\n", 100, 100);
    fprintf(gp, "set ticslevel 0\n");

    fprintf(gp, "set xlabel 'X (Right)'\n");
    fprintf(gp, "set ylabel 'Y (Up)'\n");
    fprintf(gp, "set zlabel 'Z (Forward)'\n");

    std::vector<PanoViewer::gaze> gazes;
    size_t frame_count = 0;
    cv::Mat perspective_view;
    std::vector<cv::Rect> people;

    // array of gaze_windows based on index per person
    std::vector<PanoViewer::gaze_window> gaze_windows(num_people);
    for (int i = 0; i < num_people; i++) {
        gaze_windows[i].personID = i;
        gaze_windows[i].max_window_length = 20; 
        gaze_windows[i].counts.resize(num_people, 0);
        gaze_windows[i].sights.clear();
    }


    Sort object_tracker;

    while (true) {
        cap >> pano_frame;
        cv::resize(pano_frame, pano_frame, cv::Size(2880,1440)); // new width, height


        if (pano_frame.empty()) {
            std::cerr << "Failed to capture frame!" << std::endl;
            break;
        }
        // Ensure we have the right aspect ratio (2:1 for equirectangular)
        if (pano_frame.cols / (float)pano_frame.rows != 2.0f) {
            printf("Error: Incorrect aspect ratio\n");
            exit(1);
        }
        if(frame_count % 10 == 0){
            printf("Running YOLO on frame %zu\n", frame_count);
            people = tracker.run_yolo(pano_frame);
        }
        
        
        std::vector<Sort::Track> tracks = object_tracker.update(people);
        
        
        // Generate perspective view
         // Counter for unique window names
        //person 0 = suresh
        //person 1 = naveen
        //person 2 = other
        cv::Scalar color;
        for(const Sort::Track& track : tracks){
            int person_id = track.id - 1; // IDs start at 1
            cv::Rect person = track.box;
            if(person_id == 0){
                viewer.setFOV(55);
                color = cv::Scalar (255, 0, 0);
            }
            else if(person_id == 1){
                viewer.setFOV(30);
                color = cv::Scalar (0, 255, 0);
            }
            else if(person_id == 2){
                viewer.setFOV(40);
                color = cv::Scalar (0, 0, 255);
            }
        // Generate perspective view
            int yaw = ((person.x + person.width / 2) * 0.125) - 180.0;
            viewer.setYaw(yaw);
            perspective_view = viewer.generatePerspectiveView(pano_frame);
            rotation_output pose = tracker.run(perspective_view);
            printf("personID: %d, yaw: %f, pitch: %f\n", person_id, pose.yaw, pose.pitch);
            PanoViewer::gaze gaze = viewer.addGaze(person_id, pose.yaw, pose.pitch, person);
            gazes.push_back(gaze);
            auto pano_pixel = viewer.rayToPanoPixel(cv::Vec3f(gaze.start.x, gaze.start.y, gaze.start.z),
                                  gaze.direction,
                                  90.0f,
                                  GLOBAL_FRAME_WIDTH,
                                  GLOBAL_FRAME_HEIGHT);

            printf("Pano Pixel: x=%d, y=%d\n", pano_pixel.x, pano_pixel.y);
            // calculates if gaze_pt intersects others' bboxes

            // just draws scaled box, for visualization and testing purposes
            double scale_factor = 0.2;
            int dw = static_cast<int>(person.width  * scale_factor);
            int dh = static_cast<int>(person.height * scale_factor);
         // Create new rect centered on the same point
            cv::Rect scaled( person.x - dw / 2, person.y - dh / 2, person.width  + dw, person.height + dh);
            cv::rectangle(pano_frame, scaled, cv::Scalar(255,0,0), 2);

            
            int intersect = PanoViewer::bbox_intersections(cv::Point(pano_pixel.x, pano_pixel.y), cv::Point(person.x, person.y), tracks);
            cv::circle(pano_frame, pano_pixel, 40, color, 2);
            std::string text = "Person: " + std::to_string(person_id) + " looking at person: " + std::to_string(intersect);
            cv::putText(pano_frame, text ,  cv::Point(person.x, person.y), cv::FONT_HERSHEY_SIMPLEX,  1.0,  color, 2);


            cv::rectangle(pano_frame, person, cv::Scalar(0, 255, 0), 2);  // Green rectangle, 2px thick

            PanoViewer::add_frame_to_window(gaze_windows[person_id], person_id, intersect);
            PanoViewer::print_gaze_window(gaze_windows[person_id], pano_frame, scaled);
            person_id++;
            perspective_view.release();
        }
        visualize(gp, 90, gazes);
        gazes.clear();
        imshow("360° Viewer", pano_frame);
        char key = cv::waitKey(1) & 0xFF;
        if (key == 27) break; // ESC to exit
        frame_count++;
    }



    // pclose(gp);
    // USE ONE BELOW FOR WINDOWS
    _pclose(gp);
    cap.release();
    cv::destroyAllWindows();
    return 0;
}   
