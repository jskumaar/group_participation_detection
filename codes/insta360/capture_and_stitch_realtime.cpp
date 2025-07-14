// #include <iostream>
// #include <atomic>
// #include <thread>
// #include <mutex>
// #include <condition_variable>
// #include <filesystem>
// #include <camera/camera.h>
// #include <camera/device_discovery.h>
// #include <stitcher/ins_realtime_stitcher.h>
// #include <opencv2/opencv.hpp>

// namespace fs = std::filesystem;

// std::shared_ptr<ins::RealTimeStitcher> g_stitcher;
// cv::Mat g_stitched, g_fisheye0, g_fisheye1;
// std::mutex g_mtx;
// std::atomic<bool> g_run(true);

// std::string ts_name(int64_t ts, const std::string& prefix, const std::string& ext)
// {
//     return prefix + "_" + std::to_string(ts) + ext;
// }

// class DualStreamDelegate : public ins_camera::StreamDelegate
// {
// public:
//     DualStreamDelegate(std::shared_ptr<ins::RealTimeStitcher> st) : st_(std::move(st)) {}

//     void OnVideoData(const uint8_t* data, size_t size, int64_t ts,
//                      uint8_t streamType, int stream_index) override
//     {
//         std::cout << "[Delegate] VideoData received. ts: " << ts << ", stream index: " << (int)stream_index << ", size: " << size << std::endl;

//         std::string fname = ts_name(ts,
//                                     stream_index == 0 ? "fisheye0" : "fisheye1",
//                                     ".h265");
//         std::ofstream out(fname, std::ios::binary);
//         out.write(reinterpret_cast<const char*>(data), size);

//         st_->HandleVideoData(data, size, ts, streamType, stream_index);
//     }

//     void OnGyroData(const std::vector<ins_camera::GyroData>& d) override
//     {
//         std::cout << "[Delegate] GyroData received. Count: " << d.size() << std::endl;

//         std::vector<ins::GyroData> converted(d.size());
//         std::memcpy(converted.data(), d.data(), d.size() * sizeof(ins_camera::GyroData));
//         st_->HandleGyroData(converted);
//     }

//     void OnExposureData(const ins_camera::ExposureData& d) override
//     {
//         std::cout << "[Delegate] ExposureData received. Exposure time: " << d.exposure_time << ", ts: " << d.timestamp << std::endl;

//         ins::ExposureData e{};
//         e.exposure_time = d.exposure_time;
//         e.timestamp     = d.timestamp;
//         st_->HandleExposureData(e);
//     }

//     void OnAudioData(const uint8_t* data, size_t size, int64_t timestamp) override {
//         // no-op
//     }

// private:
//     std::shared_ptr<ins::RealTimeStitcher> st_;
// };

// void viewer()
// {
//     std::cout << "[Viewer] Thread started\n";
//     while (g_run)
//     {
//         cv::Mat frame;
//         {
//             std::lock_guard<std::mutex> lk(g_mtx);
//             if (!g_stitched.empty()) frame = g_stitched.clone();
//         }
//         if (!frame.empty())
//         {
//             std::cout << "[Viewer] Displaying frame\n";
//             cv::imshow("Stitched Preview", frame);
//             int key = cv::waitKey(1);
//             if (key == 'q' || key == 'Q') {
//                 std::cout << "[Viewer] Quit key pressed\n";
//                 g_run = false;
//             }
//         }
//         std::this_thread::sleep_for(std::chrono::milliseconds(10));
//     }
// }

// int main()
// {
//     std::cout << "[Main] Setting log level...\n";
//     ins_camera::SetLogLevel(ins_camera::LogLevel::INFO);

//     std::cout << "[Main] Discovering devices...\n";
//     ins_camera::DeviceDiscovery dd;
//     auto devices = dd.GetAvailableDevices();
//     if (devices.empty()) {
//         std::cerr << "[Main] No camera found\n";
//         return 1;
//     }

//     std::cout << "[Main] Found camera: " << devices[0].serial_number << std::endl;

//     auto cam = std::make_shared<ins_camera::Camera>(devices[0].info);
//     if (!cam->Open()) {
//         std::cerr << "[Main] Failed to open camera\n";
//         return 1;
//     }
//     std::cout << "[Main] Camera opened successfully\n";

//     auto prev = cam->GetPreviewParam();
//     g_stitcher = std::make_shared<ins::RealTimeStitcher>();

//     std::cout << "[Main] Setting camera info...\n";
//     ins::CameraInfo ci;
//     ci.cameraName            = prev.camera_name;
//     ci.decode_type           = static_cast<ins::VideoDecodeType>(prev.encode_type);
//     ci.offset                = prev.offset;
//     ci.window_crop_info_ = {
//         static_cast<uint32_t>(prev.crop_info.crop_offset_x),
//         static_cast<uint32_t>(prev.crop_info.crop_offset_y),
//         static_cast<uint32_t>(prev.crop_info.dst_width),
//         static_cast<uint32_t>(prev.crop_info.dst_height),
//         static_cast<int32_t>(prev.crop_info.src_width),
//         static_cast<int32_t>(prev.crop_info.src_height)
//     };
//     ci.gyro_timestamp        = prev.gyro_timestamp;
//     g_stitcher->SetCameraInfo(ci);

//     std::cout << "[Main] Setting stitch type and AI model...\n";
//     g_stitcher->SetStitchType(ins::STITCH_TYPE::AIFLOW);
//     g_stitcher->SetAiStitchModelFile("sdk/MediaSDK/modelfile/ai_stitch_model_v1.ins");

//     g_stitcher->SetStitchRealTimeDataCallback([](uint8_t* yuv[4], int ls[4], int w, int h, int fmt, int64_t ts)
//     {
//         std::cout << "[Stitcher] Frame stitched: " << ts << ", size: " << w << "x" << h << std::endl;
//         std::lock_guard<std::mutex> lk(g_mtx);
//         g_stitched = cv::Mat(h, w, CV_8UC4, yuv[0]).clone();
//         cv::imwrite(ts_name(ts, "stitched", ".png"), g_stitched);
//     });

//     std::cout << "[Main] Setting stream delegate...\n";
//     std::shared_ptr<ins_camera::StreamDelegate> delegate = std::make_shared<DualStreamDelegate>(g_stitcher);
//     cam->SetStreamDelegate(delegate);

//     std::cout << "[Main] Starting live stream...\n";
//     ins_camera::LiveStreamParam lp;
//     lp.video_resolution      = ins_camera::VideoResolution::RES_1920_960P30;
//     lp.lrv_video_resulution  = ins_camera::VideoResolution::RES_1920_960P30;
//     lp.video_bitrate         = static_cast<uint32_t>(512 * 1024);
//     lp.enable_audio          = false;
//     lp.using_lrv             = false;

//     if (!cam->StartLiveStreaming(lp)) {
//         std::cerr << "[Main] Failed to start live streaming\n";
//         return 1;
//     }

//     std::cout << "[Main] Live streaming started\n";
//     g_stitcher->StartStitch();
//     std::cout << "[Main] Stitching started\n";

//     std::thread t(viewer);

//     std::cout << "Controls:  r=start record  s=stop record  q=quit\n";
//     while (g_run)
//     {
//         int key = cv::waitKey(0);
//         if (key == 'r' || key == 'R')
//         {
//             std::cout << "[Main] Start recording requested\n";
//             if (!cam->StartRecording()) std::cerr << "[Main] StartRecording failed\n";
//             else  std::cout << "[Main] Recording started\n";
//         }
//         else if (key == 's' || key == 'S')
//         {
//             std::cout << "[Main] Stop recording requested\n";
//             auto url = cam->StopRecording();
//             if (url.Empty()) std::cerr << "[Main] StopRecording failed\n";
//             else if (url.IsSingleOrigin()) {
//             std::cout << "[Main] Recording stopped. File: " << url.GetSingleOrigin() << std::endl;
//             } 
//             else {
//                 std::cout << "[Main] Recording stopped. Multiple files:\n";
//                 for (const auto& u : url.OriginUrls()) {
//                     std::cout << "  - " << u << std::endl;
//                 }
//             }

//         }
//         else if (key == 'q' || key == 'Q')
//         {
//             std::cout << "[Main] Quit requested\n";
//             g_run = false;
//         }
//     }

//     std::cout << "[Main] Cleaning up...\n";
//     t.join();
//     cam->StopLiveStreaming();
//     g_stitcher->CancelStitch();
//     cam->Close();
//     std::cout << "[Main] Shutdown complete\n";
//     return 0;
// }


/////////////////////////////////////////////////


// #include <iostream>
// #include <stitcher/ins_realtime_stitcher.h>
// #include <camera/camera.h>
// #include <camera/device_discovery.h>

// #include <iostream>
// #include <condition_variable>
// #include <mutex>
// #include <thread>
// #include <chrono>
// #include <vector>
// #include <sstream>
// #include <opencv2/opencv.hpp>

// const std::string window_name = "realtime_stitcher";

// std::vector<std::string> split(const std::string& s, char delimiter) {
//     std::vector<std::string> tokens;
//     std::string token;
//     std::istringstream tokenStream(s);

//     while (std::getline(tokenStream, token, delimiter)) {
//         tokens.push_back(token);
//     }

//     return tokens;
// }

// class StitchDelegate : public ins_camera::StreamDelegate {
// public:
//     StitchDelegate(const std::shared_ptr<ins::RealTimeStitcher>& stitcher) :stitcher_(stitcher) {
//     }

//     virtual ~StitchDelegate() {
//     }

//     void OnAudioData(const uint8_t* data, size_t size, int64_t timestamp) override {}

//     void OnVideoData(const uint8_t* data, size_t size, int64_t timestamp, uint8_t streamType, int stream_index) override {
//         stitcher_->HandleVideoData(data, size, timestamp, streamType, stream_index);
//     }

//     void OnGyroData(const std::vector<ins_camera::GyroData>& data) override {
//         std::vector<ins::GyroData> data_vec(data.size());
//         memcpy(data_vec.data(), data.data(), data.size() * sizeof(ins_camera::GyroData));
//         stitcher_->HandleGyroData(data_vec);
//     }

//     void OnExposureData(const ins_camera::ExposureData& data) override {
//         ins::ExposureData exposure_data{};
//         exposure_data.exposure_time = data.exposure_time;
//         exposure_data.timestamp = data.timestamp;
//         stitcher_->HandleExposureData(exposure_data);
//     }

// private:
//     std::shared_ptr<ins::RealTimeStitcher> stitcher_;
// };

// int main(int argc, char* argv[]) {
//     ins::InitEnv();
//     std::cout << "begin open camera" << std::endl;
//     ins_camera::SetLogLevel(ins_camera::LogLevel::WARNING);
//     ins::SetLogLevel(ins::InsLogLevel::WARNING);
//     for (int i = 1; i < argc; i++) {
//         const std::string arg = argv[i];
//         if (arg == std::string("--debug")) {
//             ins_camera::SetLogLevel(ins_camera::LogLevel::VERBOSE);
//         }
//         else if (arg == std::string("--log_file")) {
//             const std::string log_file = argv[++i];
//             ins_camera::SetLogPath(log_file);
//         }
//     }

//     ins_camera::DeviceDiscovery discovery;
//     auto list = discovery.GetAvailableDevices();
//     if (list.empty()) {
//         std::cerr << "no device found." << std::endl;
//         discovery.FreeDeviceDescriptors(list);
//         return -1;
//     }

//     for (const auto& camera : list) {
//         std::cout << "serial:" << camera.serial_number << "\t"
//             << ";camera type:" << camera.camera_name << "\t"
//             << ";fw version:" << camera.fw_version << "\t"
//             << std::endl;
//     }

//     auto cam = std::make_shared<ins_camera::Camera>(list[0].info);
//     if (!cam->Open()) {
//         std::cerr << "failed to open camera" << std::endl;
//         return -1;
//     }

//     const auto serial_number = list[0].serial_number;

//     discovery.FreeDeviceDescriptors(list);

//     cv::Mat show_image_;
//     std::thread show_thread_;
//     std::mutex show_image_mutex_;
//     bool is_stop_ = true;
//     std::condition_variable show_image_cond_;

//     std::shared_ptr<ins::RealTimeStitcher> stitcher = std::make_shared<ins::RealTimeStitcher>();
//     ins::CameraInfo camera_info;
//     auto preview_param = cam->GetPreviewParam();
//     camera_info.cameraName = preview_param.camera_name;
//     camera_info.decode_type = static_cast<ins::VideoDecodeType>(preview_param.encode_type);
//     camera_info.offset = preview_param.offset;
//     auto window_crop_info = preview_param.crop_info;
//     camera_info.window_crop_info_.crop_offset_x = window_crop_info.crop_offset_x;
//     camera_info.window_crop_info_.crop_offset_y = window_crop_info.crop_offset_y;
//     camera_info.window_crop_info_.dst_width = window_crop_info.dst_width;
//     camera_info.window_crop_info_.dst_height = window_crop_info.dst_height;
//     camera_info.window_crop_info_.src_width = window_crop_info.src_width;
//     camera_info.window_crop_info_.src_height = window_crop_info.src_height;
//     camera_info.gyro_timestamp = preview_param.gyro_timestamp;

//     stitcher->SetCameraInfo(camera_info);
//     stitcher->SetStitchType(ins::STITCH_TYPE::DYNAMICSTITCH);
//     stitcher->EnableFlowState(true);
//     stitcher->SetOutputSize(960, 480);
//     stitcher->SetStitchRealTimeDataCallback([&](uint8_t* data[4], int linesize[4], int width, int height, int format, int64_t timestamp) {
//         std::unique_lock<std::mutex> lck(show_image_mutex_);
//         show_image_ = cv::Mat(height, width, CV_8UC4, data[0]).clone();
//         show_image_cond_.notify_one();
//     });

//     std::shared_ptr<ins_camera::StreamDelegate> delegate = std::make_shared<StitchDelegate>(stitcher);
//     cam->SetStreamDelegate(delegate);

//     std::cout << "Succeed to open camera..." << std::endl;

//     std::cout << "Usage:" << std::endl;
//     std::cout << "1: start preview live streaming:" << std::endl;
//     std::cout << "2: stop preview live streaming:" << std::endl;

//     int option = 0;
//     while (true) {
//         std::cout << "please enter index: ";
//         std::cin >> option;
//         if (option < 0 || option > 39) {
//             std::cout << "Invalid index" << std::endl;
//             continue;
//         }

//         if (option == 0) {
//             break;
//         }

//         if (option == 1) {
//             if (!is_stop_) {
//                 std::cout << "" << std::endl;
//                 continue;
//             }
//             ins_camera::LiveStreamParam param;
//             param.video_resolution = ins_camera::VideoResolution::RES_1440_720P30;
//             param.lrv_video_resulution = ins_camera::VideoResolution::RES_1440_720P30;
//             param.video_bitrate = 1024 * 1024 / 2;
//             param.enable_audio = false;
//             param.using_lrv = false;
//             if (cam->StartLiveStreaming(param)) {
//                 stitcher->StartStitch();
//                 std::cout << "successfully started live stream" << std::endl;
//             }

//             show_thread_ = std::thread([&]() {
//                 cv::namedWindow(window_name, cv::WINDOW_NORMAL);
//                 is_stop_ = false;
//                 while (!is_stop_)
//                 {
//                     std::unique_lock<std::mutex> lck(show_image_mutex_);
//                     show_image_cond_.wait(lck, [&]() {
//                         return  is_stop_ || !show_image_.empty();
//                     });

//                     if (is_stop_) {
//                         break;
//                     }

//                     auto temp = show_image_.clone();
//                     show_image_ = cv::Mat();
//                     lck.unlock();
//                     cv::cvtColor(temp, temp, cv::COLOR_RGBA2BGRA);
//                     cv::imshow(window_name, temp);
//                     cv::waitKey(5);
//                 }
//             });
//         }

//         if (option == 2) {
//             std::unique_lock<std::mutex> lck(show_image_mutex_);
//             is_stop_ = true;
//             show_image_cond_.notify_one();
//             lck.unlock();
//             if (show_thread_.joinable()) {
//                 show_thread_.join();
//             }
//             cv::destroyWindow(window_name);
//             if (cam->StopLiveStreaming()) {
//                 stitcher->CancelStitch();
//                 std::cout << "success!" << std::endl;
//             }
//             else {
//                 std::cerr << "failed to stop live." << std::endl;
//             }
//         }
//     }
//     std::unique_lock<std::mutex> lck(show_image_mutex_);
//     is_stop_ = true;
//     show_image_cond_.notify_one();
//     lck.unlock();
//     if (show_thread_.joinable()) {
//         show_thread_.join();
//     }
//     cv::destroyWindow(window_name);
//     cam->Close();
//     return 0;
// }


///////////////////////////////


/*******************************************************
 * Insta360 live preview + recorder + saver (with prints)
 *******************************************************/

#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <filesystem>
#include <camera/camera.h>
#include <camera/device_discovery.h>
#include <stitcher/ins_realtime_stitcher.h>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

std::shared_ptr<ins::RealTimeStitcher> g_stitcher;
cv::Mat g_stitched;
std::mutex g_mtx;
std::atomic<bool> g_run(true);

std::string ts_name(int64_t ts, const std::string& prefix, const std::string& ext)
{
    return prefix + "_" + std::to_string(ts) + ext;
}

class DualStreamDelegate : public ins_camera::StreamDelegate
{
public:
    explicit DualStreamDelegate(std::shared_ptr<ins::RealTimeStitcher> st) : st_(std::move(st)) {}

    void OnVideoData(const uint8_t* data, size_t size, int64_t ts,
                     uint8_t streamType, int stream_index) override
    {
        std::cout << "[Delegate] Video ts=" << ts << " idx=" << stream_index << " size=" << size << '\n';
        st_->HandleVideoData(data, size, ts, streamType, stream_index);
    }

    void OnGyroData(const std::vector<ins_camera::GyroData>& d) override
    {
        std::vector<ins::GyroData> v(d.size());
        std::memcpy(v.data(), d.data(), d.size() * sizeof(ins_camera::GyroData));
        st_->HandleGyroData(v);
    }

    void OnExposureData(const ins_camera::ExposureData& d) override
    {
        ins::ExposureData e{d.exposure_time, d.timestamp};
        st_->HandleExposureData(e);
    }

    void OnAudioData(const uint8_t*, size_t, int64_t) override {}

private:
    std::shared_ptr<ins::RealTimeStitcher> st_;
};

void viewer()
{
    std::cout << "[Viewer] thread started\n";
    while (g_run) {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lk(g_mtx);
            if (!g_stitched.empty()) frame = g_stitched.clone();
        }
        if (!frame.empty()) {
            cv::imshow("Stitched Preview", frame);
            int k = cv::waitKey(1);
            if (k == 'q' || k == 'Q') g_run = false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

int main()
{
    std::cout << "[Main] Initialising MediaSDK...\n";
    ins::InitEnv();                                // **** REQUIRED ****
    std::cout << "[Main] MediaSDK initialised\n";

    ins_camera::SetLogLevel(ins_camera::LogLevel::INFO);

    ins_camera::DeviceDiscovery dd;
    auto devs = dd.GetAvailableDevices();
    std::cout << "[Main] devices=" << devs.size() << '\n';
    if (devs.empty()) return 1;

    auto cam = std::make_shared<ins_camera::Camera>(devs[0].info);
    if (!cam->Open()) { std::cerr << "open failed\n"; return 1; }

    auto prev = cam->GetPreviewParam();
    g_stitcher = std::make_shared<ins::RealTimeStitcher>();

    ins::CameraInfo ci;
    ci.cameraName  = prev.camera_name;
    ci.decode_type = static_cast<ins::VideoDecodeType>(prev.encode_type);
    ci.offset      = prev.offset;
    ci.window_crop_info_ = {
        static_cast<uint32_t>(prev.crop_info.crop_offset_x), static_cast<uint32_t>(prev.crop_info.crop_offset_y),
        static_cast<uint32_t>(prev.crop_info.dst_width),     static_cast<uint32_t>(prev.crop_info.dst_height),
        static_cast<int32_t>(prev.crop_info.src_width),
        static_cast<int32_t>(prev.crop_info.src_height)};
    ci.gyro_timestamp = prev.gyro_timestamp;
    g_stitcher->SetCameraInfo(ci);

    g_stitcher->SetStitchType(ins::STITCH_TYPE::DYNAMICSTITCH);      // start simple
    g_stitcher->SetStitchRealTimeDataCallback([](uint8_t* yuv[4], int*, int w, int h, int, int64_t ts){
        std::cout << "[Stitcher] stitched frame ts=" << ts << '\n';
        std::lock_guard<std::mutex> lk(g_mtx);
        g_stitched = cv::Mat(h, w, CV_8UC4, yuv[0]).clone();
    });

    // Stream delegate
    std::cout << "[Main] Setting stream delegate...\n";
    std::shared_ptr<ins_camera::StreamDelegate> delegate = std::make_shared<DualStreamDelegate>(g_stitcher);
    cam->SetStreamDelegate(delegate);

    ins_camera::LiveStreamParam lp;
    lp.video_resolution     = ins_camera::VideoResolution::RES_1920_960P30;
    lp.lrv_video_resulution = ins_camera::VideoResolution::RES_1920_960P30;
    lp.video_bitrate        = 512 * 1024;

    std::cout << "[Main] Starting live stream...\n";
    if (!cam->StartLiveStreaming(lp)) { std::cerr << "stream start failed\n"; return 1; }
    std::cout << "[Main] live stream started\n";

    g_stitcher->StartStitch();
    std::cout << "[Main] StartStitch() called\n";

    std::thread v(viewer);

    while (g_run) {
        int k = cv::waitKey(0);
        if (k == 'q' || k == 'Q') g_run = false;
    }

    v.join();
    cam->StopLiveStreaming();
    g_stitcher->CancelStitch();
    cam->Close();
    return 0;
}
