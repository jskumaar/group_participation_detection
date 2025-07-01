// Capture and stitch realtime using Insta360 SDK
// Note: This is a work in progress and may not compile or run as expected.

#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <camera/camera.h>
#include <camera/device_discovery.h>
#include <camera/photography_settings.h>

#ifdef _WIN32
#include <windows.h>
#endif

class TestStreamDelegate : public ins_camera::StreamDelegate {
public:
    void OnAudioData(const uint8_t* data, size_t size, int64_t timestamp) override {
        // No-op
    }

    void OnVideoData(const uint8_t* data, size_t size, int64_t timestamp, uint8_t streamType, int stream_index) override {
        std::cout << "[Stream] Video data received. size=" << size << ", ts=" << timestamp << std::endl;
    }

    void OnGyroData(const std::vector<ins_camera::GyroData>& data) override {
        std::cout << "[Stream] Gyro data received. size=" << data.size() << std::endl;
    }

    void OnExposureData(const ins_camera::ExposureData& data) override {
        std::cout << "[Stream] Exposure data received." << std::endl;
    }
};

int main(int argc, char** argv) {
    std::cout << ">>> Starting Insta360 SDK minimal test" << std::endl;
    ins_camera::SetLogLevel(ins_camera::LogLevel::ERR);

    if (argc > 1 && std::string(argv[1]) == "--debug") {
        ins_camera::SetLogLevel(ins_camera::LogLevel::VERBOSE);
    }

    ins_camera::DeviceDiscovery discovery;
    auto devices = discovery.GetAvailableDevices();

    std::cout << ">>> Found " << devices.size() << " camera(s)" << std::endl;
    if (devices.empty()) return 1;

    auto cam = std::make_shared<ins_camera::Camera>(devices[0].info);
    if (!cam->Open()) {
        std::cerr << "Failed to open camera." << std::endl;
        return 1;
    }

    std::cout << ">>> Succeed to open camera..." << std::endl;
    std::shared_ptr<ins_camera::StreamDelegate> delegate = std::make_shared<TestStreamDelegate>();
    cam->SetStreamDelegate(delegate);

    ins_camera::LiveStreamParam param;
    param.video_resolution = ins_camera::VideoResolution::RES_3840_1920P30;
    param.lrv_video_resulution = ins_camera::VideoResolution::RES_1440_720P30;
    param.video_bitrate = 1024 * 1024 / 2;
    param.enable_audio = false;
    param.using_lrv = false;

    std::cout << ">>> Starting live stream..." << std::endl;
    if (!cam->StartLiveStreaming(param)) {
        std::cerr << "Failed to start live stream." << std::endl;
        return 1;
    }

    std::this_thread::sleep_for(std::chrono::seconds(10));
    cam->StopLiveStreaming();
    cam->Close();
    std::cout << ">>> Live stream stopped and camera closed." << std::endl;
    return 0;
}
