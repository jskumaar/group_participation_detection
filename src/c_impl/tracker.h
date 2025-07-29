#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <filesystem>

class OPNetTracker {
    public:
        OPNetTracker() = default;
        ~OPNetTracker() = default;
    private:
        bool initialize();
        Ort::Env env{nullptr};
        Ort::MemoryInfo allocator_info_{nullptr};
        std::optional<Localizer> localizer_;
}