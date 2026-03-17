#pragma once

#include <fstream>
#include <string>

#include "vision/360_image_process.h"

namespace io {

class CsvGazeWriter {
public:
    CsvGazeWriter() = default;
    ~CsvGazeWriter();

    bool open(const std::string& path);
    void close();
    bool isOpen() const;

    void writeFrame(long frameNumber, const std::vector<PanoViewer::gaze>& gazes);
    void flush();

private:
    std::ofstream file_;
};

} // namespace io

