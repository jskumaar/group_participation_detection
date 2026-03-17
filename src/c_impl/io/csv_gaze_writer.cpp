#include "csv_gaze_writer.h"

namespace io {

CsvGazeWriter::~CsvGazeWriter() {
    close();
}

bool CsvGazeWriter::open(const std::string& path) {
    close();
    file_.open(path);
    if (!file_.is_open()) return false;
    file_ << "Frame,PersonID,BoxCenterX,BoxCenterY,"
             "GazeStartX,GazeStartY,GazeStartZ,"
             "GazeDirX,GazeDirY,GazeDirZ,LookingAtIDs\n";
    return true;
}

void CsvGazeWriter::close() {
    if (file_.is_open()) file_.close();
}

bool CsvGazeWriter::isOpen() const {
    return file_.is_open();
}

void CsvGazeWriter::writeFrame(long frameNumber, const std::vector<PanoViewer::gaze>& gazes) {
    if (!file_.is_open()) return;
    // Delegate actual analysis formatting to existing PanoViewer helper.
    // This keeps CSV format consistent with current behavior.
    PanoViewer tmp;
    tmp.saveGazeAnalysis(file_, frameNumber, gazes);
}

void CsvGazeWriter::flush() {
    if (file_.is_open()) file_.flush();
}

} // namespace io

