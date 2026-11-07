#include "csv_gaze_writer.h"

#include <unordered_map>

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

void CsvGazeWriter::writeFrame(
    long frameNumber,
    const std::vector<PanoViewer::gaze>& gazes,
    const std::vector<domain::InteractionPair>& interactions) {
    if (!file_.is_open()) return;

    std::unordered_map<int, std::string> lookingAtByPerson;
    lookingAtByPerson.reserve(gazes.size());
    for (const auto& interaction : interactions) {
        if (!interaction.is_looking) continue;
        //converts interactions into a dictionary
        auto& ids = lookingAtByPerson[interaction.from_person_id];
        if (!ids.empty()) ids += ";";
        ids += std::to_string(interaction.to_person_id);
    }

    for (const auto& g : gazes) {
        const auto it = lookingAtByPerson.find(g.personID);
        const std::string& looking_at_ids = (it != lookingAtByPerson.end()) ? it->second : std::string{};

        file_ << frameNumber << ","
              << g.personID << ","
              << (g.box.x + g.box.width / 2.0f) << "," << (g.box.y + g.box.height / 2.0f) << ","
              << g.start.x << "," << g.start.y << "," << g.start.z << ","
              << g.direction[0] << "," << g.direction[1] << "," << g.direction[2] << ","
              << "\"" << looking_at_ids << "\"\n";
    }
}

void CsvGazeWriter::flush() {
    if (file_.is_open()) file_.flush();
}

} // namespace io

