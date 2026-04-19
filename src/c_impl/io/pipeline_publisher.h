#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "domain/types.h"

namespace io {

struct InteractionPair {
    int from_person_id = -1;
    int to_person_id = -1;
    float angle_deg = 0.0f;
    bool is_looking = false;
};

struct PipelineFrameContext {
    std::uint64_t frame_index = 0;
    std::uint64_t source_timestamp_ns = 0;
    std::uint64_t processed_timestamp_ns = 0;
    std::vector<InteractionPair> interactions;
};

class IPipelinePublisher {
public:
    virtual ~IPipelinePublisher();

    virtual bool start() = 0;
    virtual void stop() = 0;
    virtual void publish(const domain::VisionFrameResult& result, const PipelineFrameContext& context) = 0;
};

} // namespace io
