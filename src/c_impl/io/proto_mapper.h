#pragma once

#include <cstdint>

#include "io/pipeline_publisher.h"
#include "rapport.pb.h"

namespace io {

rapport::PipelineUpdate mapToPipelineUpdate(
    const domain::VisionFrameResult& result,
    const PipelineFrameContext& context,
    std::uint64_t sequence_id,
    std::uint64_t dropped_updates_total);

} // namespace io
