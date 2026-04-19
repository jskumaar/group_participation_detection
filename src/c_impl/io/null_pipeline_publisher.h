#pragma once

#include "io/pipeline_publisher.h"

namespace io {

class NullPipelinePublisher final : public IPipelinePublisher {
public:
    bool start() override { return true; }
    void stop() override {}
    void publish(const domain::VisionFrameResult&, const PipelineFrameContext&) override {}
};

} // namespace io
