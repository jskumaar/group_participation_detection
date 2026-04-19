#pragma once

#include <memory>

#include "io/pipeline_publisher.h"

namespace io {

std::unique_ptr<IPipelinePublisher> createPipelinePublisherFromEnv();

} // namespace io
