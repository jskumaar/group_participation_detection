#include "io/pipeline_publisher_factory.h"

#include <cstdlib>
#include <memory>
#include <string>

#include "io/null_pipeline_publisher.h"

#ifdef ENABLE_RAPPORT_GRPC
#include "io/grpc_pipeline_publisher.h"
#endif

namespace io {

std::unique_ptr<IPipelinePublisher> createPipelinePublisherFromEnv() {
#ifdef ENABLE_RAPPORT_GRPC
    const char* enabled = std::getenv("RAPPORT_GRPC_ENABLED");
    if (enabled != nullptr && std::string(enabled) == "0") {
        return std::make_unique<NullPipelinePublisher>();
    }

    const char* target = std::getenv("RAPPORT_GRPC_TARGET");
    const std::string endpoint = (target != nullptr && *target != '\0') ? std::string(target) : std::string("127.0.0.1:50051");

    const char* maxQueuedEnv = std::getenv("RAPPORT_GRPC_MAX_QUEUED");
    std::size_t maxQueued = 256;
    if (maxQueuedEnv != nullptr) {
        try {
            maxQueued = static_cast<std::size_t>(std::stoul(maxQueuedEnv));
        } catch (...) {
            maxQueued = 256;
        }
    }

    return std::make_unique<GrpcPipelinePublisher>(endpoint, maxQueued);
#else
    return std::make_unique<NullPipelinePublisher>();
#endif
}

} // namespace io
