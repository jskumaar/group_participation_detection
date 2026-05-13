#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

#include "domain/types.h"
#include "io/grpc_pipeline_stream_service.h"

namespace io {

class GrpcPipelinePublisher final {
public:
    explicit GrpcPipelinePublisher(std::string endpoint, std::size_t max_queued);
    ~GrpcPipelinePublisher();

    bool start();
    void stop();
    void publish(const domain::VisionFrameResult& result, const domain::PipelineFrameContext& context);

private:
    std::string endpoint_;
    std::size_t max_queued_ = 256;
    RapportStreamImpl service_;
    std::unique_ptr<grpc::Server> server_;

    std::atomic<bool> running_{false};
};

} // namespace io
