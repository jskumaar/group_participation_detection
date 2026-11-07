#include "io/grpc_pipeline_publisher.h"

#include <utility>

namespace io {

GrpcPipelinePublisher::GrpcPipelinePublisher(std::string endpoint, std::size_t max_queued)
    : endpoint_(std::move(endpoint)),
      max_queued_(max_queued > 0 ? max_queued : 256),
      service_(running_, max_queued_) {
}

GrpcPipelinePublisher::~GrpcPipelinePublisher() {
    stop();
}

bool GrpcPipelinePublisher::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return true;
    }

    grpc::ServerBuilder builder;
    // selected_port stays 0 if the endpoint could not be bound, even when a server is returned.
    int selected_port = 0;
    builder.AddListeningPort(endpoint_, grpc::InsecureServerCredentials(), &selected_port);
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
    if (!server_ || selected_port == 0) {
        if (server_) {
            server_->Shutdown();
            server_.reset();
        }
        running_.store(false);
        return false;
    }
    return true;
}

void GrpcPipelinePublisher::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) {
        return;
    }

    service_.notifyStop();
    if (server_) {
        server_->Shutdown();
        server_.reset();
    }
}

void GrpcPipelinePublisher::publish(const domain::VisionFrameResult& result, const domain::PipelineFrameContext& context) {
    service_.publish(result, context);
}

} // namespace io
