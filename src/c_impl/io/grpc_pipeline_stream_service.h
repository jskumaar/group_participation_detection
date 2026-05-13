#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>

#include <grpcpp/grpcpp.h>
#include <google/protobuf/empty.pb.h>

#include "domain/types.h"
#include "rapport.grpc.pb.h"

namespace io {

class RapportStreamImpl final : public rapport::RapportStream::Service {
public:
    RapportStreamImpl(std::atomic<bool>& running, std::size_t max_queued);

    void publish(const domain::VisionFrameResult& result, const domain::PipelineFrameContext& context);
    void notifyStop();

    grpc::Status StreamPipelineUpdates(
        grpc::ServerContext* context,
        const google::protobuf::Empty* request,
        grpc::ServerWriter<rapport::PipelineUpdate>* writer) override;

private:
    std::atomic<bool>& running_;
    std::size_t max_queued_ = 256;

    std::mutex mutex_;
    std::condition_variable cond_;
    std::deque<rapport::PipelineUpdate> queue_;
};

} // namespace io

