#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <grpcpp/grpcpp.h>

#include "io/pipeline_publisher.h"
#include "rapport.grpc.pb.h"

namespace io {

class GrpcPipelinePublisher final : public IPipelinePublisher {
public:
    explicit GrpcPipelinePublisher(std::string endpoint, std::size_t max_queued);
    ~GrpcPipelinePublisher() override;

    bool start() override;
    void stop() override;
    void publish(const domain::VisionFrameResult& result, const PipelineFrameContext& context) override;

private:
    struct PendingFrame {
        domain::VisionFrameResult result;
        PipelineFrameContext context;
    };

    void workerLoop();
    bool ensureStream();
    void resetStream();

    std::string endpoint_;
    std::size_t max_queued_ = 256;

    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<rapport::RapportStream::Stub> stub_;
    std::unique_ptr<grpc::ClientContext> client_context_;
    rapport::StreamAck stream_ack_;
    std::unique_ptr<grpc::ClientWriter<rapport::PipelineUpdate>> writer_;

    std::atomic<bool> running_{false};
    std::thread worker_;

    std::mutex mutex_;
    std::condition_variable cond_;
    std::deque<PendingFrame> queue_;
    std::uint64_t sequence_id_ = 0;
    std::atomic<std::uint64_t> dropped_updates_total_{0};
};

} // namespace io
