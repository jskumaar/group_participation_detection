#include "io/grpc_pipeline_publisher.h"

#include <chrono>
#include <utility>

#include "io/proto_mapper.h"

namespace io {

GrpcPipelinePublisher::GrpcPipelinePublisher(std::string endpoint, std::size_t max_queued)
    : endpoint_(std::move(endpoint)),
      max_queued_(max_queued > 0 ? max_queued : 256) {
}

GrpcPipelinePublisher::~GrpcPipelinePublisher() {
    stop();
}

bool GrpcPipelinePublisher::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return true;
    }

    channel_ = grpc::CreateChannel(endpoint_, grpc::InsecureChannelCredentials());
    stub_ = rapport::RapportStream::NewStub(channel_);
    worker_ = std::thread(&GrpcPipelinePublisher::workerLoop, this);
    return true;
}

void GrpcPipelinePublisher::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) {
        return;
    }

    cond_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
    resetStream();
}

void GrpcPipelinePublisher::publish(const domain::VisionFrameResult& result, const PipelineFrameContext& context) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.size() >= max_queued_) {
        queue_.pop_front();
        dropped_updates_total_.fetch_add(1, std::memory_order_relaxed);
    }
    queue_.push_back(PendingFrame{result, context});
    cond_.notify_one();
}

bool GrpcPipelinePublisher::ensureStream() {
    if (writer_) {
        return true;
    }

    if (!stub_) {
        return false;
    }

    client_context_ = std::make_unique<grpc::ClientContext>();
    writer_ = stub_->StreamPipelineUpdates(client_context_.get(), &stream_ack_);
    return static_cast<bool>(writer_);
}

void GrpcPipelinePublisher::resetStream() {
    if (writer_) {
        writer_.reset();
    }
    if (client_context_) {
        client_context_->TryCancel();
        client_context_.reset();
    }
}

void GrpcPipelinePublisher::workerLoop() {
    while (running_.load()) {
        PendingFrame pending;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cond_.wait(lock, [this]() { return !running_.load() || !queue_.empty(); });
            if (!running_.load()) {
                break;
            }
            pending = std::move(queue_.front());
            queue_.pop_front();
        }

        if (!ensureStream()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }

        const std::uint64_t seq = ++sequence_id_;
        rapport::PipelineUpdate update = mapToPipelineUpdate(
            pending.result,
            pending.context,
            seq,
            dropped_updates_total_.load(std::memory_order_relaxed));

        if (!writer_->Write(update)) {
            writer_->Finish();
            resetStream();
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
    }

    if (writer_) {
        writer_->WritesDone();
        writer_->Finish();
    }
}

} // namespace io
