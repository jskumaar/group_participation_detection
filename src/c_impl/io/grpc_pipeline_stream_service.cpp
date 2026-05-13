#include "io/grpc_pipeline_stream_service.h"

#include <cmath>
#include <chrono>
#include <utility>

namespace {

float finiteOrZero(float value) {
    return std::isfinite(value) ? value : 0.0f;
}

rapport::PipelineUpdate mapToPipelineUpdate(
    const domain::VisionFrameResult& result,
    const domain::PipelineFrameContext& context) {
    rapport::PipelineUpdate update;
    update.set_frame_index(context.frame_index);
    update.set_playback_timestamp_ns(context.playback_timestamp_ns);

    for (const auto& gaze : result.gazes) {
        rapport::GazeSample* outGaze = update.add_gazes();
        outGaze->set_person_id(gaze.personID);

        rapport::Vector3* origin = outGaze->mutable_origin();
        origin->set_x(finiteOrZero(gaze.start.x));
        origin->set_y(finiteOrZero(gaze.start.y));
        origin->set_z(finiteOrZero(gaze.start.z));

        rapport::Vector3* direction = outGaze->mutable_direction();
        direction->set_x(finiteOrZero(gaze.direction[0]));
        direction->set_y(finiteOrZero(gaze.direction[1]));
        direction->set_z(finiteOrZero(gaze.direction[2]));
    }

    for (const auto& edge : context.interactions) {
        rapport::InteractionEdge* outEdge = update.add_interactions();
        outEdge->set_from_person_id(edge.from_person_id);
        outEdge->set_to_person_id(edge.to_person_id);
        outEdge->set_angle_deg(finiteOrZero(edge.angle_deg));
        outEdge->set_is_looking(edge.is_looking);
    }

    return update;
}

} // namespace

namespace io {

RapportStreamImpl::RapportStreamImpl(std::atomic<bool>& running, std::size_t max_queued)
    : running_(running),
      max_queued_(max_queued > 0 ? max_queued : 256) {
}

void RapportStreamImpl::publish(
    const domain::VisionFrameResult& result,
    const domain::PipelineFrameContext& context) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_.load()) {
        return;
    }
    if (queue_.size() >= max_queued_) {
        queue_.pop_front();
    }
    queue_.push_back(mapToPipelineUpdate(result, context));
    cond_.notify_one();
}

void RapportStreamImpl::notifyStop() {
    cond_.notify_all();
}

grpc::Status RapportStreamImpl::StreamPipelineUpdates(
    grpc::ServerContext* context,
    const google::protobuf::Empty*,
    grpc::ServerWriter<rapport::PipelineUpdate>* writer) {
    while (running_.load() && !context->IsCancelled()) {
        rapport::PipelineUpdate update;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cond_.wait_for(lock, std::chrono::milliseconds(200), [this]() {
                return !running_.load() || !queue_.empty();
            });
            if (!running_.load()) {
                break;
            }
            if (queue_.empty()) {
                continue;
            }
            update = std::move(queue_.front());
            queue_.pop_front();
        }

        if (!writer->Write(update)) {
            break;
        }
    }
    return grpc::Status::OK;
}

} // namespace io

