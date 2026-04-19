#include "io/proto_mapper.h"

#include <cmath>

namespace {

float finiteOrZero(float value) {
    return std::isfinite(value) ? value : 0.0f;
}

} // namespace

namespace io {

rapport::PipelineUpdate mapToPipelineUpdate(
    const domain::VisionFrameResult& result,
    const PipelineFrameContext& context,
    std::uint64_t sequence_id,
    std::uint64_t dropped_updates_total) {
    (void)result;

    rapport::PipelineUpdate update;
    update.set_schema_version(1);
    update.set_sequence_id(sequence_id);
    update.set_frame_index(context.frame_index);
    update.set_source_timestamp_ns(context.source_timestamp_ns);
    update.set_processed_timestamp_ns(context.processed_timestamp_ns);
    update.set_dropped_updates_total(dropped_updates_total);

    for (const auto& edge : context.interactions) {
        rapport::InteractionEdge* outEdge = update.add_interactions();
        outEdge->set_from_person_id(edge.from_person_id);
        outEdge->set_to_person_id(edge.to_person_id);
        outEdge->set_angle_deg(finiteOrZero(edge.angle_deg));
        outEdge->set_is_looking(edge.is_looking);
    }

    return update;
}

} // namespace io
