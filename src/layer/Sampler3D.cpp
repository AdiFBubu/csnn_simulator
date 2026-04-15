#include "layer/Sampler3D.h"
#include <iostream>

namespace layer {

    // ==========================================
    // Sampler3DRandom Implementation
    // ==========================================

    static RegisterClassParameter<Sampler3DRandom, SamplerFactory> _facial_random_sampler_register("Sampler3DRandom");

    Sampler3DRandom::Sampler3DRandom()
            : ISampler3D(_facial_random_sampler_register)
    {
    }

    void Sampler3DRandom::add_sequence_bboxes(const std::vector<BBox>& bboxes) {
        // No pre-processing needed for random sampling
    }

    std::tuple<size_t, size_t, size_t> Sampler3DRandom::sample_patch(
            size_t video_index, size_t input_width, size_t input_height, size_t input_time,
            size_t filter_w, size_t filter_h, size_t filter_t, std::default_random_engine& rng)
    {
        std::uniform_int_distribution<size_t> rand_x(0, input_width - filter_w);
        std::uniform_int_distribution<size_t> rand_y(0, input_height - filter_h);
        std::uniform_int_distribution<size_t> rand_t(0, input_time - filter_t);

        return {rand_x(rng), rand_y(rng), rand_t(rng)};
    }


    // ==========================================
    // Sampler3DFacialPerFrame Implementation
    // ==========================================

    static RegisterClassParameter<Sampler3DFacialPerFrame, SamplerFactory> _facial_sampler_register("Sampler3DFacialPerFrame");

    Sampler3DFacialPerFrame::Sampler3DFacialPerFrame()
            : ISampler3D(_facial_sampler_register)
    {
    }

    void Sampler3DFacialPerFrame::add_sequence_bboxes(const std::vector<BBox>& bboxes) {
        _video_frame_bboxes.push_back(bboxes);
    }

    std::tuple<size_t, size_t, size_t> Sampler3DFacialPerFrame::sample_patch(
            size_t video_index, size_t input_width, size_t input_height, size_t input_time,
            size_t filter_w, size_t filter_h, size_t filter_t, std::default_random_engine& rng)
    {
        // 1. Pick a random starting frame (t)
        std::uniform_int_distribution<size_t> rand_t(0, input_time - filter_t);
        size_t t = rand_t(rng);

        // Safety check to ensure we have data for this video
        if (video_index >= _video_frame_bboxes.size() || _video_frame_bboxes[video_index].empty()) {
            return {0, 0, t}; // Fallback if data is missing
        }

        // 2. Calculate the Union Bounding Box across the temporal window
        size_t u_xmin = input_width, u_xmax = 0, u_ymin = input_height, u_ymax = 0;

        for (size_t k = 0; k < filter_t; ++k) {
            // Protect against out-of-bounds frame access
            size_t frame_idx = std::min(t + k, _video_frame_bboxes[video_index].size() - 1);
            const BBox& box = _video_frame_bboxes[video_index][frame_idx];

            u_xmin = std::min(u_xmin, box.x_min);
            u_xmax = std::max(u_xmax, box.x_max);
            u_ymin = std::min(u_ymin, box.y_min);
            u_ymax = std::max(u_ymax, box.y_max);
        }

        // 3. Ensure valid sampling area. If bbox is too small, clamp to 0.
        size_t max_x_start = (u_xmax > u_xmin + filter_w) ? (u_xmax - filter_w) : u_xmin;
        size_t max_y_start = (u_ymax > u_ymin + filter_h) ? (u_ymax - filter_h) : u_ymin;

        // 4. Pick random x, y within the union box
        std::uniform_int_distribution<size_t> rand_x(u_xmin, max_x_start);
        std::uniform_int_distribution<size_t> rand_y(u_ymin, max_y_start);

        return {rand_x(rng), rand_y(rng), t};
    }


    // ==========================================
    // Sampler3DFacialPerVideo Implementation
    // ==========================================

    static RegisterClassParameter<Sampler3DFacialPerVideo, SamplerFactory> _facial_video_sampler_register("Sampler3DFacialPerVideo");

    Sampler3DFacialPerVideo::Sampler3DFacialPerVideo()
            : ISampler3D(_facial_video_sampler_register)
    {
    }

    void Sampler3DFacialPerVideo::add_sequence_bboxes(const std::vector<BBox>& bboxes) {
        if (bboxes.empty()) return;

        size_t u_xmin = bboxes[0].x_min;
        size_t u_xmax = bboxes[0].x_max;
        size_t u_ymin = bboxes[0].y_min;
        size_t u_ymax = bboxes[0].y_max;

        for (const auto& box : bboxes) {
            u_xmin = std::min(u_xmin, box.x_min);
            u_xmax = std::max(u_xmax, box.x_max);
            u_ymin = std::min(u_ymin, box.y_min);
            u_ymax = std::max(u_ymax, box.y_max);
        }

        _video_bboxes.push_back({u_xmin, u_xmax, u_ymin, u_ymax});
    }

    std::tuple<size_t, size_t, size_t> Sampler3DFacialPerVideo::sample_patch(
            size_t video_index, size_t input_width, size_t input_height, size_t input_time,
            size_t filter_w, size_t filter_h, size_t filter_t, std::default_random_engine& rng)
    {
        std::uniform_int_distribution<size_t> rand_t(0, input_time - filter_t);
        size_t t = rand_t(rng);

        if (video_index >= _video_bboxes.size()) { return {0, 0, t}; }

        const BBox& box = _video_bboxes[video_index];

        size_t max_x_start = (box.x_max > box.x_min + filter_w) ? (box.x_max - filter_w) : box.x_min;
        size_t max_y_start = (box.y_max > box.y_min + filter_h) ? (box.y_max - filter_h) : box.y_min;

        std::uniform_int_distribution<size_t> rand_x(box.x_min, max_x_start);
        std::uniform_int_distribution<size_t> rand_y(box.y_min, max_y_start);

        return {rand_x(rng), rand_y(rng), t};
    }


    static RegisterClassParameter<Sampler3DLandmark, SamplerFactory> _reg_sampler_landmark("Sampler3DLandmark");

    Sampler3DLandmark::Sampler3DLandmark() : ISampler3D(_reg_sampler_landmark) {}

    void Sampler3DLandmark::add_sequence_bboxes(const std::vector<BBox>& bboxes) {
        _video_frame_bboxes.push_back(bboxes);
    }

    std::tuple<size_t, size_t, size_t> Sampler3DLandmark::sample_patch(
            size_t video_index,
            size_t input_width, size_t input_height, size_t input_time,
            size_t filter_w, size_t filter_h, size_t filter_t,
            std::default_random_engine& rng) {

        if (video_index >= _video_frame_bboxes.size() || _video_frame_bboxes[video_index].empty()) {
            return {0, 0, 0};
        }

        const auto& frames = _video_frame_bboxes[video_index];

        // random temporal frame
        std::uniform_int_distribution<size_t> rand_t(0, input_time > filter_t ? input_time - filter_t : 0);
        size_t t = rand_t(rng);

        const auto& frame_data = frames[std::min(t, frames.size() - 1)];

        int x = 0, y = 0;

        // if the landmarks are available
        if (!frame_data.landmarks.empty()) {

            // points 36-67 represent the eyes, nose, and mouth in the standard 68-point format
            size_t max_idx = std::min(size_t(67), frame_data.landmarks.size() - 1);
            size_t min_idx = std::min(size_t(17), max_idx);

            std::uniform_int_distribution<size_t> rand_pt(min_idx, max_idx);
            size_t pt_idx = rand_pt(rng);

            float target_x = frame_data.landmarks[pt_idx].first;
            float target_y = frame_data.landmarks[pt_idx].second;

            // we add jitter to avoid always sampling the exact same region around the landmark, which can help with robustness and generalization
            std::uniform_int_distribution<int> jitter(-2, 2);
            x = static_cast<int>(target_x - (filter_w / 2.0f)) + jitter(rng);
            y = static_cast<int>(target_y - (filter_h / 2.0f)) + jitter(rng);

        } else {
            // Fallback if the landmarks file misses (we use the bounding box)
            int max_x_start = static_cast<int>(frame_data.x_max) - static_cast<int>(filter_w);
            int max_y_start = static_cast<int>(frame_data.y_max) - static_cast<int>(filter_h);

            std::uniform_int_distribution<int> rand_x(frame_data.x_min, std::max((int)frame_data.x_min, max_x_start));
            std::uniform_int_distribution<int> rand_y(frame_data.y_min, std::max((int)frame_data.y_min, max_y_start));

            x = rand_x(rng);
            y = rand_y(rng);
        }

        // claming - we ensure that the patch does not go out of image bounds
        x = std::clamp(x, 0, static_cast<int>(input_width > filter_w ? input_width - filter_w : 0));
        y = std::clamp(y, 0, static_cast<int>(input_height > filter_h ? input_height - filter_h : 0));

        return {static_cast<size_t>(x), static_cast<size_t>(y), t};
    }

} // namespace layer