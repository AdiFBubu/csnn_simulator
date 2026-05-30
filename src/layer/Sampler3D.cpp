#include "layer/Sampler3D.h"
#include "Spike.h"
#include <cmath>
#include <iostream>
#include <limits>

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

    // ==========================================
    // Sampler3DOnOffSaliency Implementation
    // ==========================================

    static RegisterClassParameter<Sampler3DOnOffSaliency, SamplerFactory> _saliency_sampler_register("Sampler3DOnOffSaliency");

    Sampler3DOnOffSaliency::Sampler3DOnOffSaliency()
            : IInputAwareSampler3D(_saliency_sampler_register),
              _random_mix(0.2f),
              _latency_coding(true),
              _use_bboxes(true) {
        add_parameter("random_mix", _random_mix, 0.2f);
        add_parameter("latency_coding", _latency_coding, true);
        add_parameter("use_bboxes", _use_bboxes, true);
    }

    void Sampler3DOnOffSaliency::add_sequence_bboxes(const std::vector<BBox>& bboxes) {
        _video_frame_bboxes.push_back(bboxes);
    }

    static size_t clamp_start(int value, size_t min_value, size_t max_value) {
        if (max_value < min_value) {
            return min_value;
        }
        if (value < static_cast<int>(min_value)) {
            return min_value;
        }
        if (value > static_cast<int>(max_value)) {
            return max_value;
        }
        return static_cast<size_t>(value);
    }

    void Sampler3DOnOffSaliency::_get_roi(size_t video_index, size_t frame_index,
                                          size_t input_width, size_t input_height,
                                          size_t& x_min, size_t& x_max,
                                          size_t& y_min, size_t& y_max) const {
        if (input_width == 0 || input_height == 0) {
            x_min = x_max = y_min = y_max = 0;
            return;
        }

        x_min = 0;
        y_min = 0;
        x_max = input_width - 1;
        y_max = input_height - 1;

        if (!_use_bboxes) {
            return;
        }

        if (video_index >= _video_frame_bboxes.size()) {
            return;
        }

        const auto& frames = _video_frame_bboxes[video_index];
        if (frames.empty()) {
            return;
        }

        const auto& box = frames[std::min(frame_index, frames.size() - 1)];
        x_min = std::min(box.x_min, x_max);
        x_max = std::min(box.x_max, x_max);
        y_min = std::min(box.y_min, y_max);
        y_max = std::min(box.y_max, y_max);

        if (x_min > x_max) {
            std::swap(x_min, x_max);
        }
        if (y_min > y_max) {
            std::swap(y_min, y_max);
        }
    }

    float Sampler3DOnOffSaliency::_pixel_saliency(const Tensor<float>& sample, size_t x, size_t y, size_t k, size_t depth) const {
        float best = 0.0f;
        const bool has_time = sample.shape().number() > 3;

        for (size_t c = 0; c < depth; ++c) {
            float v = has_time ? sample.at(x, y, c, k) : sample.at(x, y, c);

            if (!std::isfinite(v)) {
                continue;
            }

            float s = 0.0f;
            if (_latency_coding) {
                if (v >= INFINITE_TIME) {
                    continue;
                }
                s = std::max(0.0f, 1.0f - v);
            } else {
                s = std::max(0.0f, v);
            }

            if (s > best) {
                best = s;
            }
        }

        return best;
    }

    std::tuple<size_t, size_t, size_t> Sampler3DOnOffSaliency::sample_patch(
            size_t video_index, size_t input_width, size_t input_height, size_t input_time,
            size_t filter_w, size_t filter_h, size_t filter_t, std::default_random_engine& rng) {

        size_t t = 0;
        if (input_time > filter_t) {
            std::uniform_int_distribution<size_t> rand_t(0, input_time - filter_t);
            t = rand_t(rng);
        }

        size_t x_min = 0, x_max = input_width > 0 ? input_width - 1 : 0;
        size_t y_min = 0, y_max = input_height > 0 ? input_height - 1 : 0;
        _get_roi(video_index, t, input_width, input_height, x_min, x_max, y_min, y_max);

        size_t max_x_start = (input_width > filter_w) ? (input_width - filter_w) : 0;
        size_t max_y_start = (input_height > filter_h) ? (input_height - filter_h) : 0;

        size_t x_start_min = std::min(x_min, max_x_start);
        size_t y_start_min = std::min(y_min, max_y_start);

        size_t x_start_max = std::min(max_x_start, (x_max > x_min + filter_w) ? (x_max - filter_w) : x_min);
        size_t y_start_max = std::min(max_y_start, (y_max > y_min + filter_h) ? (y_max - filter_h) : y_min);

        if (x_start_min > x_start_max) {
            x_start_min = x_start_max;
        }
        if (y_start_min > y_start_max) {
            y_start_min = y_start_max;
        }

        std::uniform_int_distribution<size_t> rand_x(x_start_min, x_start_max);
        std::uniform_int_distribution<size_t> rand_y(y_start_min, y_start_max);

        return {rand_x(rng), rand_y(rng), t};
    }

    std::tuple<size_t, size_t, size_t> Sampler3DOnOffSaliency::sample_patch_with_input(
            size_t video_index,
            const Tensor<float>& sample,
            size_t input_width, size_t input_height, size_t input_time,
            size_t filter_w, size_t filter_h, size_t filter_t,
            std::default_random_engine& rng) {

        if (input_width == 0 || input_height == 0 || input_time == 0) {
            return {0, 0, 0};
        }

        // exploration
        std::uniform_real_distribution<float> rand01(0.0f, 1.0f);
        if (_random_mix > 0.0f && rand01(rng) < _random_mix) {
            return sample_patch(video_index, input_width, input_height, input_time, filter_w, filter_h, filter_t, rng);
        }

        size_t depth = sample.shape().dim(2);
        size_t time_depth = sample.shape().number() > 3 ? sample.shape().dim(3) : 1;
        size_t time_limit = std::min(input_time, time_depth);

        std::vector<float> frame_scores(time_limit, 0.0f);
        float total_score = 0.0f;

        for (size_t k = 0; k < time_limit; ++k) {
            size_t x_min = 0, x_max = input_width - 1;
            size_t y_min = 0, y_max = input_height - 1;
            _get_roi(video_index, k, input_width, input_height, x_min, x_max, y_min, y_max);

            float sum = 0.0f;
            for (size_t y = y_min; y <= y_max; ++y) {
                for (size_t x = x_min; x <= x_max; ++x) {
                    sum += _pixel_saliency(sample, x, y, k, depth);
                }
            }

            frame_scores[k] = sum;
            total_score += sum;
        }

        size_t t_center = 0;
        if (total_score > 0.0f) {
            std::uniform_real_distribution<float> rand_total(0.0f, total_score);
            float r = rand_total(rng);
            float acc = 0.0f;
            for (size_t k = 0; k < time_limit; ++k) {
                acc += frame_scores[k];
                if (r <= acc) {
                    t_center = k;
                    break;
                }
            }
        } else {
            std::uniform_int_distribution<size_t> rand_t(0, input_time > filter_t ? input_time - filter_t : 0);
            t_center = rand_t(rng);
        }

        size_t x_min = 0, x_max = input_width - 1;
        size_t y_min = 0, y_max = input_height - 1;
        _get_roi(video_index, t_center, input_width, input_height, x_min, x_max, y_min, y_max);

        float spatial_total = 0.0f;
        for (size_t y = y_min; y <= y_max; ++y) {
            for (size_t x = x_min; x <= x_max; ++x) {
                spatial_total += _pixel_saliency(sample, x, y, t_center, depth);
            }
        }

        size_t x_center = x_min;
        size_t y_center = y_min;

        if (spatial_total > 0.0f) {
            std::uniform_real_distribution<float> rand_spatial(0.0f, spatial_total);
            float r = rand_spatial(rng);
            float acc = 0.0f;
            for (size_t y = y_min; y <= y_max; ++y) {
                for (size_t x = x_min; x <= x_max; ++x) {
                    acc += _pixel_saliency(sample, x, y, t_center, depth);
                    if (r <= acc) {
                        x_center = x;
                        y_center = y;
                        y = y_max;
                        break;
                    }
                }
            }
        } else {
            std::uniform_int_distribution<size_t> rand_x(x_min, x_max);
            std::uniform_int_distribution<size_t> rand_y(y_min, y_max);
            x_center = rand_x(rng);
            y_center = rand_y(rng);
        }

        size_t max_x_start = (input_width > filter_w) ? (input_width - filter_w) : 0;
        size_t max_y_start = (input_height > filter_h) ? (input_height - filter_h) : 0;

        size_t x_start_min = std::min(x_min, max_x_start);
        size_t y_start_min = std::min(y_min, max_y_start);

        size_t x_start_max = std::min(max_x_start, (x_max > x_min + filter_w) ? (x_max - filter_w) : x_min);
        size_t y_start_max = std::min(max_y_start, (y_max > y_min + filter_h) ? (y_max - filter_h) : y_min);

        if (x_start_min > x_start_max) {
            x_start_min = x_start_max;
        }
        if (y_start_min > y_start_max) {
            y_start_min = y_start_max;
        }

        int x_start_i = static_cast<int>(x_center) - static_cast<int>(filter_w / 2);
        int y_start_i = static_cast<int>(y_center) - static_cast<int>(filter_h / 2);

        size_t x_start = clamp_start(x_start_i, x_start_min, x_start_max);
        size_t y_start = clamp_start(y_start_i, y_start_min, y_start_max);

        size_t max_t_start = (input_time > filter_t) ? (input_time - filter_t) : 0;
        int t_start_i = static_cast<int>(t_center) - static_cast<int>(filter_t / 2);
        size_t t_start = clamp_start(t_start_i, 0, max_t_start);

        return {x_start, y_start, t_start};
    }

} // namespace layer