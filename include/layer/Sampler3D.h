#pragma once

#include <tuple>
#include <random>
#include <string>
#include <vector>
#include <algorithm>

#include "ClassParameter.h"
#include "Tensor.h"

namespace layer {

    // Simple structure to hold our bounding box coordinates
    struct BBox {
        size_t x_min;
        size_t x_max;
        size_t y_min;
        size_t y_max;

        std::vector<std::pair<float, float>> landmarks;
    };

    // The abstract base class (Interface)
    class ISampler3D : public ClassParameter {
    public:

        template<class T, class Factory>
        ISampler3D(const RegisterClassParameter<T, Factory>& registration)
                : ClassParameter(registration) {}

        virtual ~ISampler3D() = default;

        // Process data set: Read landmarks or run detector to build BBoxes
        virtual void add_sequence_bboxes(const std::vector<BBox>& bboxes) = 0;

        // Sample patch: Returns {x, y, t} based on the video index and filter dimensions
        virtual std::tuple<size_t, size_t, size_t> sample_patch(
                size_t video_index,
                size_t input_width, size_t input_height, size_t input_time,
                size_t filter_w, size_t filter_h, size_t filter_t,
                std::default_random_engine& rng) = 0;
    };

        // Input-aware sampler interface (can inspect the input tensor)
        class IInputAwareSampler3D : public ISampler3D {
        public:

        template<class T, class Factory>
        IInputAwareSampler3D(const RegisterClassParameter<T, Factory>& registration)
            : ISampler3D(registration) {}

        virtual ~IInputAwareSampler3D() = default;

        virtual std::tuple<size_t, size_t, size_t> sample_patch_with_input(
            size_t video_index,
            const Tensor<float>& sample,
            size_t input_width, size_t input_height, size_t input_time,
            size_t filter_w, size_t filter_h, size_t filter_t,
            std::default_random_engine& rng) = 0;
        };

    // Random baseline sampler
    class Sampler3DRandom : public ISampler3D {
    public:
        Sampler3DRandom();
        void add_sequence_bboxes(const std::vector<BBox>& bboxes) override;

        std::tuple<size_t, size_t, size_t> sample_patch(
                size_t video_index,
                size_t input_width, size_t input_height, size_t input_time,
                size_t filter_w, size_t filter_h, size_t filter_t,
                std::default_random_engine& rng) override;
    };

    // Per-Frame spatial-temporal sampler
    class Sampler3DFacialPerFrame : public ISampler3D {
    public:
        Sampler3DFacialPerFrame();
    private:
        // Outer vector: videos. Inner vector: frames.
        std::vector<std::vector<BBox>> _video_frame_bboxes;

    public:
        void add_sequence_bboxes(const std::vector<BBox>& bboxes) override;

        std::tuple<size_t, size_t, size_t> sample_patch(
                size_t video_index,
                size_t input_width, size_t input_height, size_t input_time,
                size_t filter_w, size_t filter_h, size_t filter_t,
                std::default_random_engine& rng) override;
    };

    // Per-Video (static) sampler
    class Sampler3DFacialPerVideo : public ISampler3D {
    private:
        // One BBox per video
        std::vector<BBox> _video_bboxes;

    public:
        Sampler3DFacialPerVideo();
        void add_sequence_bboxes(const std::vector<BBox>& bboxes) override;

        std::tuple<size_t, size_t, size_t> sample_patch(
                size_t video_index,
                size_t input_width, size_t input_height, size_t input_time,
                size_t filter_w, size_t filter_h, size_t filter_t,
                std::default_random_engine& rng) override;
    };


    class SamplerFactory : public ClassParameterFactory<ISampler3D, SamplerFactory> {

    public:
        SamplerFactory() : ClassParameterFactory<ISampler3D, SamplerFactory>("ISampler3D") {

        }

    };

    // Sampler centered on Landmarks
    class Sampler3DLandmark : public ISampler3D {
    private:
        std::vector<std::vector<BBox>> _video_frame_bboxes;

    public:
        Sampler3DLandmark();
        void add_sequence_bboxes(const std::vector<BBox>& bboxes) override;

        std::tuple<size_t, size_t, size_t> sample_patch(
                size_t video_index,
                size_t input_width, size_t input_height, size_t input_time,
                size_t filter_w, size_t filter_h, size_t filter_t,
                std::default_random_engine& rng) override;
    };

    // Input-aware sampler based on ON/OFF saliency
    class Sampler3DOnOffSaliency : public IInputAwareSampler3D {
    private:
        std::vector<std::vector<BBox>> _video_frame_bboxes;
        float _random_mix;
        bool _latency_coding;
        bool _use_bboxes;

        float _pixel_saliency(const Tensor<float>& sample, size_t x, size_t y, size_t k, size_t depth) const;
        void _get_roi(size_t video_index, size_t frame_index,
                      size_t input_width, size_t input_height,
                      size_t& x_min, size_t& x_max,
                      size_t& y_min, size_t& y_max) const;

    public:
        Sampler3DOnOffSaliency();
        void add_sequence_bboxes(const std::vector<BBox>& bboxes) override;

        std::tuple<size_t, size_t, size_t> sample_patch(
                size_t video_index,
                size_t input_width, size_t input_height, size_t input_time,
                size_t filter_w, size_t filter_h, size_t filter_t,
                std::default_random_engine& rng) override;

        std::tuple<size_t, size_t, size_t> sample_patch_with_input(
                size_t video_index,
                const Tensor<float>& sample,
                size_t input_width, size_t input_height, size_t input_time,
                size_t filter_w, size_t filter_h, size_t filter_t,
                std::default_random_engine& rng) override;
    };

}