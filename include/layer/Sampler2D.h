#pragma once

#include <tuple>
#include <random>
#include <string>
#include <vector>
#include <algorithm>

#include "ClassParameter.h"
#include "layer/Sampler3D.h" // Reuse the BBox struct

namespace layer {

    // The abstract base class for 2D (Image) sampling
    class ISampler2D : public ClassParameter {
    public:
        template<class T, class Factory>
        ISampler2D(const RegisterClassParameter<T, Factory>& registration)
                : ClassParameter(registration) {}

        virtual ~ISampler2D() = default;

        // Process image bboxes
        virtual void add_image_bboxes(const std::vector<BBox>& bboxes) = 0;

        // Sample patch: Returns {x, y} based on image index
        virtual std::pair<size_t, size_t> sample_patch(
                size_t image_index,
                size_t input_width, size_t input_height,
                size_t filter_w, size_t filter_h,
                std::default_random_engine& rng) = 0;
    };

    // Random 2D Sampler
    class Sampler2DRandom : public ISampler2D {
    public:
        Sampler2DRandom();
        void add_image_bboxes(const std::vector<BBox>& bboxes) override;

        std::pair<size_t, size_t> sample_patch(
                size_t image_index,
                size_t input_width, size_t input_height,
                size_t filter_w, size_t filter_h,
                std::default_random_engine& rng) override;
    };

    // Facial Sampler (treats each image in the index as a unique entity)
    class Sampler2DFacial : public ISampler2D {
    private:
        std::vector<BBox> _image_bboxes; // One BBox per image index

    public:
        Sampler2DFacial();
        void add_image_bboxes(const std::vector<BBox>& bboxes) override;

        std::pair<size_t, size_t> sample_patch(
                size_t image_index,
                size_t input_width, size_t input_height,
                size_t filter_w, size_t filter_h,
                std::default_random_engine& rng) override;
    };

    // Sampler centered on Landmarks
    class Sampler2DLandmark : public ISampler2D {
    private:
        std::vector<BBox> _image_bboxes;

    public:
        Sampler2DLandmark();
        void add_image_bboxes(const std::vector<BBox>& bboxes) override;

        std::pair<size_t, size_t> sample_patch(
                size_t image_index,
                size_t input_width, size_t input_height,
                size_t filter_w, size_t filter_h,
                std::default_random_engine& rng) override;
    };

    class Sampler2DFactory : public ClassParameterFactory<ISampler2D, Sampler2DFactory> {
    public:
        Sampler2DFactory() : ClassParameterFactory<ISampler2D, Sampler2DFactory>("ISampler2D") {}
    };

}