#include "layer/Sampler2D.h"

namespace layer {

    // ==========================================
    // Sampler2DRandom Implementation
    // ==========================================

    static RegisterClassParameter<Sampler2DRandom, Sampler2DFactory> _random_2d_sampler_register("Sampler2DRandom");

    Sampler2DRandom::Sampler2DRandom()
            : ISampler2D(_random_2d_sampler_register) {}

    void Sampler2DRandom::add_image_bboxes(const std::vector<BBox>& bboxes) {
        // No pre-processing needed
    }

    std::pair<size_t, size_t> Sampler2DRandom::sample_patch(
            size_t image_index, size_t input_width, size_t input_height,
            size_t filter_w, size_t filter_h, std::default_random_engine& rng)
    {
        std::uniform_int_distribution<size_t> rand_x(0, input_width - filter_w);
        std::uniform_int_distribution<size_t> rand_y(0, input_height - filter_h);

        return {rand_x(rng), rand_y(rng)};
    }

    // ==========================================
    // Sampler2DFacial Implementation
    // ==========================================

    static RegisterClassParameter<Sampler2DFacial, Sampler2DFactory> _facial_2d_sampler_register("Sampler2DFacial");

    Sampler2DFacial::Sampler2DFacial()
            : ISampler2D(_facial_2d_sampler_register) {}

    void Sampler2DFacial::add_image_bboxes(const std::vector<BBox>& bboxes) {
        // For 2D, we usually expect one primary BBox per image. 
        // If multiple are provided, we take the union to ensure the face(s) are covered.
        if (bboxes.empty()) {
            _image_bboxes.push_back({0, 0, 0, 0}); // Default empty
            return;
        }

        BBox union_box = bboxes[0];
        for (const auto& box : bboxes) {
            union_box.x_min = std::min(union_box.x_min, box.x_min);
            union_box.x_max = std::max(union_box.x_max, box.x_max);
            union_box.y_min = std::min(union_box.y_min, box.y_min);
            union_box.y_max = std::max(union_box.y_max, box.y_max);
        }
        _image_bboxes.push_back(union_box);
    }

    std::pair<size_t, size_t> Sampler2DFacial::sample_patch(
            size_t image_index, size_t input_width, size_t input_height,
            size_t filter_w, size_t filter_h, std::default_random_engine& rng)
    {
        if (image_index >= _image_bboxes.size()) return {0, 0};

        const BBox& box = _image_bboxes[image_index];

        // Ensure the patch starts within a range that includes the BBox but fits in the image
        size_t max_x_start = (box.x_max > filter_w) ? (box.x_max - filter_w) : 0;
        size_t max_y_start = (box.y_max > filter_h) ? (box.y_max - filter_h) : 0;

        // Clamp to image boundaries
        max_x_start = std::min(max_x_start, input_width - filter_w);
        max_y_start = std::min(max_y_start, input_height - filter_h);

        size_t min_x_start = std::min(box.x_min, max_x_start);
        size_t min_y_start = std::min(box.y_min, max_y_start);

        std::uniform_int_distribution<size_t> rand_x(min_x_start, max_x_start);
        std::uniform_int_distribution<size_t> rand_y(min_y_start, max_y_start);

        return {rand_x(rng), rand_y(rng)};
    }

    static RegisterClassParameter<Sampler2DLandmark, Sampler2DFactory> _reg_sampler_landmark("Sampler2DLandmark");

    Sampler2DLandmark::Sampler2DLandmark() : ISampler2D(_reg_sampler_landmark) {}

    void Sampler2DLandmark::add_image_bboxes(const std::vector<BBox>& bboxes) {
        if (bboxes.empty()) {
            _image_bboxes.push_back({0, 0, 0, 0}); // Default empty
            return;
        }

        BBox union_box = bboxes[0];
        for (const auto& box : bboxes) {
            union_box.x_min = std::min(union_box.x_min, box.x_min);
            union_box.x_max = std::max(union_box.x_max, box.x_max);
            union_box.y_min = std::min(union_box.y_min, box.y_min);
            union_box.y_max = std::max(union_box.y_max, box.y_max);
        }
        _image_bboxes.push_back(union_box);
    }

    std::pair<size_t, size_t> Sampler2DLandmark::sample_patch(
            size_t image_index,
            size_t input_width, size_t input_height,
            size_t filter_w, size_t filter_h,
            std::default_random_engine& rng) {

        // 1. Validare index imagine
        if (image_index >= _image_bboxes.size()) {
            return {0, 0};
        }

        const auto& frame_data = _image_bboxes[image_index];
        int x = 0, y = 0;

        // 2. Logica de eșantionare bazată pe Landmark-uri
        if (!frame_data.landmarks.empty()) {
            // Punctele 36-67 reprezintă ochii, nasul și gura (standard 68-point format)
            // Reducem plaja pentru a ne concentra pe trăsăturile feței
            size_t max_idx = std::min(size_t(67), frame_data.landmarks.size() - 1);
            size_t min_idx = std::min(size_t(17), max_idx);

            std::uniform_int_distribution<size_t> rand_pt(min_idx, max_idx);
            size_t pt_idx = rand_pt(rng);

            float target_x = frame_data.landmarks[pt_idx].first;
            float target_y = frame_data.landmarks[pt_idx].second;

            // Adăugăm jitter (zgomot) pentru robustețe
            std::uniform_int_distribution<int> jitter(-2, 2);

            // Calculăm colțul stânga-sus astfel încât landmark-ul să fie în centrul patch-ului
            x = static_cast<int>(target_x - (filter_w / 2.0f)) + jitter(rng);
            y = static_cast<int>(target_y - (filter_h / 2.0f)) + jitter(rng);

        } else {
            // 3. Fallback: Eșantionare aleatorie în interiorul Bounding Box-ului
            int max_x_start = static_cast<int>(frame_data.x_max) - static_cast<int>(filter_w);
            int max_y_start = static_cast<int>(frame_data.y_max) - static_cast<int>(filter_h);

            std::uniform_int_distribution<int> rand_x(frame_data.x_min, std::max((int)frame_data.x_min, max_x_start));
            std::uniform_int_distribution<int> rand_y(frame_data.y_min, std::max((int)frame_data.y_min, max_y_start));

            x = rand_x(rng);
            y = rand_y(rng);
        }

        // 4. Clamping: Ne asigurăm că patch-ul nu iese din limitele imaginii
        x = std::clamp(x, 0, static_cast<int>(input_width > filter_w ? input_width - filter_w : 0));
        y = std::clamp(y, 0, static_cast<int>(input_height > filter_h ? input_height - filter_h : 0));

        return {static_cast<size_t>(x), static_cast<size_t>(y)};
    }

} // namespace layer