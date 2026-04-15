#ifndef FEATUREMAPS_H
#define FEATUREMAPS_H

#include "Process.h" //
#include <opencv2/opencv.hpp>
#include <string>

namespace process {
    class FeatureMaps : public UniquePassProcess {
    public:
        FeatureMaps();

        FeatureMaps(const std::string &file_prefix);

        virtual Shape compute_shape(const Shape &shape);

        virtual void process_train(const std::string &label, Tensor<float> &sample);

        virtual void process_test(const std::string &label, Tensor<float> &sample);

    private:
        void _process(const std::string &label, const Tensor<float> &sample) const;

        std::string _file_prefix;
        size_t _cell_w;
        size_t _cell_h;

        size_t _width;
        size_t _height;
        size_t _depth;       // Corresponds to num_filters (z)
        size_t _conv_depth;  // Corresponds to 3D depth (k)

        size_t _no_samples; // Counter to keep track of the number of samples processed, used for naming output files
    };
}
#endif // FEATUREMAPS_H