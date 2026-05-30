#ifndef _PROCESS_PERCENTILE_SCALING_H
#define _PROCESS_PERCENTILE_SCALING_H

#include <filesystem>
#include <iostream>
#include <vector>
#include <algorithm>
#include "Process.h"
#include "NumpyReader.h"

namespace process
{
    class PercentileScaling : public UniquePassProcess
    {
    public:
        PercentileScaling();
        PercentileScaling(float percentile);

        virtual Shape compute_shape(const Shape &shape);
        virtual void process_train(const std::string &label, Tensor<float> &sample);
        virtual void process_test(const std::string &label, Tensor<float> &sample);

    private:
        void _process(const std::string &label, Tensor<float> &in) const;

        size_t _width;
        size_t _height;
        size_t _depth;
        size_t _conv_depth;

        float _percentile;
    };
}

#endif