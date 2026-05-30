#include "process/PercentileScaling.h"

using namespace process;

// Înregistrarea clasei în fabrică (folosind același mecanism ca în MaxScaling)
static RegisterClassParameter<PercentileScaling, ProcessFactory> _register_2("PercentileScaling");

PercentileScaling::PercentileScaling()
        : UniquePassProcess(_register_2), _width(0), _height(0), _depth(0), _conv_depth(0), _percentile(1.0f)
{
}

PercentileScaling::PercentileScaling(float percentile) : PercentileScaling()
{
    _percentile = percentile;
}

Shape PercentileScaling::compute_shape(const Shape &shape)
{
    _height = shape.dim(0);
    _width = shape.dim(1);
    _depth = shape.dim(2);
    _conv_depth = shape.dim(3);

    return Shape({_height, _width, _depth, _conv_depth});
}

void PercentileScaling::process_train(const std::string &label, Tensor<float> &sample)
{
    _process(label, sample);
}

void PercentileScaling::process_test(const std::string &label, Tensor<float> &sample)
{
    _process(label, sample);
}

void PercentileScaling::_process(const std::string &label, Tensor<float> &in) const
{
    std::vector<float> data(in.begin(), in.end());

    auto nth = data.begin() + static_cast<size_t>(_percentile * (data.size() - 1));

    std::nth_element(data.begin(), nth, data.end());
    float thresholdValue = *nth;

    if (thresholdValue > 0.0f)
    {
        for (auto &pixel : in)
        {
            pixel /= thresholdValue;
            if (pixel > 1.0f) pixel = 1.0f;
        }
    }
}