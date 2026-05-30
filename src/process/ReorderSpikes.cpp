#include "process/ReorderSpikes.h"
#include <cmath>
#include <algorithm>

using namespace process;

static RegisterClassParameter<ReorderSpikes, ProcessFactory> _register("ReorderSpikes");

ReorderSpikes::ReorderSpikes()
        : UniquePassProcess(_register),
          _t_obj(0),
          _sigma(0.05f)
{
    add_parameter("t_obj", _t_obj);
    add_parameter("sigma", _sigma);
}

ReorderSpikes::ReorderSpikes(Time t_obj, float sigma) : ReorderSpikes() {
    parameter<float>("t_obj").set(t_obj);
    parameter<float>("sigma").set(sigma);
}

Shape ReorderSpikes::compute_shape(const Shape& previous_shape)
{
    return previous_shape;
}

void ReorderSpikes::process_train(const std::string& label, Tensor<float>& sample)
{
    reorder(sample);
}

void ReorderSpikes::process_test(const std::string& label, Tensor<float>& sample)
{
    reorder(sample);
}

void ReorderSpikes::reorder(Tensor<float>& sample)
{
    size_t size = sample.shape().product();

    for (size_t i = 0; i < size; i++)
    {
        float t = sample.at_index(i);

        if (t == INFINITE_TIME) {
            continue;
        }

        const double miu = static_cast<double>(_t_obj);
        const double s = static_cast<double>(_sigma);

        double diff = static_cast<double>(t) - miu;
        double importance = std::exp(-(diff * diff) / (2.0 * s * s));

        float result = 1.0f - static_cast<float>(importance);

        sample.at_index(i) = std::clamp(result, 0.0f, 1.0f);
    }
}