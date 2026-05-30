#include "process/DistributionAnalysis.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

using namespace process;

static RegisterClassParameter<DistributionAnalysis, ProcessFactory> _register("DistributionAnalysis");

DistributionAnalysis::DistributionAnalysis()
        : TwoPassProcess(_register),
          _total_elements(0),
          _was_displayed(false)
{
    _bins.resize(10, 0);
}

Shape DistributionAnalysis::compute_shape(const Shape& previous_shape)
{
    return previous_shape;
}

void DistributionAnalysis::compute(const std::string& label, const Tensor<float>& sample)
{
    for (size_t i = 0; i < sample.shape().product(); ++i)
    {
        float val = sample.at_index(i);

        if (val == INFINITE_TIME)
        {
            continue;
        }

        val = std::max(0.0f, std::min(1.0f, val));

        int bin_index = static_cast<int>(val * 10.0f);

        if (bin_index >= 10) bin_index = 9;

        _bins[bin_index]++;
        _total_elements++;
    }
}

void DistributionAnalysis::process_train(const std::string& label, Tensor<float>& sample)
{

    if (!_was_displayed)
    {
        display_distribution();
        _was_displayed = true;
    }

}

void DistributionAnalysis::process_test(const std::string& label, Tensor<float>& sample)
{
}

void DistributionAnalysis::display_distribution()
{
    std::cout << "\n==========================================================" << std::endl;
    std::cout << "   GLOBAL VALUE DISTRIBUTION ANALYSIS" << std::endl;
    std::cout << "   Total elements processed: " << _total_elements << std::endl;
    std::cout << "==========================================================" << std::endl;

    if (_total_elements == 0)
    {
        std::cout << "   [!] No data collected during first pass." << std::endl;
        return;
    }

    for (size_t i = 0; i < _bins.size(); ++i)
    {
        float start = i / 10.0f;
        float end = (i + 1) / 10.0f;
        double percentage = (static_cast<double>(_bins[i]) / _total_elements) * 100.0;

        // Formatare tabelara
        std::cout << "   [" << std::fixed << std::setprecision(1) << start
                  << " - " << end << "] : "
                  << std::setw(6) << std::setprecision(2) << percentage << "% "
                  << "(" << _bins[i] << " px)" << std::endl;
    }
    std::cout << "==========================================================\n" << std::endl;
}