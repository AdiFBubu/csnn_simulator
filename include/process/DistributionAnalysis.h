#ifndef _DISTRIBUTION_ANALYSIS_H
#define _DISTRIBUTION_ANALYSIS_H

#include "Process.h"
#include <vector>
#include <string>

namespace process
{
    class DistributionAnalysis : public TwoPassProcess
    {
    public:
        DistributionAnalysis();

        virtual Shape compute_shape(const Shape& previous_shape) override;

        virtual void compute(const std::string& label, const Tensor<float>& sample) override;

        virtual void process_train(const std::string& label, Tensor<float>& sample) override;

        virtual void process_test(const std::string& label, Tensor<float>& sample) override;

    private:
        void display_distribution();

        std::vector<long long> _bins;
        long long _total_elements;
        bool _was_displayed;
    };
}

#endif