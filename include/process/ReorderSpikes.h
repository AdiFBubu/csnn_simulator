#ifndef _REORDER_SPIKES_H
#define _REORDER_SPIKES_H

#include "Process.h"
#include "Tensor.h"

namespace process
{

    class ReorderSpikes : public UniquePassProcess
    {
    public:
        ReorderSpikes();
        ReorderSpikes(Time t_obj, float sigma);

        virtual Shape compute_shape(const Shape& previous_shape) override;

        virtual void process_train(const std::string& label, Tensor<float>& sample) override;
        virtual void process_test(const std::string& label, Tensor<float>& sample) override;

    private:

        void reorder(Tensor<float>& sample);

        float _t_obj;
        float _sigma;

    };
}

#endif