#ifndef _ANALYSIS_SVM_H
#define _ANALYSIS_SVM_H

#include "Analysis.h"
#include "dep/libsvm/svm.h"
#include <filesystem>
#include <map>
#include <string>
#include "tool/Operations.h"

namespace analysis {
    class Svm : public TwoPassAnalysis {

    public:
        // default constructor (without saving in a file)
        Svm();
        // constructor with saving the probabilities in a file
        Svm(const std::string& output_dir);
        // draw = 0: no saving, draw = 1: save probabilities in a file
        Svm(const size_t &draw, const std::string& output_dir = "");

        Svm(const Svm& that) = delete;
        Svm& operator=(const Svm& that) = delete;

        virtual void resize(const Shape& shape);
        virtual void compute(const std::string& label, const Tensor<float>& sample);
        virtual void process_train(const std::string& label, const Tensor<float>& sample);
        virtual void process_test(const std::string& label, const Tensor<float>& sample);

        virtual void before_train();
        virtual void after_train();
        virtual void before_test();
        virtual void after_test();

        virtual bool load_params(const std::string& filename);
        virtual bool save_params(const std::string& filename);

        const std::map<std::string, double>& get_probabilities() const { return _last_probabilities; }

    private:
        float _c;
        std::string _output_dir;

        std::map<std::string, double> _label_index;
        std::map<std::string, double> _last_probabilities;

        size_t _size;
        size_t _node_count;
        size_t _sample_count;
        size_t _draw;

        svm_problem _problem;
        svm_model* _model;
        svm_node* _train_nodes;
        svm_node* _test_nodes;

        size_t _correct_sample;
        size_t _total_sample;
    };
}

#endif