#include "analysis/Svm.h"
#include "Experiment.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iostream>

using namespace analysis;

static RegisterClassParameter<Svm, AnalysisFactory> _register("Svm");

// 1. Constructor Default
Svm::Svm() : TwoPassAnalysis(_register),
             _c(0), _output_dir(""), _label_index(), _size(0), _node_count(0), _sample_count(0), _draw(0),
             _problem(), _model(nullptr), _train_nodes(nullptr), _test_nodes(nullptr),
             _correct_sample(0), _total_sample(0)
{
    add_parameter("c", _c, 1.0f);
}

// 2. Constructor pentru add_analysis<Svm>("path")
Svm::Svm(const std::string& output_dir) : TwoPassAnalysis(_register),
                                          _c(0), _output_dir(output_dir), _label_index(), _size(0), _node_count(0), _sample_count(0), _draw(0),
                                          _problem(), _model(nullptr), _train_nodes(nullptr), _test_nodes(nullptr),
                                          _correct_sample(0), _total_sample(0)
{
    add_parameter("c", _c, 1.0f);
}

// 3. Constructor pentru add_analysis<Svm>(draw, "path")
Svm::Svm(const size_t &draw, const std::string& output_dir) : TwoPassAnalysis(_register),
                                                              _draw(draw), _c(0), _output_dir(output_dir), _label_index(), _size(0), _node_count(0), _sample_count(0),
                                                              _problem(), _model(nullptr), _train_nodes(nullptr), _test_nodes(nullptr),
                                                              _correct_sample(0), _total_sample(0)
{
    add_parameter("c", _c, 1.0f);
}

void Svm::resize(const Shape& shape) {
    _node_count = 0;
    _sample_count = 0;
    _size = shape.product();
    _label_index.clear();
}

void Svm::compute(const std::string& label, const Tensor<float>& sample) {
    if(_label_index.find(label) == std::end(_label_index)) {
        _label_index.emplace(label, _label_index.size());
    }

    for(size_t j = 0; j < _size; j++) {
        if(sample.at_index(j) != 0.0) {
            _node_count++;
        }
    }
    _node_count++;
    _sample_count++;

    if (_draw == 1) {
        std::string _file_path = std::filesystem::current_path().string();
        std::string _expName = experiment().name();
        std::string _LayerIndex = std::to_string(0);
        std::filesystem::create_directories(_file_path + "/ExtractedFeatures/SVM/" + _expName + "_" + _LayerIndex + "/");
        SaveWeights(_file_path + "/ExtractedFeatures/SVM/" + _expName + "_" + _LayerIndex + "/" + _expName + "_" + _LayerIndex + ".json", label, sample);
        Tensor<float>::draw_tensor(_file_path + "/ExtractedFeatures/SVM/" + _expName + "_" + _LayerIndex + "/" + _expName + "_" + _LayerIndex + "_" + std::to_string(_sample_count) + "_", sample);
    }
}

void Svm::before_train() {
    _train_nodes = new struct svm_node[_node_count];
    _test_nodes = new struct svm_node[_size];

    _problem.l = _sample_count;
    _problem.y = new double[_sample_count];
    _problem.x = new struct svm_node*[_sample_count];

    _sample_count = 0;
    _node_count = 0;
}

void Svm::process_train(const std::string& label, const Tensor<float>& sample) {
    _problem.y[_sample_count] = _label_index[label];
    _problem.x[_sample_count] = _train_nodes + _node_count;

    for(size_t j = 0; j < _size; j++) {
        float v = sample.at_index(j);
        if(v != 0.0) {
            _train_nodes[_node_count].index = j + 1;
            _train_nodes[_node_count].value = v;
            _node_count++;
        }
    }
    _train_nodes[_node_count].index = -1;
    _node_count++;
    _sample_count++;
}

void Svm::after_train() {
    struct svm_parameter parameters;

    parameters.svm_type = C_SVC;
    parameters.kernel_type = LINEAR;
    parameters.degree = 3;
    parameters.gamma = 1.0 / static_cast<float>(_size);
    parameters.coef0 = 0;
    parameters.nu = 0.5;
    parameters.cache_size = 100;
    parameters.C = _c;
    parameters.eps = 1e-3;
    parameters.p = 0.1;
    parameters.shrinking = 1;
    parameters.probability = 1;

    parameters.nr_weight = 0;
    parameters.weight_label = NULL;
    parameters.weight = NULL;

    experiment().print() << "Train svm" << std::endl;
    _model = ::svm_train(&_problem, &parameters);
}

void Svm::before_test() {
    if (_test_nodes == nullptr) {
        _test_nodes = new struct svm_node[_size];
    }
    _correct_sample = 0;
    _total_sample = 0;
}

void Svm::process_test(const std::string& label, const Tensor<float>& sample) {
    if (_model == nullptr) return;

    if (_test_nodes == nullptr) {
        _test_nodes = new struct svm_node[_size];
    }

    size_t node_cursor = 0;
    for(size_t j = 0; j < _size; j++) {
        float v = sample.at_index(j);
        if(v != 0.0) {
            _test_nodes[node_cursor].index = j + 1;
            _test_nodes[node_cursor].value = v;
            node_cursor++;
        }
    }
    _test_nodes[node_cursor].index = -1;

    double y_pred;
    _last_probabilities.clear();

    if (svm_check_probability_model(_model) != 0) {
        int nr_class = svm_get_nr_class(_model);
        double* prob_estimates = new double[nr_class];
        int* labels = new int[nr_class];

        y_pred = svm_predict_probability(_model, _test_nodes, prob_estimates);
        svm_get_labels(_model, labels);

        std::map<int, std::string> reverse_labels;
        for (const auto& p : _label_index) {
            reverse_labels[static_cast<int>(p.second)] = p.first;
        }

        for (int i = 0; i < nr_class; i++) {
            std::string class_name = reverse_labels[labels[i]];
            if (class_name.empty()) class_name = std::to_string(labels[i]);
            _last_probabilities[class_name] = prob_estimates[i];
        }

        // --- SALVARE ÎN FIȘIER (doar dacă _output_dir nu e gol) ---
        if (!_output_dir.empty()) {
            try {
                std::filesystem::create_directories(_output_dir);
                std::string file_path = _output_dir + "/sample_" + std::to_string(_total_sample) + ".txt";
                std::ofstream out(file_path);
                if (out.is_open()) {
                    out << "Image_Index: " << _total_sample << "\n";
                    out << "True_Label: " << label << "\n";
                    out << "--- Probabilities ---\n";
                    for (const auto& p : _last_probabilities) {
                        out << p.first << ": " << std::fixed << std::setprecision(4) << p.second << "\n";
                    }
                    out.close();
                }
            } catch (...) {}
        }

        delete[] prob_estimates;
        delete[] labels;
    } else {
        y_pred = ::svm_predict(_model, _test_nodes);
    }

    auto it = _label_index.find(label);
    if(it != std::end(_label_index) && y_pred == it->second) {
        _correct_sample++;
    }
    _total_sample++;
}

void Svm::after_test() {
    experiment().log() << "===SVM===" << std::endl;
    experiment().log() << "classification rate: " <<
                       (static_cast<float>(_correct_sample) / static_cast<float>(_total_sample) * 100.0) << "% (" <<
                       _correct_sample << "/" << _total_sample << ")" << std::endl;

    if (_problem.y) { delete[] _problem.y; _problem.y = nullptr; }
    if (_problem.x) { delete[] _problem.x; _problem.x = nullptr; }
    if (_train_nodes) { delete[] _train_nodes; _train_nodes = nullptr; }
    if (_test_nodes) { delete[] _test_nodes; _test_nodes = nullptr; }
    if (_model) { svm_free_and_destroy_model(&_model); _model = nullptr; }
}

bool Svm::save_params(const std::string& filename) {
    try {
        std::filesystem::path dir(filename);
        std::filesystem::create_directories(dir);

        if (_model == nullptr) {
            experiment().log() << "SVM::save_params: no model to save" << std::endl;
            return false;
        }

        std::string model_file = (dir / "svm.model").string();
        int res = svm_save_model(model_file.c_str(), _model);
        if (res != 0) {
            experiment().log() << "SVM::save_params: svm_save_model failed (" << res << ")" << std::endl;
            return false;
        }

        std::string labels_file = (dir / "labels.txt").string();
        std::ofstream out(labels_file);
        if (!out.good()) {
            experiment().log() << "SVM::save_params: failed to open labels file " << labels_file << std::endl;
            return false;
        }
        for (const auto &p : _label_index) {
            out << p.first << '\t' << p.second << '\n';
        }
        out.close();

        experiment().log() << "SVM::save_params: saved model and labels to " << dir.string() << std::endl;
        return true;
    } catch (const std::exception &e) {
        experiment().log() << "SVM::save_params exception: " << e.what() << std::endl;
        return false;
    }
}

bool Svm::load_params(const std::string& filename) {
    try {
        std::filesystem::path dir(filename);
        if (!std::filesystem::exists(dir)) return false;

        std::string model_file = (dir / "svm.model").string();
        if (std::filesystem::exists(model_file)) {
            svm_model *loaded = svm_load_model(model_file.c_str());
            if (loaded) {
                if (_model) svm_free_and_destroy_model(&_model);
                _model = loaded;
                experiment().log() << "SVM::load_params: loaded model from " << model_file << std::endl;
            } else {
                experiment().log() << "SVM::load_params: svm_load_model returned null for " << model_file << std::endl;
                return false;
            }
        } else {
            experiment().log() << "SVM::load_params: model file not found: " << model_file << std::endl;
            return false;
        }

        std::string labels_file = (dir / "labels.txt").string();
        _label_index.clear();
        if (std::filesystem::exists(labels_file)) {
            std::ifstream in(labels_file);
            std::string line;
            while (std::getline(in, line)) {
                if (line.empty()) continue;
                size_t sep = line.find_last_of('\t');
                if (sep == std::string::npos) sep = line.find_last_of(' ');
                if (sep == std::string::npos) continue;
                std::string label = line.substr(0, sep);
                std::string idxs = line.substr(sep + 1);
                try {
                    double idx = std::stod(idxs);
                    _label_index[label] = idx;
                } catch (...) {}
            }
            experiment().log() << "SVM::load_params: loaded labels from " << labels_file << std::endl;
        } else {
            int nr = svm_get_nr_class(_model);
            if (nr > 0) {
                std::vector<int> labels(nr);
                svm_get_labels(_model, labels.data());
                for (int i = 0; i < nr; ++i) {
                    std::string key = std::to_string(labels[i]);
                    _label_index[key] = labels[i];
                }
                experiment().log() << "SVM::load_params: created label mapping from model labels" << std::endl;
            }
        }

        return true;
    } catch (const std::exception &e) {
        experiment().log() << "SVM::load_params exception: " << e.what() << std::endl;
        return false;
    }
}