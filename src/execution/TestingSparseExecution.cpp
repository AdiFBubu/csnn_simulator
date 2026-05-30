#include "execution/TestingSparseExecution.h"
#include "Math.h"

TestingSparseExecution::TestingSparseExecution(ExperimentType& experiment) :
	_experiment(experiment), _test_set() {

}

void TestingSparseExecution::process(size_t refresh_interval) {
	_load_data();

	for(size_t i=0; i<_experiment.process_number(); i++) {
		_experiment.print() << "Process " << _experiment.process_at(i).factory_name() << "." << _experiment.process_at(i).class_name();
		if(!_experiment.process_at(i).name().empty()) {
			_experiment.print() << " (" << _experiment.process_at(i).name() << ")";
		}
		_experiment.print() << std::endl;

		// Load trained parameters
		std::string process_load_path = _experiment.model_path() + "/" + _experiment.process_at(i).factory_name() + "." + _experiment.process_at(i).class_name();
		if (!_experiment.process_at(i).name().empty()) {
			process_load_path += "." + _experiment.process_at(i).name();
		}
		process_load_path += "/";
		bool loaded = _experiment.process_at(i).load_params(process_load_path);
		if (loaded) {
			_experiment.log() << "Load trained parameters at " << process_load_path << std::endl;
		}

		// Load analysis parameters for outputs tied to this process
		for (size_t o = 0; o < _experiment.output_count(); ++o) {
			if (_experiment.output_at(o).index() == i) {
				Output &output = _experiment.output_at(o);
				for (Analysis *analysis : output.analysis()) {
                    std::string full_output_name = output.name();
                    size_t last_dash_idx = full_output_name.find_last_of('-');
                    std::string short_name = (last_dash_idx != std::string::npos)
                                             ? full_output_name.substr(last_dash_idx + 1)
                                             : full_output_name;
                    std::string analysis_load_path = _experiment.model_path() + "/" + short_name + "." + analysis->class_name() + "/";
                    _experiment.log() << "Attempting to load analysis params from " << analysis_load_path << std::endl;
                    bool a_loaded = analysis->load_params(analysis_load_path);
                    if (a_loaded) {
                        _experiment.log() << "Load trained analysis parameters at " << analysis_load_path << std::endl;
                    } else {
                        _experiment.log() << "No analysis parameters found at " << analysis_load_path << std::endl;
                    }
				}
			}
		}

		_process_test_data(_experiment.process_at(i), _test_set);
		_process_output(i);
	}

	_test_set.clear();
}

Tensor<Time> TestingSparseExecution::compute_time_at(size_t i) const {
	throw std::runtime_error("Unimplemented");
}

void TestingSparseExecution::_load_data() {
	for(Input* input : _experiment.test_data()) {
		size_t count = 0;
		while(input->has_next()) {
			auto entry = input->next();
			_test_set.emplace_back(entry.first, to_sparse_tensor(entry.second));
			count ++;
		}
		_experiment.log() << "Load " << count << " test samples from " << input->to_string() << std::endl;
		input->close();
	}
}

void TestingSparseExecution::_process_test_data(AbstractProcess& process, std::vector<std::pair<std::string, SparseTensor<float>>>& data) {
	for(size_t j=0; j<_test_set.size(); j++) {
		Tensor<float> current = from_sparse_tensor(data[j].second);
		process.process_test_sample(data[j].first, current, j, data.size());
		data[j].second = to_sparse_tensor(current);

		if(data[j].second.shape() != process.shape()) {
			throw std::runtime_error("Unexpected shape (actual: "+data[j].second.shape().to_string()+", expected: "+process.shape().to_string()+")");
		}
	}
}

void TestingSparseExecution::_process_output(size_t index) {

	for(size_t i=0; i<_experiment.output_count(); i++) {
		if(_experiment.output_at(i).index() == index) {
			Output& output = _experiment.output_at(i);

			std::cout << "Output " << output.name() << std::endl;

			std::vector<std::pair<std::string, SparseTensor<float>>> output_test_set;

			for(std::pair<std::string, SparseTensor<float>>& entry : _test_set) {
				Tensor<float> current = from_sparse_tensor(entry.second);
				output_test_set.emplace_back(entry.first, to_sparse_tensor(output.converter().process(current)));
			}

			for(Process* process : output.postprocessing()) {
				std::string full_output_name = output.name();
				size_t last_dash_idx = full_output_name.find_last_of('-');
				std::string short_name = (last_dash_idx != std::string::npos)
						? full_output_name.substr(last_dash_idx + 1)
						: full_output_name;
				std::string post_name = short_name + ".post." + process->class_name();
				if (!process->name().empty()) {
					post_name += "." + process->name();
				}
				std::string post_load_path = _experiment.model_path() + "/" + post_name + "/";
				bool p_loaded = process->load_params(post_load_path);
				if (p_loaded) {
					_experiment.log() << "Load postprocess parameters at " << post_load_path << std::endl;
				} else {
					_experiment.log() << "No postprocess parameters found at " << post_load_path << std::endl;
				}

				_experiment.print() << "Process " << process->class_name() << std::endl;
				_process_test_data(*process, output_test_set);
			}

			for(Analysis* analysis : output.analysis()) {
				analysis->before_test();
				for(std::pair<std::string, SparseTensor<float>>& entry : output_test_set) {
					Tensor<float> current = from_sparse_tensor(entry.second);
					analysis->process_test_sample(entry.first, current);
				}
				analysis->after_test();
			}
		}
	}
}
