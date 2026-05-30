#include "Experiment.h"

#include "dataset/CK_Plus.h"

#include "stdp/Multiplicative.h"

#include "stdp/Biological.h"

//#include "layer/FaceElypsesCutout3D.h"

//#include "layer/FaceElypsesCutout3D.h"

#include "layer/Convolution3D.h"

#include "layer/Pooling.h"

#include "Distribution.h"

#include "execution/SparseIntermediateExecution.h"

#include "analysis/Svm.h"

#include "analysis/Activity.h"

#include "analysis/Coherence.h"

#include "process/Input.h"

#include "process/Scaling.h"

#include "process/Pooling.h"

#include "process/SimplePreprocessing.h"

#include "process/CompositeChannels.h"

#include "process/OnOffFilter.h"

#include "process/OnOffTempFilter.h"

#include "process/EarlyFusion.h"

#include "process/LateFusion.h"

#include "process/SeparateSign.h"

#include "process/MaxScaling.h"

#include "process/PercentileScaling.h"

#include "layer/Sampler3D.h"

#include "process/FeatureMaps.h"

#include "execution/TrainingSparseExecution.h"

#include "execution/TestingSparseExecution.h"

#include "process/DistributionAnalysis.h"

#include "process/ReorderSpikes.h"

int main(int argc, char **argv)
{
    // Parametri existen?i
    size_t _filter_width = (argc > 1) ? atoi(argv[1]) : 5;
    size_t _filter_height = (argc > 2) ? atoi(argv[2]) : 5;
    size_t _filter_depth = (argc > 3) ? atoi(argv[3]) : 3;
    size_t _temporal_sum_pooling = (argc > 4) ? atoi(argv[4]) : 2;
    int _epochs = (argc > 5) ? atoi(argv[5]) : 800;

    float _th = (argc > 14) ? atof(argv[14]) : 4.0f;
    size_t _on_off_size = (argc > 15) ? atoi(argv[15]) : 7;
    float _on_off_contrast = (argc > 16) ? atof(argv[16]) : 0.1f;    
    unsigned int random_seed = (argc > 6) ? atoi(argv[6]) : 42;

    size_t _spatial_pooling = (argc > 7) ? atoi(argv[7]) : 4;
    float t_obj = (argc > 8) ? atof(argv[8]) : 0.80f;

    // NOI PARAMETRI PENTRU NESTED CV
    int run_mode = (argc > 9) ? atoi(argv[9]) : 0; // 0 = Optuna CV, 1 = Testare Finala
    int holdout_test_fold = (argc > 10) ? atoi(argv[10]) : 10;
    
    std::cout << "Random seed: " << random_seed << std::endl;
    std::cout << "Spatial pooling: " << _spatial_pooling << std::endl;
    std::cout << "Run mode (0=CV, 1=Test): " << run_mode << std::endl;
    std::cout << "Holdout test fold: " << holdout_test_fold << std::endl;
    
    // Dataset paths din environment variables
    const char* csv_path_ptr = std::getenv("CK_PLUS_CSV_PATH");
    const char* images_dir_ptr = std::getenv("CK_PLUS_IMAGES_DIR");
    const char* landmarks_dir_ptr = std::getenv("CK_PLUS_LANDMARKS_DIR");
    
    std::string csv_path(csv_path_ptr);
    std::string images_dir(images_dir_ptr);
    std::string landmarks_dir = (landmarks_dir_ptr != nullptr) ? landmarks_dir_ptr : "";
    
    int num_folds = 10;
    size_t _frame_size_width = 48;
    size_t _frame_size_height = 48;

    time_t start_time;
    time(&start_time);

    // Setam fold-urile care trebuie rulate în func?ie de modul de execu?ie
    std::vector<int> val_folds_to_run;
    if (run_mode == 0) {
        // Pentru validare, luam toate foldurile exceptând cel de test holdout
        // 1. Modul Nested CV (Inner loop) - validare pe 8 folduri
        for(int i = 1; i <= num_folds; i++) {
            if (i != holdout_test_fold) val_folds_to_run.push_back(i);
        }
    } else if (run_mode == 1) {
        // 2. Modul Test Holdout (Single fold)
        val_folds_to_run.push_back(holdout_test_fold);
    }
    else if (run_mode == 2) {
    // 3. NOU: Modul Standard 10-Fold CV (Pentru hiperparametri universali)
    	for(int i = 1; i <= num_folds; i++) {
        	val_folds_to_run.push_back(i);
    	}
    }

        for (int current_fold : val_folds_to_run) {
	// for (int current_fold = 1; current_fold < 2; current_fold ++) {


        std::string _dataset = "CK_Plus_" + std::to_string(start_time) + "_3D_" + 
                               std::to_string(_filter_width) + "x" + 
                               std::to_string(_filter_height) + "x" + 
                               std::to_string(_filter_depth) + "_tp" +
                               std::to_string(_temporal_sum_pooling) + "_sp" +
                               std::to_string(_spatial_pooling) + "_fold" + 
                               std::to_string(current_fold) + "_epochs" + std::to_string(_epochs) +
                               "_seed" + std::to_string(random_seed);

        Experiment<SparseIntermediateExecution> experiment(argc, argv, _dataset);

        dataset::CK_Plus ck_plus(csv_path, images_dir, num_folds, random_seed,
                                 _frame_size_width, _frame_size_height, false, landmarks_dir);
        if (!ck_plus.load()) {
            experiment.log() << "Failed to load CK+ dataset" << std::endl;
            return 1;
        }

        // --- EXTRAGEREA SETURILOR PENTRU NESTED CV ---
        std::vector<dataset::CK_Plus::ImageSequence> training_sequences;
        std::vector<dataset::CK_Plus::ImageSequence> testing_sequences;

        if (run_mode == 0) {
            // CV: Fold-ul curent este cel de validare
            testing_sequences = ck_plus.getTestSequences(current_fold);
            // Antrenarea se face pe restul de 8 fold-uri
            for (int i = 1; i <= num_folds; i++) {
                if (i != holdout_test_fold && i != current_fold) {
                    auto seqs = ck_plus.getTestSequences(i);
                    training_sequences.insert(training_sequences.end(), seqs.begin(), seqs.end());
                }
            }
            experiment.log() << "Mod CV - Antrenare pe 8 fold-uri, Validare pe fold: " << current_fold << std::endl;
        } else if (run_mode == 1) {
            // TESTARE FINALA: Antrenare pe 9 fold-uri, Testare pe fold-ul de holdout
            training_sequences = ck_plus.getTrainingSequences(holdout_test_fold);
            testing_sequences = ck_plus.getTestSequences(holdout_test_fold);
            experiment.log() << "Mod TEST - Antrenare pe 9 fold-uri, Testare pe fold: " << holdout_test_fold << std::endl;
        }
	else if (run_mode == 2) {
    	    // NOU - Standard 10-Fold CV: 
    	    // Pentru fiecare pas, 'current_fold' este fold-ul de test (1 singur), restul de 9 sunt pentru antrenare
    	    training_sequences = ck_plus.getTrainingSequences(current_fold);
    	    testing_sequences = ck_plus.getTestSequences(current_fold);
    	    experiment.log() << "Standard 10-Fold CV - Antrenare pe 9 fold-uri, Testare pe fold: " << current_fold << std::endl;
	}

        // Analiza cadre (doar pentru logging silen?ios ca înainte)
        size_t total_training_frames = 0;
        size_t total_testing_frames = 0;
        std::map<int, int> training_emotions;
        std::map<int, int> testing_emotions;

        for (auto& seq : training_sequences) {
            total_training_frames += seq.frames.size();
            training_emotions[seq.emotion]++;
        }
        for (auto& seq : testing_sequences) {
            total_testing_frames += seq.frames.size();
            testing_emotions[seq.emotion]++;
        }

        size_t filter_number = (argc > 12) ? atoi(argv[12]) : 64;
        size_t tmp_stride = 1;
        size_t sampling_size = _epochs;
        float w_lr = (argc > 11) ? atof(argv[11]) : 0.009f;
	float th_lr = w_lr * 10.0f;

        experiment.push<process::DefaultOnOffFilter>(_on_off_size, _on_off_contrast, 1.0);
        // experiment.push<process::FeatureMaps>("Debug_Preprocess/");
        experiment.push<process::MaxScaling>();
        experiment.push<LatencyCoding>();
        // experiment.push<process::DistributionAnalysis>();

        auto &conv1 = experiment.push<layer::Convolution3D>(
                _filter_width, _filter_height, _filter_depth, filter_number, "", 1, 1, tmp_stride);
        conv1.set_name("conv1");
        conv1.parameter<bool>("draw").set(false);
        conv1.parameter<bool>("draw_feature_maps").set(false);
        conv1.parameter<bool>("save_weights").set(false);
        conv1.parameter<bool>("save_random_start").set(false);
        conv1.parameter<bool>("log_spiking_neuron").set(false);
        conv1.parameter<bool>("inhibition").set(true);
        conv1.parameter<bool>("wta_infer").set(false);
        conv1.parameter<uint32_t>("epoch").set(sampling_size);
        conv1.parameter<float>("annealing").set(0.95f);
        conv1.parameter<float>("min_th").set(1.0f);
        conv1.parameter<float>("t_obj").set(t_obj);
        conv1.parameter<float>("lr_th").set(th_lr);
        conv1.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
        conv1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(_th, 0.1);
        conv1.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 0.1f);
       // conv1.parameter<layer::ISampler3D>("sampler").set<layer::Sampler3DLandmark>();
	 conv1.parameter<layer::ISampler3D>("sampler").set<layer::Sampler3DRandom>();
	// conv1.parameter<layer::ISampler3D>("sampler").set<layer::Sampler3DFacialPerFrame>();
	// conv1.parameter<layer::ISampler3D>("sampler").set<layer::Sampler3DOnOffSaliency>();

	int output_mode = (argc > 13) ? atoi(argv[13]) : 1;

        auto &conv1_out = experiment.output<TimeObjectiveOutput>(conv1, t_obj, output_mode);
        // conv1_out.add_postprocessing<process::FeatureMaps>("FMs3D/");
        conv1_out.add_postprocessing<process::SumPooling>(_spatial_pooling, _spatial_pooling);
        conv1_out.add_postprocessing<process::TemporalPooling>(_temporal_sum_pooling);
        conv1_out.add_postprocessing<process::FeatureScaling>();
        // conv1_out.add_analysis<analysis::Activity>();
        // conv1_out.add_analysis<analysis::Coherence>();
        conv1_out.add_analysis<analysis::Svm>("svm_probabilities");

        // experiment.push<process::DistributionAnalysis>();
        // experiment.push<process::ReorderSpikes>(t_obj, 0.1);
        // experiment.push<process::DistributionAnalysis>();

        layer::ISampler3D* sampler_ptr = conv1.get_sampler();

        int training_count = 0;
        for (auto& seq : training_sequences) {
            if (seq.frames.empty()) continue;
            try {
                if (sampler_ptr != nullptr) sampler_ptr->add_sequence_bboxes(seq.bboxes);
                experiment.add_train<dataset::CK_Plus_Input>(seq, _frame_size_width, _frame_size_height);
                training_count++;
            } catch (const std::exception& e) {
                experiment.log() << "Eroare: " << e.what() << std::endl;
            }
        }
        
        int testing_count = 0;
        for (auto& seq : testing_sequences) {
            if (seq.frames.empty()) continue;
            try {
                experiment.add_test<dataset::CK_Plus_Input>(seq, _frame_size_width, _frame_size_height);
                testing_count++;
            } catch (const std::exception& e) {
                experiment.log() << "Eroare: " << e.what() << std::endl;
            }
        }

        experiment.log() << "Antrenare pe " << training_count << " / Testare pe " << testing_count << " secvente." << std::endl;
        experiment.run(10000);
    }
    
    return 0;
}