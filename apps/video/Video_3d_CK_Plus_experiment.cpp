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
#include "layer/Sampler3D.h"
#include "process/FeatureMaps.h"
#include "execution/TrainingSparseExecution.h"
#include "execution/TestingSparseExecution.h"

int main(int argc, char **argv)
{
    // Parse command line arguments with new parameters
    size_t _filter_width = (argc > 1) ? atoi(argv[1]) : 5;
    size_t _filter_height = (argc > 2) ? atoi(argv[2]) : 5;
    size_t _filter_depth = (argc > 3) ? atoi(argv[3]) : 3;
    size_t _temporal_sum_pooling = (argc > 4) ? atoi(argv[4]) : 2;
    
    // Keep epochs and threshold unchanged
    int _epochs = (argc > 5) ? atoi(argv[5]) : 800;
    float _th = 8.0;
    
    // Add random seed parameter
    unsigned int random_seed = (argc > 6) ? atoi(argv[6]) : 42;
    
    // Add spatial pooling parameter
    size_t _spatial_pooling = (argc > 7) ? atoi(argv[7]) : 4;
    
    // Print parameters
    std::cout << "Random seed: " << random_seed << std::endl;
    std::cout << "Spatial pooling: " << _spatial_pooling << std::endl;
    
    // Get dataset paths from environment variables
    const char* csv_path_ptr = std::getenv("CK_PLUS_CSV_PATH");
    const char* images_dir_ptr = std::getenv("CK_PLUS_IMAGES_DIR");
    const char* landmarks_dir_ptr = std::getenv("CK_PLUS_LANDMARKS_DIR");
    
    // Convert to std::string
    std::string csv_path(csv_path_ptr);
    std::string images_dir(images_dir_ptr);
    std::string landmarks_dir = (landmarks_dir_ptr != nullptr) ? landmarks_dir_ptr : "";
    
    int num_folds = 10;
    
    // Video frame dimensions
    size_t _frame_size_width = 112;
    size_t _frame_size_height = 112;

    time_t start_time;
    time(&start_time);

    for (int fold = 1; fold <= num_folds; fold++) {
        // Update experiment name to include all parameters
        std::string _dataset = "CK_Plus_" + std::to_string(start_time) + "_3D_" + 
                               std::to_string(_filter_width) + "x" + 
                               std::to_string(_filter_height) + "x" + 
                               std::to_string(_filter_depth) + "_tp" +
                               std::to_string(_temporal_sum_pooling) + "_sp" +
                               std::to_string(_spatial_pooling) + "_fold" + 
                               std::to_string(fold) + "_epochs" + std::to_string(_epochs) +
                               "_seed" + std::to_string(random_seed);

        Experiment<SparseIntermediateExecution> experiment(argc, argv, _dataset);
//        Experiment<TrainingSparseExecution> experiment(argc, argv, _dataset);
//        Experiment<TestingSparseExecution> experiment(argc, argv, _dataset);

        // Load CK+ dataset with paths from environment variables and use the command line random_seed
        dataset::CK_Plus ck_plus(csv_path, images_dir, num_folds, random_seed,
                                 _frame_size_width, _frame_size_height, false, landmarks_dir);
        if (!ck_plus.load()) {
            experiment.log() << "Failed to load CK+ dataset" << std::endl;
            return 1;
        }
        else {
            experiment.log() << "CK+ dataset loaded successfully" << std::endl;
        }
        
        // Get training and testing sequences for this fold
        auto training_sequences = ck_plus.getTrainingSequences(fold);
        auto testing_sequences = ck_plus.getTestSequences(fold);
        
        
        // Count frames silently - we need this data for validation but won't print details
        size_t total_training_frames = 0;
        size_t total_testing_frames = 0;
        std::map<int, int> training_emotions;
        std::map<int, int> testing_emotions;

        for (auto& seq : training_sequences) {
            total_training_frames += seq.frames.size();
            training_emotions[seq.emotion]++;
        }
        
        for (auto& seq : testing_sequences) {
            if (seq.emotion == 1)
                std::cout << seq.subject << " " << seq.ipostase << " " << seq.emotion << std::endl;
            total_testing_frames += seq.frames.size();
            testing_emotions[seq.emotion]++;
        }
        
        // Network parameters - use the parameterized values
        size_t filter_number = 64;
        size_t tmp_stride = 1;

        size_t sampling_size = _epochs;

        // Preprocessing
        experiment.push<process::DefaultOnOffFilter>(7, 0.35, 5.0);

        // Doar pentru debug, să vezi imaginea după OnOffFilter
        experiment.push<process::FeatureMaps>("Debug_Preprocess/");

        experiment.push<process::MaxScaling>();
        experiment.push<LatencyCoding>();


        // Network parameters
        float t_obj = 0.65;
        float th_lr = 0.09f;
        float w_lr = 0.009f;

        //volatile int debug_marker = 123;
        auto &conv1 = experiment.push<layer::Convolution3D>(
                _filter_width, _filter_height, _filter_depth, filter_number, "", 1, 1, tmp_stride);
        conv1.set_name("conv1");
        conv1.parameter<bool>("draw").set(false);
        conv1.parameter<bool>("draw_feature_maps").set(false);
        conv1.parameter<bool>("save_weights").set(false);
        conv1.parameter<bool>("save_random_start").set(false);
        conv1.parameter<bool>("log_spiking_neuron").set(false);
        conv1.parameter<bool>("inhibition").set(true);
        conv1.parameter<bool>("wta_infer").set(true);
        conv1.parameter<uint32_t>("epoch").set(sampling_size);
        conv1.parameter<float>("annealing").set(0.95f);
        conv1.parameter<float>("min_th").set(1.0f);
        conv1.parameter<float>("t_obj").set(t_obj);
        conv1.parameter<float>("lr_th").set(th_lr);
        conv1.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
        conv1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(_th, 0.1);
        conv1.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 0.1f);
        conv1.parameter<layer::ISampler3D>("sampler").set<layer::Sampler3DLandmark>();

        // Setup output
        auto &conv1_out = experiment.output<TimeObjectiveOutput>(conv1, t_obj, 2);
        conv1_out.add_postprocessing<process::FeatureMaps>("FMs3D/");
        conv1_out.add_postprocessing<process::SumPooling>(_spatial_pooling, _spatial_pooling);
        conv1_out.add_postprocessing<process::TemporalPooling>(_temporal_sum_pooling);
        conv1_out.add_postprocessing<process::FeatureScaling>();
        conv1_out.add_analysis<analysis::Activity>();
        conv1_out.add_analysis<analysis::Coherence>();
        conv1_out.add_analysis<analysis::Svm>("svm_probabilities");

//        auto &conv2 = experiment.push<layer::Convolution3D>(
//                3, 3, 3, 128, "", 1, 1, tmp_stride);
//        conv2.set_name("conv2");
//        conv2.parameter<bool>("draw").set(false);
//        conv2.parameter<bool>("draw_feature_maps").set(false);
//        conv2.parameter<bool>("save_weights").set(false);
//        conv2.parameter<bool>("save_random_start").set(false);
//        conv2.parameter<bool>("log_spiking_neuron").set(false);
//        conv2.parameter<bool>("inhibition").set(true);
//        conv2.parameter<bool>("wta_infer").set(true);
//        conv2.parameter<uint32_t>("epoch").set(sampling_size);
//        conv2.parameter<float>("annealing").set(0.95f);
//        conv2.parameter<float>("min_th").set(1.0f);
//        conv2.parameter<float>("t_obj").set(t_obj + 0.2);
//        conv2.parameter<float>("lr_th").set(th_lr);
//        conv2.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
//        conv2.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(_th - 4, 0.1);
//        conv2.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 0.1f);
//        conv2.parameter<layer::ISampler3D>("sampler").set<layer::Sampler3DRandom>();
//
//        auto &conv2_out = experiment.output<TimeObjectiveOutput>(conv2, t_obj, 2);
//        conv2_out.add_postprocessing<process::FeatureMaps>("FMs3D-layer2/");
//        conv2_out.add_postprocessing<process::SumPooling>(_spatial_pooling / 2, _spatial_pooling / 2);
//        conv2_out.add_postprocessing<process::TemporalPooling>(1);
//        conv2_out.add_postprocessing<process::FeatureScaling>();
//        conv2_out.add_analysis<analysis::Activity>();
//        conv2_out.add_analysis<analysis::Coherence>();
//        conv2_out.add_analysis<analysis::Svm>();



        layer::ISampler3D* sampler_ptr = conv1.get_sampler();

        // Add training and testing data with additional error handling
        int training_count = 0;
        for (auto& seq : training_sequences) {
            if (seq.frames.empty()) {
                experiment.log() << "Skipping empty training sequence" << std::endl;
                continue;
            }
            
            try {
                if (sampler_ptr != nullptr) {
                    sampler_ptr->add_sequence_bboxes(seq.bboxes);
                }
                // Use the CK_Plus_Input class instead of the nested class
                experiment.add_train<dataset::CK_Plus_Input>(seq, _frame_size_width, _frame_size_height);
                training_count++;
            } catch (const std::exception& e) {
                experiment.log() << "Error adding training sequence: " << e.what() << std::endl;
            }
        }
        experiment.log() << "Successfully added " << training_count << " training sequences" << std::endl;
        
        int testing_count = 0;
        for (auto& seq : testing_sequences) {
            if (seq.frames.empty()) {
                experiment.log() << "Skipping empty testing sequence" << std::endl;
                continue;
            }
            
            try {
                // Use the CK_Plus_Input class instead of the nested class
                experiment.add_test<dataset::CK_Plus_Input>(seq, _frame_size_width, _frame_size_height);
                testing_count++;
            } catch (const std::exception& e) {
                experiment.log() << "Error adding testing sequence: " << e.what() << std::endl;
            }
        }
        experiment.log() << "Successfully added " << testing_count << " testing sequences" << std::endl;

        experiment.log() << "Running experiment for fold " << fold << std::endl;
        experiment.run(10000);
    }
    
    return 0;
}
