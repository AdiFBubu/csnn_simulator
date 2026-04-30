#include "Experiment.h"
#include "dataset/Video.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
#include "layer/Convolution3D.h"
#include "layer/Pooling.h"
#include "Distribution.h"
#include "execution/ProcessExecution.h"
#include "execution/FusedExecution.h"
#include "analysis/Svm.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "process/Input.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "layer/Stream.h"
#include "process/SimplePreprocessing.h"
#include "process/CompositeChannels.h"
#include "process/OnOffFilter.h"
#include "process/OnOffTempFilter.h"
#include "process/EarlyFusion.h"
#include "process/SeparateSign.h"
#include "process/ResizeInput.h"
#include "tool/AutoFrameNumberSelector.h"
#include "process/MotionGridV1.h"
#include "dataset/CK_Plus.h"


int main(int argc, char **argv)
{
    // for (int _repeat = 1; _repeat < 4; _repeat++)
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
        size_t _frame_size_width = 48;
        size_t _frame_size_height = 48;

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

            Experiment<ProcessExecution> experiment(argc, argv, _dataset, false, true);
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

            int training_count = 0;
            for (auto& seq : training_sequences) {
                if (seq.frames.empty()) {
                    experiment.log() << "Skipping empty training sequence" << std::endl;
                    continue;
                }

                try {
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

            auto &svm = experiment.push<layer::Stream>(1, 1, 1, 1);

            // add another SVM step to classify the fused result.
            auto &svm_out = experiment.output<NoOutputConversion>(svm);
            svm_out.add_postprocessing<process::SumPooling>(10, 10);
//            svm_out.add_postprocessing<process::TemporalPooling>(2);
            svm_out.add_analysis<analysis::Svm>();

            experiment.run(10000);
        }
    }
}