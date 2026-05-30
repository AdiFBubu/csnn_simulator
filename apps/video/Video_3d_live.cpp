#include "Experiment.h"
#include "dataset/CK_Plus.h"
#include "dataset/BackendBatch.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
//#include "layer/FaceElypsesCutout3D.h"
//#include "layer/FaceElypsesCutout3D.h"
#include "layer/Convolution3D.h"
#include "layer/Pooling.h"
#include "Distribution.h"
#include "execution/TestingExecution.h"
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
#include "process/DistributionAnalysis.h"
#include "process/ReorderSpikes.h"
#include "layer/Sampler3D.h"
#include "process/FeatureMaps.h"
#include "execution/TestingSparseExecution.h"
#include <deque>
#include <sstream>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdint>
#include <random>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

#ifdef _WIN32
using sock_t = SOCKET;
#else
using sock_t = int;
#endif

static bool read_exact(sock_t sock, void *buf, size_t len) {
    size_t received = 0;
    char *p = reinterpret_cast<char*>(buf);
    while (received < len) {
        int r = recv(sock, p + received, static_cast<int>(len - received), 0);
        if (r <= 0) return false;
        received += static_cast<size_t>(r);
    }
    return true;
}

static bool send_all(sock_t sock, const char *data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        int s = send(sock, data + sent, static_cast<int>(len - sent), 0);
        if (s <= 0) return false;
        sent += static_cast<size_t>(s);
    }
    return true;
}

int main(int argc, char **argv)
{
    // Parse command line arguments with new parameters
    size_t _filter_width = (argc > 1) ? atoi(argv[1]) : 5;
    size_t _filter_height = (argc > 2) ? atoi(argv[2]) : 5;
    size_t _filter_depth = (argc > 3) ? atoi(argv[3]) : 3;
    size_t _temporal_sum_pooling = (argc > 4) ? atoi(argv[4]) : 2;

    // Keep epochs and threshold unchanged
    int _epochs = (argc > 5) ? atoi(argv[5]) : 800;
    float _th = 4.0;

    // Add random seed parameter
    unsigned int random_seed = (argc > 6) ? atoi(argv[6]) : 42;

    // Add spatial pooling parameter
    size_t _spatial_pooling = (argc > 7) ? atoi(argv[7]) : 4;

    // Print parameters
    std::cout << "Random seed: " << random_seed << std::endl;
    std::cout << "Spatial pooling: " << _spatial_pooling << std::endl;

    int num_folds = 10;

    // Video frame dimensions
    size_t _frame_size_width = 48;
    size_t _frame_size_height = 48;

    time_t start_time;
    time(&start_time);

    bool tcp_mode = false;
    int tcp_port = 0;
    std::string model_dir = "";
    size_t batch_size = _filter_depth; // default batch matches temporal depth unless overridden
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == std::string("--tcp") && i + 1 < argc) { tcp_mode = true; tcp_port = atoi(argv[i + 1]); }
        if (a == std::string("--port") && i + 1 < argc) { tcp_mode = true; tcp_port = atoi(argv[i + 1]); }
        if (a == std::string("--model") && i + 1 < argc) {
            model_dir = argv[i + 1];
        }
        if (a == std::string("--batch-size") && i + 1 < argc) {
            batch_size = static_cast<size_t>(atoi(argv[i + 1]));
        }
    }

    if (tcp_mode) {
#ifdef _WIN32
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2,2), &wsaData) != 0) {
            std::cerr << "WSAStartup failed" << std::endl;
            return 1;
        }
#endif

        // create listening socket
        sock_t listen_sock;
#ifdef _WIN32
        listen_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (listen_sock == INVALID_SOCKET) { std::cerr << "Failed to create socket" << std::endl; WSACleanup(); return 1; }
#else
        listen_sock = socket(AF_INET, SOCK_STREAM, 0);
        if (listen_sock < 0) { std::cerr << "Failed to create socket" << std::endl; return 1; }
#endif

        int opt = 1;
        setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

        struct sockaddr_in servaddr;
        memset(&servaddr, 0, sizeof(servaddr));
        servaddr.sin_family = AF_INET;
        servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
        servaddr.sin_port = htons((uint16_t)tcp_port);

        if (bind(listen_sock, (struct sockaddr*)&servaddr, sizeof(servaddr)) < 0) {
            std::cerr << "Bind failed" << std::endl;
#ifdef _WIN32
            closesocket(listen_sock);
            WSACleanup();
#else
            close(listen_sock);
#endif
            return 1;
        }

        if (listen(listen_sock, 1) < 0) {
            std::cerr << "Listen failed" << std::endl;
#ifdef _WIN32
            closesocket(listen_sock);
            WSACleanup();
#else
            close(listen_sock);
#endif
            return 1;
        }

        std::cout << "TCP server listening on port " << tcp_port << std::endl;

        std::vector<cv::Mat> buffer_vec;

        while (true) {
            struct sockaddr_in clientaddr;
            socklen_t clientlen = sizeof(clientaddr);
            sock_t client_sock = accept(listen_sock, (struct sockaddr*)&clientaddr, &clientlen);
#ifdef _WIN32
            if (client_sock == INVALID_SOCKET) { std::cerr << "Accept failed" << std::endl; break; }
#else
            if (client_sock < 0) { std::cerr << "Accept failed" << std::endl; break; }
#endif

            std::cout << "Client connected" << std::endl;

            while (true) {
                unsigned char hdr[4];
                if (!read_exact(client_sock, hdr, 4)) break;
                uint32_t len = (uint32_t(hdr[0]) << 24) | (uint32_t(hdr[1]) << 16) | (uint32_t(hdr[2]) << 8) | uint32_t(hdr[3]);
                if (len == 0) continue;
                std::vector<uchar> imgBuf(len);
                if (!read_exact(client_sock, imgBuf.data(), len)) break;

                cv::Mat frame = cv::imdecode(imgBuf, cv::IMREAD_COLOR);
                if (frame.empty()) {
                    std::cerr << "Failed to decode image frame" << std::endl;
                    continue;
                }

                cv::Mat gray;
                cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

                // Do NOT resize / normalize here – let BackendBatch handle detection + resize
                buffer_vec.push_back(gray.clone());

                if (buffer_vec.size() < batch_size) {
                    std::string msg = "{\"status\":\"buffering\"}\n";
                    send_all(client_sock, msg.c_str(), msg.size());
                    continue;
                }

                // =========================================================================
                // INIȚIALIZARE EXPERIMENT - Acum se creează o instanță nouă la fiecare batch
                // =========================================================================
                try {
                    // Adăugăm un pointer de timp la instanță pentru loguri unice (opțional, dar recomandat)
                    std::string _dataset = "TCP_" + std::to_string(time(nullptr)) + "_3D_" + std::to_string(_filter_width) + "x" + std::to_string(_filter_height) + "x" + std::to_string(_filter_depth);
                    Experiment<TestingSparseExecution> experiment(argc, argv, _dataset);

                    // build the same preprocessing + layer as live
                    experiment.push<process::DefaultOnOffFilter>(7, 0.1, 1.0);
                    // experiment.push<process::FeatureMaps>("Debug_Preprocess/");
                    experiment.push<process::MaxScaling>();
                    experiment.push<LatencyCoding>();

                    // experiment.push<process::DistributionAnalysis>();

                    float t_obj = 0.80;
                    float th_lr = 0.09f;
                    float w_lr = 0.009f;

                    size_t filter_number = 64;
                    size_t tmp_stride = 1;
                    size_t sampling_size = _epochs;

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
                    conv1.parameter<layer::ISampler3D>("sampler").set<layer::Sampler3DLandmark>();

                    auto &conv1_out = experiment.output<TimeObjectiveOutput>(conv1, t_obj, 1);
                    conv1_out.add_postprocessing<process::SumPooling>(_spatial_pooling, _spatial_pooling);
                    conv1_out.add_postprocessing<process::TemporalPooling>(_temporal_sum_pooling);
                    conv1_out.add_postprocessing<process::FeatureScaling>();
                    conv1_out.add_analysis<analysis::Activity>();
                    conv1_out.add_analysis<analysis::Coherence>();
                    conv1_out.add_analysis<analysis::Svm>("svm_probabilities");

                    // experiment.push<process::DistributionAnalysis>();
                    // experiment.push<process::ReorderSpikes>(t_obj, 0.1);
                    // experiment.push<process::DistributionAnalysis>();

                    // load pretrained params if available
                    // for (size_t i = 0; i < experiment.process_number(); i++) { ... }

                    layer::ISampler3D* sampler_ptr = conv1.get_sampler();

                    // Build a backend batch input from the collected frames and run inference once
                    experiment.add_test<dataset::BackendBatch_Input>(buffer_vec, _frame_size_width, _frame_size_height);
                    experiment.log() << "Added backend batch (frames=" << buffer_vec.size() << ") to test set" << std::endl;

                    // If sampler expects per-frame BBoxes, provide them from the newly created BackendBatch_Input
                    const auto &tests = experiment.test_data();
                    if (!tests.empty() && sampler_ptr != nullptr) {
                        Input* last = tests.back();
                        auto* bb = dynamic_cast<dataset::BackendBatch_Input*>(last);
                        if (bb != nullptr) {
                            sampler_ptr->add_sequence_bboxes(bb->bboxes());
                        }
                    }

                    experiment.run(10000);

                    // extract probabilities from SVM analysis (if exists) and send as JSON
                    std::map<std::string, double> probs;

//// Generator de numere aleatorii între 0.0 și 1.0
//                    std::random_device rd;
//                    std::mt19937 gen(rd());
//                    std::uniform_real_distribution<> dis(0.0, 1.0);
//
//// Definirea celor 7 etichete (labels)
//                    std::vector<std::string> labels = {
//                            "anger", "disgust", "fear", "happiness", "sadness", "surprise", "contempt"
//                    };
//
//                    for (const auto& label : labels) {
//                        probs[label] = dis(gen);
//                    }

                    for (Analysis* a : conv1_out.analysis()) {
                        analysis::Svm* svm_a = dynamic_cast<analysis::Svm*>(a);
                        if (svm_a) {
                            probs = svm_a->get_probabilities();
                            break;
                        }
                    }

                    if (!probs.empty()) {
                        std::ostringstream ss;
                        ss << "{\"status\":\"ok\",\"probabilities\":{";
                        bool first = true;
                        for (const auto &p : probs) {
                            if (!first) ss << ",";
                            // label as string, probability as numeric
                            ss << "\"" << p.first << "\":" << p.second;
                            first = false;
                        }
                        ss << "}}\n";
                        std::string msg = ss.str();
                        send_all(client_sock, msg.c_str(), msg.size());
                    } else {
                        std::string ok = "{\"status\":\"ok\"}\n";
                        send_all(client_sock, ok.c_str(), ok.size());
                    }

                } catch (const std::exception &e) {
                    std::cerr << "Error during backend inference: " << e.what() << std::endl;
                    // Am fixat ușor și structura JSON-ului de eroare ca să fie complet validă
                    std::string err = "{\"status\":\"error\",\"message\":\"" + std::string(e.what()) + "\"}\n";
                    send_all(client_sock, err.c_str(), err.size());
                }

                // clear the buffer for the next batch
                buffer_vec.clear();
            }

#ifdef _WIN32
            closesocket(client_sock);
#else
            close(client_sock);
#endif
            std::cout << "Client disconnected" << std::endl;
        }

#ifdef _WIN32
        closesocket(listen_sock);
        WSACleanup();
#else
        close(listen_sock);
#endif

        return 0;
    }
}