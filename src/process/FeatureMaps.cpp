#include "process/FeatureMaps.h"
#include <cmath>
#include <filesystem>

using namespace process;

static RegisterClassParameter<FeatureMaps, ProcessFactory> _registerFeatureMaps("FeatureMaps");

FeatureMaps::FeatureMaps() : UniquePassProcess(_registerFeatureMaps),
                             _file_prefix(""), _cell_w(128), _cell_h(128),
                             _width(0), _height(0), _depth(0), _conv_depth(0), _no_samples(0)
{
    add_parameter("cell_w", _cell_w);
    add_parameter("cell_h", _cell_h);
}

FeatureMaps::FeatureMaps(const std::string& file_prefix) : FeatureMaps()
{
    _file_prefix = (std::filesystem::current_path() / file_prefix).string();
    parameter<size_t>("cell_w").set(128);
    parameter<size_t>("cell_h").set(128);
}

Shape FeatureMaps::compute_shape(const Shape &shape)
{
    _width = shape.dim(0);
    _height = shape.dim(1);
    _depth = shape.dim(2);
    _conv_depth = shape.number() > 3 ? shape.dim(3) : 1;

    // Visualization does not mutate the tensor's shape.
    return shape;
}

void FeatureMaps::process_train(const std::string &label, Tensor<float> &sample)
{
    // Append "_train" to distinguish the output file
    _no_samples++;
    _process(label + "_train", sample);
}

void FeatureMaps::process_test(const std::string &label, Tensor<float> &sample)
{
    // Append "_test" to distinguish the output file
    _no_samples++;
    _process(label + "_test", sample);
}

void FeatureMaps::_process(const std::string &label, const Tensor<float> &in) const
{
    if (true) {
//    if (label == "2_test" || label == "2_train") {
        // Use filesystem to build the path safely
        std::filesystem::path base_path = std::filesystem::current_path() / _file_prefix;
        std::filesystem::path folder_path = base_path / label;

        if (!std::filesystem::exists(folder_path)) {
            std::filesystem::create_directories(folder_path);
        }

        std::string final_path_prefix = (folder_path / std::to_string(_no_samples)).string();

        // Branch based on whether the tensor is 2D or 3D
        if (_conv_depth <= 1) {
            // --- 2D Logic ---
            size_t num_filters = _depth;

            int cols = std::ceil(std::sqrt(num_filters));
            int rows = std::ceil((float) num_filters / cols);

            cv::Mat grid(rows * _cell_h, cols * _cell_w, CV_8UC1, cv::Scalar(0));

            for (size_t f = 0; f < num_filters; f++) {
                cv::Mat filter_map(_height, _width, CV_32F);

                for (size_t i = 0; i < _height; i++) {
                    for (size_t j = 0; j < _width; j++) {
                        filter_map.at<float>(i, j) = in.at(j, i, f);
                    }
                }

                cv::Mat cell_8u;
                filter_map.convertTo(cell_8u, CV_8U, 255.0);

                cv::resize(cell_8u, cell_8u, cv::Size(_cell_w, _cell_h), 0, 0, cv::INTER_NEAREST);

                int r = f / cols;
                int c = f % cols;
                cv::Rect roi(c * _cell_w, r * _cell_h, _cell_w, _cell_h);
                cell_8u.copyTo(grid(roi));

                cv::putText(grid, "F" + std::to_string(f), cv::Point(roi.x + 2, roi.y + 12),
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255), 1);
            }

            cv::imwrite(final_path_prefix + "_2D_Grid.png", grid);
        } else {
            // --- 3D Logic ---
            size_t num_filters = _depth;
            size_t temporal_depth = _conv_depth;

            cv::Mat grid(num_filters * _cell_h, temporal_depth * _cell_w, CV_8UC1, cv::Scalar(0));

            for (size_t f = 0; f < num_filters; f++) {
                for (size_t d = 0; d < temporal_depth; d++) {
                    cv::Mat filter_slice(_height, _width, CV_32F);

                    for (size_t i = 0; i < _height; i++) {
                        for (size_t j = 0; j < _width; j++) {
                            filter_slice.at<float>(i, j) = in.at(j, i, f, d);
                        }
                    }

                    cv::Mat cell_8u;
                    filter_slice.convertTo(cell_8u, CV_8U, 255.0);

                    cv::resize(cell_8u, cell_8u, cv::Size(_cell_w, _cell_h), 0, 0, cv::INTER_NEAREST);

                    cv::Rect roi(d * _cell_w, f * _cell_h, _cell_w, _cell_h);
                    cell_8u.copyTo(grid(roi));

                    if (d == 0) {
                        cv::putText(grid, "Filter " + std::to_string(f), cv::Point(roi.x + 2, roi.y + 12),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255), 1);
                    }
                }
            }

            cv::imwrite(final_path_prefix + "_3D_Flow.png", grid);
        }
    }
}