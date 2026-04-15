#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "Input.h"
#include "Tensor.h"
#include "Spike.h"
#include "layer/Sampler3D.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include <filesystem>
#include <algorithm>

namespace dataset {

// Forward declaration
class CK_Plus_Input;

class CK_Plus {
public:
    struct ImageSequence {
        std::string subject = "";
        int ipostase = 0;
        int emotion = 0;
        std::vector<std::shared_ptr<Tensor<float>>> frames;
        std::vector<layer::BBox> bboxes;
    };
    
    // Emotion mapping constants
    enum Emotion {
        // entire dataset
        ANGER = 1,
        CONTEMPT = 2,
        DISGUST = 3,
        FEAR = 4,
        HAPPINESS = 5,
        SADNESS = 6,
        SURPRISE = 7,

        // cropped dataset
//        HAPPY = 1,
//        FEAR = 2,
//        SURPRISE = 3,
//        ANGER = 4,
//        DISGUST = 5,
//        SADNESS = 6

    };
    
    CK_Plus(const std::string& csv_path, const std::string& images_dir,
            int num_folds = 10, unsigned int random_seed = 42, 
            int image_width = 48, int image_height = 48, bool roi_detection = false, const std::string& landmarks_dir = "");
    ~CK_Plus();

    // Core dataset functions
    bool load();
    std::vector<ImageSequence> getSequences(int fold, int emotion);
    std::vector<ImageSequence> getTrainingSequences(int test_fold);
    std::vector<ImageSequence> getTestSequences(int test_fold);
    
    // Helper functions
    int getNumEmotions() const { return 6; }
    int getNumFolds() const { return m_num_folds; }
    unsigned int getRandomSeed() const { return m_random_seed; }
    std::map<int, std::map<int, int>> getEmotionCounts() const;
    std::string getEmotionName(int emotion) const;
    void printEmotionDistribution() const;
    
    // Data conversion functions
    std::shared_ptr<Tensor<float>> sequenceToTensor(const ImageSequence& seq);
    std::shared_ptr<Input> createInput(const ImageSequence& seq);
    std::shared_ptr<Spike> sequenceToSpike(const ImageSequence& seq); // Placeholder

private:
    // Configuration
    std::string m_csv_path;
    std::string m_images_dir;
    std::string m_landmarks_dir;
    int m_image_width;
    int m_image_height;
    int m_num_folds;
    bool m_roi_detection;
    unsigned int m_random_seed;
    
    // Data structure: fold -> emotion -> sequences
    std::map<int, std::map<int, std::vector<ImageSequence>>> m_data;

    // Internal helper methods
    std::shared_ptr<Tensor<float>> loadImage(const std::string& path, bool verbose = false);
    void populateSequenceData(ImageSequence& seq, bool verbose);
    void distributeSequences(std::vector<ImageSequence>& sequences);
    std::shared_ptr<Tensor<float>> convertMatToTensor(cv::Mat& image);
    bool initFaceTracking(const cv::Mat& first_gray, cv::CascadeClassifier& face_cascade,
                                   std::vector<cv::Point2f>& out_pts, std::vector<cv::Point2f>& out_bounding_box);
    bool updateFaceTracking(const cv::Mat& prev_gray, const cv::Mat& curr_gray,
                                     std::vector<cv::Point2f>& in_out_pts,
                                     std::vector<cv::Point2f>& in_out_bounding_box);
    std::shared_ptr<Tensor<float>> convertFullImageToTensor(const cv::Mat& img_gray);
    std::vector<std::shared_ptr<Tensor<float>>> loadTrackedSequence(const std::vector<std::filesystem::path>& file_paths,
            cv::CascadeClassifier& face_cascade, std::vector<layer::BBox>& out_bboxes);
};

// Standalone input class for CK+ dataset
class CK_Plus_Input : public Input {
public:
    CK_Plus_Input(const CK_Plus::ImageSequence& sequence, int image_width, int image_height);
    virtual ~CK_Plus_Input();

    // Implementation of Input interface
    virtual const Shape& shape() const override;
    virtual bool has_next() const override;
    virtual std::pair<std::string, Tensor<float>> next() override;
    virtual void reset() override;
    virtual void close() override;
    virtual std::string to_string() const override;

private:
    static Shape createShape(const CK_Plus::ImageSequence& sequence, int width, int height);
    void validateSequence();
    
    CK_Plus::ImageSequence _sequence;
    int _width;
    int _height;
    std::string _label;
    std::string _class_name;
    size_t _current_index;
    Shape _shape;
    bool _has_valid_data;
};

}
