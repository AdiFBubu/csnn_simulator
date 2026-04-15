#include "dataset/CK_Plus.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <iomanip> // For std::setw and std::setfill
#include <random>
#include <algorithm>

namespace dataset {

// Helper function to read the 68 landmarks and compute the bounding box
layer::BBox get_scaled_bbox_from_landmarks(
        const std::string& landmark_path,
        float orig_width = 640.0f,
        float orig_height = 490.0f,
        float target_width = 48.0f,
        float target_height = 48.0f)
{
    std::ifstream file(landmark_path);

    layer::BBox result = {0, static_cast<size_t>(target_width), 0, static_cast<size_t>(target_height), {}};

    if (!file.is_open()) {
        // Fallback to the whole frame if the landmark file is missing
        return result;
    }

    float x, y;
    float min_x = 10000.0f, max_x = 0.0f;
    float min_y = 10000.0f, max_y = 0.0f;
    std::vector<std::pair<float, float>> raw_points;

    // Read the (x,y) pairs (usually 68 points)
    while (file >> x >> y) {
        raw_points.emplace_back(x, y);
        if (x < min_x) min_x = x;
        if (x > max_x) max_x = x;
        if (y < min_y) min_y = y;
        if (y > max_y) max_y = y;
    }

    if (raw_points.empty()) return result;

    // Scale coordinates from original resolution to network resolution (48x48)
    float scale_x = target_width / orig_width;
    float scale_y = target_height / orig_height;

    result.x_min = static_cast<size_t>(std::max(0.0f, min_x * scale_x));
    result.x_max = static_cast<size_t>(std::min(target_width, max_x * scale_x));
    result.y_min = static_cast<size_t>(std::max(0.0f, min_y * scale_y));
    result.y_max = static_cast<size_t>(std::min(target_height, max_y * scale_y));

    // we also store the scaled landmarks in the BBox structure for potential use in landmark-based sampling
    for (const auto& pt : raw_points) {
        result.landmarks.push_back({pt.first * scale_x, pt.second * scale_y});
    }

    return result;
}

// Initialize static log level (0=none, 1=minimal, 2=verbose)
static const int LOG_LEVEL = 1;

// Helper macro for conditional logging
#define LOG_INFO(level, message) if (LOG_LEVEL >= level) { std::cout << message; }
#define LOG_ERROR(message) std::cerr << message;

CK_Plus::CK_Plus(const std::string& csv_path, const std::string& images_dir,
                 int num_folds, unsigned int random_seed,
                 int image_width, int image_height, bool roi_detection,  const std::string& landmarks_dir)
        : m_csv_path(csv_path), m_images_dir(images_dir), m_landmarks_dir(landmarks_dir),
          m_image_width(image_width), m_image_height(image_height),
          m_num_folds(num_folds), m_random_seed(random_seed), m_roi_detection(roi_detection),m_data() {
}

CK_Plus::~CK_Plus() {
}

bool CK_Plus::load() {
    std::ifstream file(m_csv_path);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open CSV file: " << m_csv_path << std::endl);
        return false;
    }

    LOG_INFO(1, "Loading dataset from " << m_csv_path << std::endl);
    std::vector<ImageSequence> all_sequences;

    std::string line;
    int line_count = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string subject, ipostase_str, emotion_str;

        std::getline(ss, subject, ',');
        std::getline(ss, ipostase_str, ',');
        std::getline(ss, emotion_str, ',');

        try {
            int ipostase = std::stoi(ipostase_str);
            int emotion = std::stoi(emotion_str);

            // Create and populate the image sequence
            ImageSequence seq;
            seq.subject = subject;
            seq.ipostase = ipostase;
            seq.emotion = emotion;

            // Use verbosity level 2 for detailed image loading logs
            populateSequenceData(seq, LOG_LEVEL >= 2);

            // Only add sequences that have frames
            if (!seq.frames.empty()) {
                all_sequences.push_back(seq);
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Error processing line " << line_count << ": " << e.what() << std::endl);
        }

        line_count++;
    }

    // Distribute sequences randomly but evenly across folds
    distributeSequences(all_sequences);

    return !all_sequences.empty();
}

void CK_Plus::distributeSequences(std::vector<ImageSequence>& sequences) {
    // Clear existing data
    m_data.clear();

    // Group sequences by emotion
    std::map<int, std::vector<ImageSequence>> sequences_by_emotion;
    for (auto& seq : sequences) {
        sequences_by_emotion[seq.emotion].push_back(seq);
    }

    std::mt19937 rng(m_random_seed);

    // For each emotion, randomly distribute sequences across folds
    for (auto& [emotion, emotion_sequences] : sequences_by_emotion) {
        // Shuffle sequences to randomize distribution
        std::shuffle(emotion_sequences.begin(), emotion_sequences.end(), rng);

        // Distribute sequences evenly across folds
        for (size_t i = 0; i < emotion_sequences.size(); i++) {
            int fold = (i % m_num_folds) + 1; // Folds are 1-indexed
            m_data[fold][emotion].push_back(emotion_sequences[i]);
        }
    }

    // Print distribution statistics
    if (LOG_LEVEL >= 1) {
        printEmotionDistribution();
    }
}

std::vector<CK_Plus::ImageSequence> CK_Plus::getSequences(int fold, int emotion) {
    if (m_data.find(fold) != m_data.end() && m_data[fold].find(emotion) != m_data[fold].end()) {
        return m_data[fold][emotion];
    }
    return {};
}

std::vector<CK_Plus::ImageSequence> CK_Plus::getTrainingSequences(int test_fold) {
    std::vector<ImageSequence> training_sequences;

    for (int fold = 1; fold <= getNumFolds(); fold++) {
        if (fold != test_fold) {
            for (int emotion = 1; emotion <= getNumEmotions(); emotion++) {
                auto sequences = getSequences(fold, emotion);
                training_sequences.insert(training_sequences.end(), sequences.begin(), sequences.end());
            }
        }
    }

    return training_sequences;
}

std::vector<CK_Plus::ImageSequence> CK_Plus::getTestSequences(int test_fold) {
    std::vector<ImageSequence> test_sequences;

    for (int emotion = 1; emotion <= getNumEmotions(); emotion++) {
        auto sequences = getSequences(test_fold, emotion);
        test_sequences.insert(test_sequences.end(), sequences.begin(), sequences.end());
    }

    return test_sequences;
}

std::shared_ptr<Tensor<float>> CK_Plus::loadImage(const std::string& path, bool verbose) {
    if (verbose) {
        LOG_INFO(2, "Loading image: " << path << std::endl);
    }

    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        LOG_ERROR("Failed to load image: " << path << std::endl);
        return nullptr;
    }

    // Resize if necessary
    if (image.rows != m_image_height || image.cols != m_image_width) {
        cv::resize(image, image, cv::Size(m_image_width, m_image_height));
    }

    // Create shape for the tensor - note the ordering
    std::vector<size_t> dims = {static_cast<size_t>(m_image_width),
                                static_cast<size_t>(m_image_height),
                                1, 1};
    Shape shape(dims);

    // Convert to Tensor using the Shape constructor
    auto tensor = std::make_shared<Tensor<float>>(shape);

    // Copy image data to tensor using at() method
    for (int y = 0; y < m_image_height; y++) {
        for (int x = 0; x < m_image_width; x++) {
            float pixel_value = static_cast<float>(image.at<uchar>(y, x)) / 255.0f;
            tensor->at(x, y, 0, 0) = pixel_value;
        }
    }

    if (verbose) {
        LOG_INFO(2, "  → Created 2D tensor [" << m_image_height << "×" << m_image_width
                                              << "] from image: " << std::filesystem::path(path).filename().string() << std::endl);
    }

    return tensor;
}

void CK_Plus::populateSequenceData(ImageSequence& seq, bool verbose) {

    // Format ipostase with leading zeros (3-digit format: 000, 001, etc.)
    std::ostringstream ss;
    ss << std::setw(3) << std::setfill('0') << seq.ipostase;
    std::string ipostase_str = ss.str();

    // Construct the directory path for this subject and ipostase with zero-padded folder name
    std::string subject_dir = m_images_dir + "/" + seq.subject + "/" + ipostase_str;

    std::string landmark_subj_dir = m_landmarks_dir + "/" + seq.subject + "/" + ipostase_str;

    if (verbose) {
        LOG_INFO(2, "Loading sequence from: " << subject_dir << std::endl);
    }

    try {
        // Collect all image paths first and sort them
        std::vector<std::filesystem::path> image_paths;

        // Iterate through all image files in the directory
        for (const auto& entry : std::filesystem::directory_iterator(subject_dir)) {
            if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg") {
                image_paths.push_back(entry.path());
            }
        }

        // Sort paths by filename
        std::sort(image_paths.begin(), image_paths.end());

        if (!m_roi_detection) {
            // Now load images in sorted order
            for (size_t i = 0; i < image_paths.size(); i++) {
                auto frame = loadImage(image_paths[i].string(), verbose);
                if (frame) {
                    seq.frames.push_back(frame);
                    if (!m_landmarks_dir.empty()) {
                        // LANDMARK PARSING ---
                        // Construct expected landmark filename: e.g., S011_001_00000001_landmarks.txt
                        std::ostringstream lm_ss;
                        lm_ss << landmark_subj_dir << "/"
                              << seq.subject << "_" << ipostase_str << "_"
                              << std::setw(8) << std::setfill('0') << (i + 1)
                              << "_landmarks.txt";

                        layer::BBox bbox = get_scaled_bbox_from_landmarks(
                                lm_ss.str(), 640.0f, 490.0f, static_cast<float>(m_image_width), static_cast<float>(m_image_height)
                        );

                        seq.bboxes.push_back(bbox);
                    }
                }
            }
        }
        else {

            cv::CascadeClassifier face_cascade;

            if (!face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")) {
                std::cerr << "Error loading Viola-Jones model" << std::endl;
                return;
            }

            seq.frames = loadTrackedSequence(image_paths, face_cascade, seq.bboxes);
        }
    }
    catch (const std::exception& e) {
        LOG_ERROR("Error loading image sequence: " << e.what() << " [" << subject_dir << "]" << std::endl);
    }

    if (verbose || seq.frames.empty()) {
        LOG_INFO(1, "Loaded " << seq.frames.size() << " frames for subject " << seq.subject
                              << ", ipostase " << seq.ipostase << std::endl);
    }
}

bool CK_Plus::initFaceTracking(const cv::Mat& first_gray,
                               cv::CascadeClassifier& face_cascade,
                               std::vector<cv::Point2f>& out_pts,
                               std::vector<cv::Point2f>& out_bounding_box) {
    std::vector<cv::Rect> faces;

    face_cascade.detectMultiScale(first_gray, faces, 1.1, 3, 0, cv::Size(30, 30));

    if (faces.empty()) {
        std::cerr << "Eroare: Nu s-a detectat nicio față în primul cadru!" << std::endl;
        return false;
    }

    cv::Rect roi = faces[0];

    out_bounding_box.clear();
    out_bounding_box.push_back(cv::Point2f(roi.x, roi.y));
    out_bounding_box.push_back(cv::Point2f(roi.x + roi.width, roi.y));
    out_bounding_box.push_back(cv::Point2f(roi.x + roi.width, roi.y + roi.height));
    out_bounding_box.push_back(cv::Point2f(roi.x, roi.y + roi.height));

    cv::Mat mask = cv::Mat::zeros(first_gray.size(), CV_8U);
    mask(roi) = 255;
    cv::goodFeaturesToTrack(first_gray, out_pts, 100, 0.01, 10, mask);

    return !out_pts.empty();
}

bool CK_Plus::updateFaceTracking(const cv::Mat& prev_gray, const cv::Mat& curr_gray,
                                 std::vector<cv::Point2f>& in_out_pts,
                                 std::vector<cv::Point2f>& in_out_bounding_box) {
    std::vector<cv::Point2f> curr_pts;
    std::vector<uchar> status;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(prev_gray, curr_gray, in_out_pts, curr_pts, status, err);

    std::vector<cv::Point2f> good_prev, good_curr;
    for (size_t i = 0; i < in_out_pts.size(); i++) {
        if (status[i] == 1) {
            good_prev.push_back(in_out_pts[i]);
            good_curr.push_back(curr_pts[i]);
        }
    }

    if (good_prev.size() < 3) {
        std::cerr << "Avertisment: S-au pierdut punctele de urmărire!" << std::endl;
        return false;
    }

    cv::Mat transform_matrix = cv::estimateAffinePartial2D(good_prev, good_curr);
    if (!transform_matrix.empty()) {
        std::vector<cv::Point2f> updated_box;
        cv::transform(in_out_bounding_box, updated_box, transform_matrix);
        in_out_bounding_box = updated_box;
    }

    in_out_pts = good_curr;
    return true;
}

std::shared_ptr<Tensor<float>> CK_Plus::convertFullImageToTensor(const cv::Mat& img_gray) {
    cv::Mat resized_img;

    // Resize the full image directly to the tensor dimensions (e.g., 48x48)
    cv::resize(img_gray, resized_img, cv::Size(m_image_width, m_image_height));

    std::vector<size_t> dims = {static_cast<size_t>(m_image_width),
                                static_cast<size_t>(m_image_height), 1, 1};
    Shape shape(dims);
    auto tensor = std::make_shared<Tensor<float>>(shape);

    for (int y = 0; y < m_image_height; y++) {
        for (int x = 0; x < m_image_width; x++) {
            float pixel_value = static_cast<float>(resized_img.at<uchar>(y, x)) / 255.0f;
            tensor->at(x, y, 0, 0) = pixel_value;
        }
    }

    return tensor;
}

std::vector<std::shared_ptr<Tensor<float>>> CK_Plus::loadTrackedSequence(
        const std::vector<std::filesystem::path>& file_paths,
        cv::CascadeClassifier& face_cascade,
        std::vector<layer::BBox>& out_bboxes) {

    cv::Mat prev_gray;
    std::vector<cv::Point2f> pts;
    std::vector<cv::Point2f> bounding_box;
    std::vector<std::shared_ptr<Tensor<float>>> frames;

    int i = 0;
    for (const auto& path : file_paths) {
        cv::Mat curr_gray = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
        if (curr_gray.empty()) continue;

        bool tracking_ok = false;

        if (i == 0) {
            tracking_ok = initFaceTracking(curr_gray, face_cascade, pts, bounding_box);
        } else {
            tracking_ok = updateFaceTracking(prev_gray, curr_gray, pts, bounding_box);
        }

        // Fallback if tracking is lost
        if (!tracking_ok) {
            tracking_ok = initFaceTracking(curr_gray, face_cascade, pts, bounding_box);
        }

        if (tracking_ok) {
            cv::Rect face_roi = cv::boundingRect(bounding_box);

            float margin_w = face_roi.width * 0.15f;
            float margin_h = face_roi.height * 0.10f;

            cv::Rect tight_roi(
                    face_roi.x + margin_w,
                    face_roi.y + margin_h,
                    face_roi.width - 2 * margin_w,
                    face_roi.height - 1.5 * margin_h
            );

            // Ensure ROI is within original image boundaries
            tight_roi &= cv::Rect(0, 0, curr_gray.cols, curr_gray.rows);

            if (tight_roi.area() > 0) {
                // 1. Calculate how much the image will be scaled down to fit the Tensor
                float scale_x = static_cast<float>(m_image_width) / curr_gray.cols;
                float scale_y = static_cast<float>(m_image_height) / curr_gray.rows;

                // 2. Scale the bounding box coordinates to match the Tensor's dimensions
                layer::BBox bbox = {
                        static_cast<size_t>(tight_roi.x * scale_x),
                        static_cast<size_t>((tight_roi.x + tight_roi.width) * scale_x),
                        static_cast<size_t>(tight_roi.y * scale_y),
                        static_cast<size_t>((tight_roi.y + tight_roi.height) * scale_y)
                };

                out_bboxes.push_back(bbox);

                // 3. Convert the FULL original frame to a tensor (no cropping)
                auto tensor = convertFullImageToTensor(curr_gray);
                if (tensor) {
                    frames.push_back(tensor);
                }
            } else {
                std::cerr << "Warning: Invalid ROI area calculated." << std::endl;
            }
        }

        prev_gray = curr_gray.clone();
        i++;
    }

    return frames;
}

std::shared_ptr<Tensor<float>> CK_Plus::sequenceToTensor(const ImageSequence& seq) {
    if (seq.frames.empty()) {
        return nullptr;
    }

    int depth = seq.frames.size();
    int width = m_image_width;
    int height = m_image_height;


    std::string emotion_name;
    switch(seq.emotion) {
        case 1: emotion_name = "Happy"; break;
        case 2: emotion_name = "Fear"; break;
        case 3: emotion_name = "Surprise"; break;
        case 4: emotion_name = "Anger"; break;
        case 5: emotion_name = "Disgust"; break;
        case 6: emotion_name = "Sadness"; break;
        default: emotion_name = "Unknown"; break;
    }

    LOG_INFO(1, "Converting sequence to 3D tensor: Subject=" << seq.subject
                                                             << ", Ipostase=" << seq.ipostase
                                                             << ", Emotion=" << seq.emotion << " (" << emotion_name << ")"
                                                             << ", Frames=" << depth << std::endl);

    std::vector<size_t> dims = {static_cast<size_t>(width),
                                static_cast<size_t>(height),
                                1,
                                static_cast<size_t>(depth)};
    Shape shape(dims);
    auto tensor = std::make_shared<Tensor<float>>(shape);


    for (int z = 0; z < depth; z++) {
        auto& frame = seq.frames[z];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float value = frame->at(x, y, 0, 0);
                tensor->at(x, y, 0, z) = value;
            }
        }

        LOG_INFO(2, "  Added frame " << z << " to tensor at depth position " << z << std::endl);
    }

    LOG_INFO(2, "Created 3D tensor for emotion " << emotion_name
                                                 << " with dimensions [" << height << "×" << width << "×" << depth << "×1]" << std::endl);

    return tensor;
}

std::shared_ptr<Tensor<float>> CK_Plus::convertMatToTensor(cv::Mat& image) {
    if (image.rows != m_image_height || image.cols != m_image_width) {
        cv::resize(image, image, cv::Size(m_image_width, m_image_height));
    }

    std::vector<size_t> dims = {static_cast<size_t>(m_image_width),
                                static_cast<size_t>(m_image_height),
                                1, 1};
    Shape shape(dims);
    auto tensor = std::make_shared<Tensor<float>>(shape);

    for (int y = 0; y < m_image_height; y++) {
        for (int x = 0; x < m_image_width; x++) {
            float pixel_value = static_cast<float>(image.at<uchar>(y, x)) / 255.0f;
            tensor->at(x, y, 0, 0) = pixel_value;
        }
    }
    return tensor;
}

std::string CK_Plus::getEmotionName(int emotion) const {
    switch(emotion) {

        case 1: return "Anger";
        case 2: return "Contempt";
        case 3: return "Disgust";
        case 4: return "Fear";
        case 5: return "Happiness";
        case 6: return "Sadness";
        case 7: return "Surprise";
        default: return "Unknown";

//        cropped dataset
//        case 1: return "Happy";
//        case 2: return "Fear";
//        case 3: return "Surprise";
//        case 4: return "Anger";
//        case 5: return "Disgust";
//        case 6: return "Sadness";
//        default: return "Unknown";
    }
}

std::map<int, std::map<int, int>> CK_Plus::getEmotionCounts() const {
    std::map<int, std::map<int, int>> counts;

    // Initialize all counters to zero for all folds and emotions
    for (int fold = 1; fold <= m_num_folds; fold++) {
        for (int emotion = 1; emotion <= getNumEmotions(); emotion++) {
            counts[fold][emotion] = 0;
        }
    }

    // Count sequences for each emotion in each fold
    for (const auto& [fold, emotions] : m_data) {
        for (const auto& [emotion, sequences] : emotions) {
            counts[fold][emotion] = sequences.size();
        }
    }

    return counts;
}

void CK_Plus::printEmotionDistribution() const {
    auto counts = getEmotionCounts();

    // Calculate totals - keep this for potential data validation
    std::map<int, int> emotion_totals;
    std::map<int, int> fold_totals;
    int grand_total = 0;

    for (int fold = 1; fold <= m_num_folds; fold++) {
        for (int emotion = 1; emotion <= getNumEmotions(); emotion++) {
            int count = counts[fold][emotion];
            emotion_totals[emotion] += count;
            fold_totals[fold] += count;
            grand_total += count;
        }
    }

    // Remove the detailed table output - just print a summary with seed info
    std::cout << "Dataset loaded: " << grand_total << " sequences across "
              << m_num_folds << " folds (using seed: " << m_random_seed << ")" << std::endl;
}

std::shared_ptr<Spike> CK_Plus::sequenceToSpike(const ImageSequence& seq) {
    // Suppress unused parameter warning
    (void)seq;

    // Since Spike has no default constructor, we need to provide parameters
    // For now, return nullptr until you implement proper conversion to Spike
    return nullptr;
}

// Implementation of CkPlusInput class
CK_Plus_Input::CK_Plus_Input(const CK_Plus::ImageSequence& sequence, int image_width, int image_height)
        : Input(),
          _sequence(sequence),
          _width(image_width),
          _height(image_height),
          _label(std::to_string(_sequence.emotion)),
          _class_name(std::to_string(_sequence.emotion)),
          _current_index(0),
          _shape(createShape(sequence, image_width, image_height)),
          _has_valid_data(false) {

    // Validate before allowing next() to be called
    try {
        validateSequence();
        _has_valid_data = true;
    } catch (const std::exception& e) {
        LOG_ERROR("Error in constructor: " << e.what() << std::endl);
        _has_valid_data = false;
    }
}

CK_Plus_Input::~CK_Plus_Input() {}

const Shape& CK_Plus_Input::shape() const {
    return _shape;
}

bool CK_Plus_Input::has_next() const {
    // Only return true if we have valid data AND we haven't provided it yet
    return _has_valid_data && _current_index < 1;
}

std::pair<std::string, Tensor<float>> CK_Plus_Input::next() {
    if (!has_next()) {
        throw std::runtime_error("No more data in CK_Plus_Input or invalid data");
    }

    try {
        Tensor<float> result(_shape);
        result.fill(0.0f); // Initialize with zeros to avoid undefined values

        // Copy frame data to tensor
        for (size_t z = 0; z < _sequence.frames.size(); z++) {
            auto& frame = _sequence.frames[z];
            if (!frame) {
                LOG_ERROR("Warning: Null frame at position " << z
                                                             << " in sequence " << _sequence.subject
                                                             << ", emotion " << _sequence.emotion << std::endl);
                continue;
            }

            for (int y = 0; y < _height; y++) {
                for (int x = 0; x < _width; x++) {
                    try {
                        result.at(y, x, 0, z) = frame->at(y, x, 0, 0);
                    } catch (const std::exception& e) {
                        LOG_ERROR("Error accessing frame data at position [" << y << "," << x << "," << z
                                                                             << "]: " << e.what() << std::endl);
                        // Continue with next pixel instead of failing entirely
                    }
                }
            }
        }

        _current_index++;
        return std::make_pair(_label, result);
    }
    catch (const std::exception& e) {
        LOG_ERROR("Error creating tensor in next(): " << e.what() << std::endl);
        // Return empty tensor with label to avoid crashing
        Tensor<float> empty_tensor(_shape);
        empty_tensor.fill(0.0f);
        return std::make_pair(_label, empty_tensor);
    }
}

void CK_Plus_Input::reset() {
    _current_index = 0;
}

void CK_Plus_Input::close() {
    // Nothing to close
}

std::string CK_Plus_Input::to_string() const {
    return "Emotion=" + _label +
           ", Frames=" + std::to_string(_sequence.frames.size());
}

// Factory method to create input from sequence
std::shared_ptr<Input> CK_Plus::createInput(const ImageSequence& seq) {
    return std::make_shared<CK_Plus_Input>(seq, m_image_width, m_image_height);
}

Shape CK_Plus_Input::createShape(const CK_Plus::ImageSequence& sequence, int width, int height) {
    // Initialize shape with explicit size_t values to avoid unexpected conversions
    size_t h = static_cast<size_t>(height);
    size_t w = static_cast<size_t>(width);
    size_t d = sequence.frames.empty() ? 1 : static_cast<size_t>(sequence.frames.size());
    size_t c = 1; // channels

    return Shape({w, h, c, d});
}

void CK_Plus_Input::validateSequence() {
    // More thorough validation
    if (_sequence.frames.empty()) {
        throw std::runtime_error("Sequence has no frames");
    }

    // Make sure all dimensions are valid (non-zero)
    if (_width <= 0 || _height <= 0) {
        throw std::runtime_error("Invalid dimensions: width=" + std::to_string(_width) +
                                 ", height=" + std::to_string(_height));
    }

    // Validate that all frames are non-null and have correct dimensions
    for (size_t i = 0; i < _sequence.frames.size(); i++) {
        if (!_sequence.frames[i]) {
            throw std::runtime_error("Frame " + std::to_string(i) + " is null");
        }

        // Check if frame dimensions match expected dimensions
        const auto& frame = _sequence.frames[i];
        if (frame->shape().dim(0) != static_cast<size_t>(_height) ||
            frame->shape().dim(1) != static_cast<size_t>(_width)) {
            throw std::runtime_error("Frame " + std::to_string(i) + " has invalid dimensions: " +
                                     frame->shape().to_string());
        }
    }
}

} // namespace dataset
