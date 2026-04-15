#include <opencv2/core.hpp>      // Structuri de bază (Mat, Rect, Point)
#include <opencv2/imgproc.hpp>   // cvtColor, goodFeaturesToTrack (Min-Eigenvalue), desenare
#include <opencv2/highgui.hpp>   // imshow, waitKey (pentru vizualizare)

// --- CELE DOUĂ IMPORTANTE PE CARE TREBUIE SĂ LE ADAUGI ---
#include <opencv2/objdetect.hpp> // Pasul 1: Viola-Jones (CascadeClassifier)
#include <opencv2/calib3d.hpp>   // Pasul 4: estimateAffinePartial2D (Geometric Transform)

// --- PENTRU URMĂRIRE ---
#include <opencv2/video.hpp>     // Pasul 3: calcOpticalFlowPyrLK (KLT Tracking)
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

vector<Mat> load_frames_from_folder(const string& folder_path) {
    vector<Mat> frames;
    vector<string> file_paths;

    // 1. Verificăm dacă folderul există
    if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) {
        cerr << "Eroare: Folderul nu există sau calea este greșită: " << folder_path << endl;
        return frames;
    }

    // 2. Colectăm toate căile fișierelor din folder
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        // În setul CK+ imaginile sunt de obicei .png
        if (entry.path().extension() == ".png") {
            file_paths.push_back(entry.path().string());
        }
    }

    // 3. Sortăm alfabetic (CRITIC pentru KLT!)
    // Fără sortare, imaginile ar putea fi încărcate aleatoriu (ex: cadrul 10 înainte de 2),
    // iar fluxul optic (mișcarea feței) nu ar mai avea sens.
    sort(file_paths.begin(), file_paths.end());

    if (file_paths.empty()) {
        cerr << "Avertisment: Nu am găsit nicio imagine .png în folder." << endl;
        return frames;
    }

    // 4. Citim fiecare imagine și o adăugăm în vector
    for (const auto& path : file_paths) {
        // Încărcăm direct în format Grayscale (Alb-Negru) pentru eficiență
        Mat img = imread(path, IMREAD_GRAYSCALE);

        if (img.empty()) {
            cerr << "Eroare la citirea imaginii: " << path << endl;
            continue;
        }

        frames.push_back(img);
    }

    cout << "Succes: Am încărcat " << frames.size() << " cadre în ordine temporală." << endl;
    return frames;
}

int main() {
    // 1. Încărcăm modelul Viola-Jones (Haar Cascade)
    CascadeClassifier face_cascade;
    // Asigură-te că acest fișier xml există în folderul tău (vine cu OpenCV)
    if (!face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")) {
        cerr << "Eroare la încărcarea modelului Viola-Jones!" << endl;
        return -1;
    }

    string folder_path = "/mnt/c/info/anul3/licenta/project-csnn/csnn_simulator/dataset/CK+/images/S010/001";

    vector<Mat> frames = load_frames_from_folder(folder_path);

    if (frames.empty()) {
        cerr << "Nu există imagini de procesat." << endl;
        return -1;
    }

    string output_folder = "faces";
    if (!fs::exists(output_folder)) {
        if (fs::create_directory(output_folder)) {
            cout << "Folderul '" << output_folder << "' a fost creat." << endl;
        }
    }

    Mat prev_gray;
    vector<Point2f> prev_pts;
    vector<Point2f> bounding_box; // Cele 4 colțuri ale feței

    for (size_t i = 0; i < frames.size(); i++) {
        Mat curr_frame = frames[i].clone();
        Mat curr_gray = curr_frame.clone();

        if (i == 0) {
            // ==========================================
            // CADRUL 1: INIȚIALIZARE (Pașii 1 și 2)
            // ==========================================

            // PASUL 1: Viola-Jones Face Detection
            vector<Rect> faces;
            face_cascade.detectMultiScale(curr_gray, faces, 1.1, 3, 0, Size(30, 30));

            if (faces.empty()) {
                cerr << "Nu s-a detectat nicio față în primul cadru!" << endl;
                return -1;
            }

            Rect roi = faces[0]; // Luăm prima față găsită

            // Definim poligonul inițial (cele 4 colțuri ale ROI-ului)
            bounding_box.push_back(Point2f(roi.x, roi.y));                           // Stânga-Sus
            bounding_box.push_back(Point2f(roi.x + roi.width, roi.y));               // Dreapta-Sus
            bounding_box.push_back(Point2f(roi.x + roi.width, roi.y + roi.height));  // Dreapta-Jos
            bounding_box.push_back(Point2f(roi.x, roi.y + roi.height));              // Stânga-Jos

            // PASUL 2: Găsirea punctelor "ancoră" (Min-Eigenvalue / Shi-Tomasi)
            // Creăm o mască neagră, și facem albă doar zona feței, ca să căutăm puncte doar acolo
            Mat mask = Mat::zeros(curr_gray.size(), CV_8U);
            mask(roi) = 255;

            // goodFeaturesToTrack folosește Min-Eigenvalue by default
            goodFeaturesToTrack(curr_gray, prev_pts, 100, 0.01, 10, mask);

            // Salvăm cadrul curent pentru a-l compara cu următorul
            prev_gray = curr_gray.clone();

            // (Opțional) Desenăm ROI-ul inițial pentru vizualizare
            rectangle(curr_frame, roi, Scalar(255, 0, 0), 2);
        }
        else {
            // ==========================================
            // CADRELE 2-10: URMĂRIRE (Pașii 3 și 4)
            // ==========================================

            // PASUL 3: Urmărirea (Algoritmul KLT)
            vector<Point2f> curr_pts;
            vector<uchar> status; // 1 dacă punctul a fost găsit, 0 dacă s-a pierdut
            vector<float> err;

            // calcOpticalFlowPyrLK = Kanade-Lucas-Tomasi
            calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, curr_pts, status, err);

            // Filtrăm punctele bune (le păstrăm doar pe cele găsite, status == 1)
            vector<Point2f> good_prev_pts, good_curr_pts;
            for (size_t j = 0; j < prev_pts.size(); j++) {
                if (status[j] == 1) {
                    good_prev_pts.push_back(prev_pts[j]);
                    good_curr_pts.push_back(curr_pts[j]);
                }
            }

            // PASUL 4: Actualizarea cutiei (Geometric Transform)
            if (good_prev_pts.size() >= 3) { // Avem nevoie de minim 3 puncte pentru transformare afină
                // Calculăm "matematica mișcării" (Translație, Rotație, Scalare)
                Mat transform_matrix = estimateAffinePartial2D(good_prev_pts, good_curr_pts);

                if (!transform_matrix.empty()) {
                    // Aplicăm transformarea asupra celor 4 colțuri ale cutiei feței
                    vector<Point2f> updated_box;
                    transform(bounding_box, updated_box, transform_matrix);
                    bounding_box = updated_box; // Actualizăm ROI-ul pentru viitor
                }
            } else {
                cout << "S-au pierdut prea multe puncte. Ar trebui reluat Pasul 1 (Viola-Jones)." << endl;
                // Aici ar interveni un mecanism de "Fallback" în viața reală
            }

            // Desenăm punctele urmărite (micro-mișcarea)
            for (size_t j = 0; j < good_curr_pts.size(); j++) {
                circle(curr_frame, good_curr_pts[j], 3, Scalar(0, 255, 0), -1);
            }

            // Desenăm noua cutie (macro-mișcarea)
            for (int j = 0; j < 4; j++) {
                line(curr_frame, bounding_box[j], bounding_box[(j + 1) % 4], Scalar(0, 0, 255), 2);
            }

            // Pregătim variabilele pentru cadrul următor
            prev_gray = curr_gray.clone();
            prev_pts = good_curr_pts;
        }

        // 1. Transformăm punctele poligonului într-un dreptunghi (AABB - Axis Aligned Bounding Box)
        // Deoarece bounding_box este vector<Point2f>, folosim boundingRect:
        Rect face_roi = boundingRect(bounding_box);

        float margin_w = face_roi.width * 0.15;
        float margin_h = face_roi.height * 0.10;

        Rect tight_roi(
                face_roi.x + margin_w,
                face_roi.y + margin_h,
                face_roi.width - 2 * margin_w,
                face_roi.height - 1.5 * margin_h // tăiem mai mult de jos (gâtul)
        );

        tight_roi &= Rect(0, 0, curr_gray.cols, curr_gray.rows);

        // 2. IMPORTANT: Verificăm ca dreptunghiul să nu iasă din marginile imaginii
        // (Altfel programul va crăpa la decupare)
//        face_roi &= Rect(0, 0, curr_gray.cols, curr_gray.rows);

        if (face_roi.width > 0 && face_roi.height > 0) {
            // 3. Decupăm fața din cadrul curent
//            Mat face_crop = curr_gray(face_roi);
            Mat face_crop = curr_gray(tight_roi);

            // 4. Generăm numele fișierului (ex: faces/face_0.png, faces/face_1.png)
            string filename = output_folder + "/face_" + to_string(i) + ".png";

            // 5. Salvăm pe disc
            imwrite(filename, face_crop);
        }

        // Afișăm rezultatul
        imshow("Face Tracking", curr_frame);
        waitKey(1500); // Pauză de 500ms pentru a vedea cum se mișcă
    }

    return 0;
}