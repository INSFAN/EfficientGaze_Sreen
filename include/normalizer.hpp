#ifndef NORMALIZER_HPP
#define NORMALIZER_HPP

#include <opencv2/opencv.hpp>
#include "data.hpp"

namespace opengaze{
class Normalizer {

public:
    Normalizer();
    ~Normalizer();

    void estimateHeadPose(const cv::Point2f *landmarks, opengaze::Sample &sample);

    void setCameraMatrix(cv::Mat camera_matrix, cv::Mat camera_dist);

    void loadFaceModel(std::string path);

    void setParameters(int focal_length, int distance, int img_w, int img_h);

    // vector<cv::Mat> normalizeFace(cv::Mat input_image, Sample &sample);

    std::vector<cv::Mat> normalizeEyesaAndFace(cv::Mat input_image, Sample &sample);

    cv::Mat cvtToCamera(cv::Point3f input, const cv::Mat cnv_mat);

private:
    cv::Mat camera_matrix_, camera_distortion_;
    cv::Mat face_model_mat_, cam_norm_;
    float focal_norm_, distance_norm_;
    cv::Size eyes_Size_norm_, face_Size_norm_;
    std::vector<cv::Point3d> face_model_;

    // const std::vector<int> lmks_index = {60, 64, 68, 72, 76, 82};
    const std::vector<int> lmks_index = {51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 64, 68, 72};
};


}




#endif //NORMALIZER_HPP
