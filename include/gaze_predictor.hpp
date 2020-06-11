//
// Created by dl on 19-7-19.
//

#ifndef GAZE_PREDICTOR_HPP
#define GAZE_PREDICTOR_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <stack>
#include "net.h"
#include "data.hpp"
#include <chrono>
using namespace std::chrono;

#define hard_nms 1
#define blending_nms 2 /* mix nms was been proposaled in paper blaze face, aims to minimize the temporal jitter*/

namespace opengaze{


class GazePredictor
{

public:
    GazePredictor();

    // void Init(const std::string &model_param, const std::string &model_bin);

    ~GazePredictor();

    void load_model(const std::string &model_path);

    cv::Point3f predict(cv::Mat &face_img, cv::Mat &eyes_img);

private:
    // face gaze aux net
    ncnn::Net gaze_face;
    
    int num_thread;
    int face_image_w;
    int face_image_h;

    int face_in_w;
    int face_in_h;

    float _mean_val[3];
    float _std_val[3];

    // main gaze eyes net
    ncnn::Net gaze_eyes;

    int eyes_image_w;
    int eyes_image_h;

    int eyes_in_w;
    int eyes_in_h;
};

}

#endif //

