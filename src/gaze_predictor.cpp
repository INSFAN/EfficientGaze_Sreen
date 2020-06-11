#include <algorithm>
//#include "omp.h"
#include "gaze_predictor.hpp"

#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

namespace opengaze{

GazePredictor::GazePredictor():
           _mean_val{0, 0, 0},
           _std_val{1.0 / 255, 1.0 / 255, 1.0 / 255}
{
    num_thread = 4;
    face_in_w = 128;
    face_in_h = 128;
    
    // gaze_face.load_param(gaze_face_param.data());
    // gaze_face.load_model(gaze_face_bin.data());
    
    eyes_in_w = 320;
    eyes_in_h = 64;
    
    // gaze_eyes.load_param(gaze_eyes_param.data());
    // gaze_eyes.load_model(gaze_eyes_bin.data());
    
}

GazePredictor::~GazePredictor() {
    gaze_face.clear(); 
    gaze_eyes.clear();}

void GazePredictor::load_model(const std::string &model_path){

    const std::string gaze_face_param = model_path + "/gaze_face.param";
    const std::string gaze_face_bin = model_path + "/gaze_face.bin";
    const std::string gaze_eyes_param = model_path + "/gaze_estimator.param";
    const std::string gaze_eyes_bin = model_path + "/gaze_estimator.bin";

    gaze_face.load_param(gaze_face_param.data());
    gaze_face.load_model(gaze_face_bin.data());
    gaze_eyes.load_param(gaze_eyes_param.data());
    gaze_eyes.load_model(gaze_eyes_bin.data());
}

cv::Point3f GazePredictor::predict(cv::Mat &face_img, cv::Mat &eyes_img){

    /**************************gaze face part****************************/
    ncnn::Mat faceinmat = ncnn::Mat::from_pixels(face_img.data, ncnn::Mat::PIXEL_BGR2RGB, face_img.cols, face_img.rows);
    face_image_h = faceinmat.h;
    face_image_w = faceinmat.w;

    faceinmat.substract_mean_normalize(_mean_val, _std_val);

    ncnn::Extractor gaze_face_ex = gaze_face.create_extractor();
    gaze_face_ex.set_num_threads(num_thread);
    gaze_face_ex.input("input", faceinmat);

    ncnn::Mat face_feature;
    gaze_face_ex.extract("face_feature",face_feature);

     /**************************gaze eyes part****************************/
    ncnn::Mat eyesinmat = ncnn::Mat::from_pixels(eyes_img.data, ncnn::Mat::PIXEL_BGR2RGB, eyes_img.cols, eyes_img.rows);
    eyes_image_h = eyesinmat.h;
    eyes_image_w = eyesinmat.w;

    eyesinmat.substract_mean_normalize(_mean_val, _std_val);

    ncnn::Extractor gaze_eyes_ex = gaze_eyes.create_extractor();
    gaze_eyes_ex.set_num_threads(num_thread);
    gaze_eyes_ex.input("input", eyesinmat);
    gaze_eyes_ex.input("face_feature", face_feature);

    ncnn::Mat gaze3d;
    gaze_eyes_ex.extract("gaze",gaze3d);

    float *gaze = gaze3d.channel(0);

    // output.gaze_data.gaze3d[0] = gaze[0]/100;
    // output.gaze_data.gaze3d[1] = gaze[1]/100;
    cv::Point3f gaze_norm_3d;
    float theta = gaze[0]/100;
    float phi = gaze[1]/100;
    gaze_norm_3d.x = (-1.0f)*cos(theta)*sin(phi);
    gaze_norm_3d.y = (-1.0f)*sin(theta);
    gaze_norm_3d.z = (-1.0f)*cos(theta)*cos(phi);

    return gaze_norm_3d;

}

}
