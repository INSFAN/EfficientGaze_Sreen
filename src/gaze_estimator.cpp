#include <iostream>

#include "gaze_estimator.hpp"

using namespace std;
using namespace cv;

namespace opengaze{

GazeEstimator::GazeEstimator() {
    method_type_ = EfficientGaze;
}
GazeEstimator::~GazeEstimator() {}

void GazeEstimator::setRootPath(std::string root_path) {
    normalizer_.loadFaceModel(root_path);
}

void  GazeEstimator::estimateGaze(cv::Mat input_image, std::vector<opengaze::Sample> &outputs) {
    face_detector_.Detect(input_image, outputs); // detect faces and facial landmarks

    //debug
    // std::cout << "face detect end...face num: " << outputs.size() << endl; 
    // return;
    for (int i=0; i< outputs.size(); ++i) {

        // estimate head pose first, no matter what gaze estimation method, head pose is estimated here
        normalizer_.estimateHeadPose(outputs[i].face_data.landmarks, outputs[i]);
         if (method_type_ == Method::EfficientGaze){

             // if we use face model

            vector<cv::Mat> eyes_face = normalizer_.normalizeEyesaAndFace(input_image, outputs[i]);
            //outputs[i].face_patch_data.debug_img = face_patch;

            Point3f gaze_norm = gaze_predictor_.predict(eyes_face[1], eyes_face[0]); // gaze estimates in normalization space
            Mat gaze_3d = normalizer_.cvtToCamera(gaze_norm, outputs[i].face_patch_data.face_rot); // convert gaze to camera coordinate system
            gaze_3d.copyTo(outputs[i].gaze_data.gaze3d);

            //  else if (input_type_ == InputType::eye) {
            //      vector<cv::Mat> eye_iamges = normalizer_.normalizeEyes(input_image, outputs[i]); // generate eye images
            //      // for left eye
            //      Point3f gaze_norm = gaze_predictor_.predictGazeMPIIGaze(eye_iamges[0]); 
            //      Mat gaze_3d = normalizer_.cvtToCamera(gaze_norm, outputs[i].eye_data.leye_rot);
            //      gaze_3d.copyTo(outputs[i].gaze_data.lgaze3d);
            //      // for right eye
            //      Mat flip_right;
            //      flip(eye_iamges[0], flip_right, 1);
            //      gaze_norm = gaze_predictor_.predictGazeMPIIGaze(flip_right); // for left right image input
            //      gaze_norm.x *= -1.0;
            //      gaze_3d = normalizer_.cvtToCamera(gaze_norm, outputs[i].face_patch_data.face_rot); // convert gaze to camera coordinate system
            //      gaze_3d.copyTo(outputs[i].gaze_data.rgaze3d);
            //  }
         }
        //  else if (method_type_ == Method::OpenFace) {
        //     cout << "Please use gaze estimation method MPIIGaze." << endl;
        //     exit(EXIT_FAILURE);
        //  }
    }
}

// void GazeEstimator::getImagePatch(cv::Mat input_image, std::vector<opengaze::Sample> &outputs) {
//     face_detector_.track_faces(input_image, outputs); // detect faces and facial landmarks
//     for (int i=0; i< outputs.size(); ++i) {
//         // estimate head pose first, no matter what gaze estimation method, head pose is estimated here
//         normalizer_.estimateHeadPose(outputs[i].face_data.landmarks, outputs[i]);
//          if (method_type_ == Method::MPIIGaze){

//              // if we use face model
//              if  (input_type_ == InputType::face){
//                  outputs[i].face_patch_data.face_patch = normalizer_.normalizeFace(input_image, outputs[i]);
//              }
//              else if (input_type_ == InputType::eye) {
//                  vector<cv::Mat> eye_iamges = normalizer_.normalizeEyes(input_image, outputs[i]); // generate eye images
//                  outputs[i].eye_data.leye_img = eye_iamges[0];
//                  outputs[i].eye_data.reye_img = eye_iamges[1];
//              }
//          }
//         //  else if (method_type_ == Method::OpenFace) {
//         //     cout << "Please use method MPIIGaze for image patch extraction." << endl;
//         //     exit(EXIT_FAILURE);
//         //  }
//     }
// }

void GazeEstimator::setMethod(Method input_method_type, const std::vector<std::string> arguments={}) {
    method_type_ = input_method_type;

    if (method_type_ == Method::EfficientGaze) {
        // gaze_predictor_.initiaMPIIGaze(arguments);
        // if (arguments.size() < 2)
        //     input_type_ = InputType::face;
        // else {
        //     if (arguments[2] == "face"){
        //         input_type_ = InputType::face;
        //         normalizer_.setParameters(1600, 1000, 224, 224);
        //     }

        //     else if (arguments[2] == "eye") {
        //         input_type_ = InputType::eye;
        //         normalizer_.setParameters(960, 600, 60, 36);
        //     }
        // }
    }

}

void GazeEstimator::setCameraParameters(cv::Mat camera_matrix, cv::Mat camera_dist) {
    camera_matrix_ = move(camera_matrix);
    camera_dist_ = move(camera_dist_);
    normalizer_.setCameraMatrix(camera_matrix_, camera_dist);
}

void GazeEstimator::initial_all_model(const std::string model_path){
    // face_detector_.initialize(number_users);
    // face_detector_.setMethodType(FaceDetector::Method::OpenFace);

    face_detector_.load_model(model_path);
    gaze_predictor_.load_model(model_path);
}

};