#ifndef GAZE_ESTIMATOR_HPP
#define GAZE_ESTIMATOR_HPP

#include <opencv2/opencv.hpp>

#include "data.hpp"
#include "face_detector.hpp"
#include "normalizer.hpp"
#include "gaze_predictor.hpp"

namespace opengaze{

class GazeEstimator {
public:
    GazeEstimator();
    ~GazeEstimator();

    /**
     * On the current implementation, we only has the "MPIIGaze" method which uses the input face/eye image
     * and output gaze direction directly. It is an appearance-based method. The "OpenFace" can also output
     * the gaze vector according to the pupil detection results. However, "OpenFace" implementation is not 
     * included inside our OpenGaze toolkit yet.
     */
    enum Method{EfficientGaze};
    /**
     * for the "MPIIGaze" method, the input image can be face or eye. The full-face patch model can output
     * more accurate gaze prediction than the eye image model, while the eye image base model is much faster.
     */
    // enum InputType{face, eye};

    /**
     * the main function to estimate the gaze. 
     * It performs the face and facial landmarks detection, head pose estimation and then gaze prediction.
     * @param input_image input scene image
     * @param output  data structure for output
     */
    void estimateGaze(cv::Mat input_image, std::vector<opengaze::Sample> &output);
    void getImagePatch(cv::Mat input_image, std::vector<opengaze::Sample> &outputs);
    void setCameraParameters(cv::Mat camera_matrix, cv::Mat camera_dist);
    void setRootPath(std::string root_path);
    void setMethod(Method, std::vector<std::string> arguments);
    void initial_all_model(const std::string model_path);

    Method method_type_;
    // InputType input_type_; // the input type

private:
    // class instances
    FaceDetector face_detector_;
    Normalizer normalizer_;
    GazePredictor gaze_predictor_;
    // camera intrinsic matrix
    cv::Mat camera_matrix_;
    // camera distortion matrix
    cv::Mat camera_dist_;
    // the root pat is used for load configuration file and models
    std::string root_path_;
};

}




#endif //GAZE_ESTIMATOR_HPP
