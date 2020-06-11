#ifndef INPUT_HANDLER_HPP
#define INPUT_HANDLER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <boost/filesystem.hpp>
#include "data.hpp"

namespace opengaze {

class InputHandler {
public:
    enum InputType {Camera, Video, Image, Directory, Memory};

    InputHandler();
    ~InputHandler();

    /**
     * get the camera intrisic parameters
     * @param camera_matrix camera instric matrix
     * @param camera_dist caemra distortion matrix
     */
    void setCameraParameters(cv::Mat camera_matrix, cv::Mat camera_dist){
        camera_matrix_ = std::move(camera_matrix);
        camera_distortion_ = std::move(camera_dist);
    }
    
    /**
     * function to return next sample, could come from any input source
     * @return next sample
     */
    cv::Mat getNextSample();

    /**
     * set the input type
     * @param type the input typ, could be found in InputType defination
     */
    void setInputType(InputType type){input_type_ = type;}

    /**
     * set the input 
     *  according the input type, here the input value are different. 
     * For the type "Camera", this input value indicates the camera id
     * For the type "video", this input value is the video file name
     * For input type "Directory", this input value is the directory path
     */
    void setInput(int camera_id) {camera_id_ = camera_id;}
    void setInput(std::vector<cv::Mat> images) {images_ = std::move(images);}
    void setInput(std::string input_path);
    
    /**
     * read the parameters related to the screen
     * @param calib_file file for the configuration 
     */
    void readScreenConfiguration(std::string calib_file);
    /**
     * read the camera instrinic parameters from the configuration file
     * @param calib_file file for the configuration 
     */

    void readPersonCalibratedMat(std::string calib_file);

    
    void readCameraConfiguration(std::string calib_file);

    /**
     * When the 3D gaze vector is achieved, there is a need to project the gaze on the 2D screen.
     * This function also needs the input to indicate if use the full-face model or not, 
     * since the initial of gaze vector will be center of the face for the full-face models 
     * and eye center for the eye-based models.
     * @param input input data contains the 3D gaze vector
     * @param is_face_model a boolen value indicates if the gaze vectors is from face model or eye model 
     */
    void projectToDisplay(std::vector<opengaze::Sample> &input, bool is_face_model=true);

    int getFrameHeight(){return cap_.get(cv::CAP_PROP_FRAME_HEIGHT);}
    int getFrameWidth(){return cap_.get(cv::CAP_PROP_FRAME_WIDTH);}
    InputType getInputType() {return input_type_;}
    int getScreenWidth() {return screen_width_;}
    int getScreenHeight() {return screen_height_;}
    std::string getFileName() {return current_file_name_;}
    
    cv::Point2f mapToDisplay(cv::Vec3f obj_center, cv::Vec3f gaze_point);

    void initialize();
    bool closeInput();
    void getScreenResolution(int &width, int &height);
    
    cv::Mat getCameraMatrix() { return camera_matrix_;}
    cv::Mat getCameraDistortion() {return camera_distortion_;}
    void setFrameSize(int frame_width, int frame_height);

    bool isReachEnd() {return is_reach_end_;}

    cv::Mat camera_matrix_;
    cv::Mat camera_distortion_;
    // person calibarted
    cv::Mat person_model_;

     // monitor
    float monitor_W_, monitor_H_; // monitor width and height in mm
    cv::Mat monitor_R_, monitor_T_;
    cv::Vec3f monitor_corners_[4];
    cv::Mat monitor_normal_;

private:

    // indicator if we reach the end of sample stream
    bool is_reach_end_;

    int camera_id_;
    int sample_height_, sample_width_;
    std::vector<cv::Mat> images_;
    std::string input_path_;
    std::string input_file_video_name_;
    int screen_width_, screen_height_;

    // input variable
    InputType input_type_;
    cv::VideoCapture cap_;
    std::string current_file_name_;

    // variable for directory input
    boost::filesystem::directory_iterator current_itr_;

};

}

#endif //INPUT_HANDLER_HPP
