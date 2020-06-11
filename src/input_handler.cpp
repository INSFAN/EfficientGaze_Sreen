#include "input_handler.hpp"
#include <iostream>
#include <cstdlib>
using namespace cv;
using namespace std;

namespace opengaze {

#if WIN32
#include <windows.h>
#else
#include <X11/Xlib.h>
#endif

void InputHandler::getScreenResolution(int &width, int &height) {
#if WIN32
    width  = (int) GetSystemMetrics(SM_CXSCREEN);
    height = (int) GetSystemMetrics(SM_CYSCREEN);
#else
    Display* disp = XOpenDisplay(NULL);
    Screen*  scrn = DefaultScreenOfDisplay(disp);
    width  = scrn->width;
    height = scrn->height;
#endif
}

InputHandler::InputHandler(){
    input_type_ = InputType::Camera;// defualt input type
    camera_id_ = 0;
    getScreenResolution(screen_width_, screen_height_);
    screen_width_ = screen_width_;
}
InputHandler::~InputHandler(){}


void InputHandler::initialize()
{
    if (input_type_ == InputType::Camera){
        cap_.open(camera_id_);
        if(!cap_.isOpened()) { // open Camera
            cout << "Could not open Camera with id " << camera_id_ << endl;
            std::exit(EXIT_FAILURE);
        }
        // setFrameSize(640, 480); // 800*600, 1280*720, 1920*1080,
    }
    else if (input_type_ == InputType::Video){
        cap_.open(input_file_video_name_);
        if(!cap_.isOpened()) { // open Camera
            cout << "Error: Could not open video file " << input_file_video_name_ << endl;
            std::exit(EXIT_FAILURE);
        }
    }
    else if (input_type_ == InputType::Directory) {
        if (!boost::filesystem::is_directory(input_path_)){
            cout << "Error: The input must be a directory, but it is " << input_path_ << endl;
            std::exit(EXIT_FAILURE);
        }
        current_itr_ = boost::filesystem::directory_iterator(input_path_);

    }
    else if (input_type_ == InputType::Memory) {}

    is_reach_end_ = false;
}

void InputHandler::setFrameSize(int frame_width, int frame_height){
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height);//720  1080
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, frame_width);//1280 1980
    double dWidth = cap_.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
    double dHeight = cap_.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video
    cout << "Input frame size is : " << dWidth << " x " << dHeight << endl;
}

Mat InputHandler::getNextSample() {
    Mat frame;
    if (input_type_ == InputType::Camera) cap_ >> frame;
    else if (input_type_ == InputType::Video) {
        cap_ >> frame;
        if (frame.empty()) // we reach the end of video
            is_reach_end_ = true;
    }
    else if (input_type_ == InputType::Directory) {
        boost::filesystem::path file_path = current_itr_->path();
        if (file_path.extension() != ".jpg" && file_path.extension() != ".png" && file_path.extension() != ".bmp"){
            cout << "Error: The input file is not image file with extension of jpg, png or bmp!" << endl;
            cout << "The input file name is: " << file_path.string() << endl;
            std::exit(EXIT_FAILURE);
        }
        cout << "process image " << file_path << endl;
        frame = imread(file_path.string());
        if (current_itr_ == boost::filesystem::directory_iterator())
            is_reach_end_ = true;
    }
    else if (input_type_ == InputType::Memory) {}

    return frame;
}

bool InputHandler::closeInput() {
    if (input_type_ == InputType::Camera || input_type_ == InputType::Video){
        cap_.release();
        is_reach_end_ = true;
    }
    return true;
}

void InputHandler::setInput(std::string input_path) {
    if (input_type_ == InputType::Directory){
        input_path_ = move(input_path);
    }
    else if (input_type_ == InputType::Video){
        input_file_video_name_ = move(input_path);
    }
}

void InputHandler::readScreenConfiguration(string calib_file) {
    FileStorage fs_disp(calib_file, FileStorage::READ);
    fs_disp["monitor_W"] >> monitor_W_;
    fs_disp["monitor_H"] >> monitor_H_;
    fs_disp["monitor_R"] >> monitor_R_;
    fs_disp["monitor_T"] >> monitor_T_;
    // compute monitor plane
    Vec3f corners[4];
    corners[0] = Vec3f(0.0, 0.0, 0.0);
    corners[1] = Vec3f(monitor_W_, 0.0, 0.0);
    corners[2] = Vec3f(0.0, monitor_H_, 0.0);
    corners[3] = Vec3f(monitor_W_, monitor_H_, 0.0);

    for(int i=0; i<4; i++){
        Mat corners_cam = monitor_R_ * Mat(corners[i]) + monitor_T_;
        corners_cam.copyTo(monitor_corners_[i]);
    }

    Vec3f normal = Vec3f(0.0, 0.0, 1.0); // normal direction
    monitor_normal_ = monitor_R_ * Mat(normal);
    monitor_normal_.convertTo(monitor_normal_, CV_32F);
}

void InputHandler::readCameraConfiguration(string calib_file){
    cout << endl << "Reading calibration information from : " << calib_file << endl;
    FileStorage fs;
    fs.open(calib_file, FileStorage::READ);
    fs["camera_matrix"] >> camera_matrix_;
    fs["dist_coeffs"] >> camera_distortion_;

    // std::cout << "camera_matrix_: " << camera_matrix_ <<std::endl;
    // std::cout << "camera_distortion_: " << camera_distortion_ <<std::endl;

    fs.release();
}

void InputHandler::readPersonCalibratedMat(string calib_file){
    cout << endl << "Reading person calibration information from : " << calib_file << endl;
    FileStorage fs;
    fs.open(calib_file, FileStorage::READ);
    fs["person_model"] >> person_model_;
    // std::cout << "camera_matrix_: " << camera_matrix_ <<std::endl;
    // std::cout << "camera_distortion_: " << camera_distortion_ <<std::endl;

    fs.release();
}

void InputHandler::projectToDisplay(std::vector<opengaze::Sample> &inputs, bool is_face_model) {
    for(auto & sample : inputs) {
        if (is_face_model) {
            Vec3f face_center(sample.face_patch_data.face_center.at<float>(0), sample.face_patch_data.face_center.at<float>(1), sample.face_patch_data.face_center.at<float>(2));
            sample.gaze_data.gaze2d = mapToDisplay(face_center, sample.gaze_data.gaze3d);
        }
        // else {
        //     Vec3f leye_pose(sample.eye_data.leye_pos.at<float>(0),sample.eye_data.leye_pos.at<float>(1),sample.eye_data.leye_pos.at<float>(2));
        //     Vec3f reye_pose(sample.eye_data.reye_pos.at<float>(0),sample.eye_data.reye_pos.at<float>(1),sample.eye_data.reye_pos.at<float>(2));
        //     sample.gaze_data.lgaze2d = mapToDisplay(leye_pose, sample.gaze_data.lgaze3d);
        //     sample.gaze_data.rgaze2d = mapToDisplay(reye_pose, sample.gaze_data.rgaze3d);
        //     float gaze_x = (sample.gaze_data.lgaze2d.x + sample.gaze_data.rgaze2d.x) / 2.0f;
        //     float gaze_y = (sample.gaze_data.lgaze2d.y + sample.gaze_data.rgaze2d.y) / 2.0f;
        //     sample.gaze_data.gaze2d.x = gaze_x;
        //     sample.gaze_data.gaze2d.y = gaze_y;
        // }
    }
}

cv::Point2f InputHandler::mapToDisplay(Vec3f origin, Vec3f gaze_vec) {
    Point2f gaze_on_screen;
    // compute intersection
    // float gaze_len = (float)(monitor_normal_.dot(Mat(monitor_corners_[0]-origin))/monitor_normal_.dot(Mat(gaze_vec)));
    float gaze_len = (float)(500/monitor_normal_.dot(Mat(gaze_vec)));
    // Vec3f gaze_pos_cam = origin + 8 * gaze_len * gaze_vec;
    Vec3f gaze_pos_cam = origin + gaze_len * gaze_vec;

    // convert to monitor coodinate system
    Mat gaze_pos_ = monitor_R_.inv() * (Mat(gaze_pos_cam) - monitor_T_);
    Vec3f gaze_pos_3d;
    gaze_pos_.copyTo(gaze_pos_3d);

    gaze_on_screen.x = gaze_pos_3d.val[0] / monitor_W_;
    gaze_on_screen.y = gaze_pos_3d.val[1] / d;

    return gaze_on_screen;

}

}