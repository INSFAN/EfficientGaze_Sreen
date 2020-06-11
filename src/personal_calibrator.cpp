
#include "personal_calibrator.hpp"

using namespace cv;
using namespace std;

void CallBackFunc(int event, int x, int y, int flags, void* is_click)  {
    if (event == EVENT_LBUTTONDOWN){
        bool* temp = (bool*)is_click;
        *temp = true;
    }
}

PersonalCalibrator::PersonalCalibrator (int screen_width, int screen_height) {
    cv::namedWindow("calibration", CV_WINDOW_NORMAL);
    cv::setWindowProperty("calibration", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    // set the mouse
    is_click_ = false;
    //set the callback function for any mouse event
    // setMouseCallback("calibration", CallBackFunc, &is_click_); // wait for clicking

    screen_width_ = screen_width;
    screen_height_ = screen_height;

    center_radius_ = (int)((float)screen_width_ / 200.0f);
}

PersonalCalibrator::~PersonalCalibrator() {
    cv::setWindowProperty("calibration", CV_WND_PROP_FULLSCREEN, CV_WINDOW_NORMAL);
    cv::destroyWindow("calibration");
}

void PersonalCalibrator::generatePoints(int num_points) {
    index_point_ = -1;
    srand(time(NULL));
    Point2i current_point;

    for (int num = 0; num < num_points; ++num) {
        current_point.x = (rand() % screen_width_); // range is [0, 1]
        current_point.y = (rand() % screen_height_); // range is [0, 1]
        points_.emplace_back(current_point);
    }
}

void PersonalCalibrator::initialWindow() {
    // get the focus of the window
    namedWindow("GetFocus", CV_WINDOW_NORMAL);
    cv::Mat img = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::imshow("GetFocus", img);
    cv::setWindowProperty("GetFocus", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    waitKey(1);
    cv::setWindowProperty("GetFocus", CV_WND_PROP_FULLSCREEN, CV_WINDOW_NORMAL);
    cv::destroyWindow("GetFocus");

    // show instruction
    cv::Mat show_img = cv::Mat::zeros(screen_height_, screen_width_, CV_8UC3);
    string show_text = "Please click/touch when looking at the dots";
    cv::putText(show_img, show_text, cv::Point(400,600), FONT_HERSHEY_COMPLEX_SMALL, 2, cv::Scalar(255,255,255), 2);
    imshow("calibration", show_img);
    cv::waitKey(3000);
    for (int i=255; i > 0; i-=5) { //Gradient disappear, nice!
        show_img = cv::Mat::zeros(screen_height_, screen_width_, CV_8UC3);
        cv::putText(show_img, show_text, cv::Point(400,600), FONT_HERSHEY_COMPLEX_SMALL, 2, cv::Scalar(i,i,i), 2);
        imshow("calibration", show_img);
        cv::waitKey(1);
    }
}

bool PersonalCalibrator::showNextPoint() {
    cv::Mat show_img = cv::Mat::zeros(screen_height_, screen_width_, CV_8UC3);
    index_point_ ++;
    cv::circle(show_img, cv::Point(points_[index_point_].x, points_[index_point_].y), center_radius_, cv::Scalar(255, 255, 255), -1);
    is_click_ = false;
    double t0 = static_cast<double>(getTickCount());
    while (true) {
        imshow("calibration", show_img);
        
        int key = cv::waitKey(10); // wait for interaction
        if (key == 27) // if press the ESC key
            return false;
        double ts = ((double)getTickCount()-t0)/getTickFrequency();
        if (is_click_|| ts > 4) {
        //  if (is_click_){
            is_click_ = true;
            break;
        }
    }
    return true;
}

void PersonalCalibrator::confirmClicking() {
    cv::Mat show_img = cv::Mat::zeros(screen_height_, screen_width_, CV_8UC3);
    cv::circle(show_img, cv::Point(points_[index_point_].x, points_[index_point_].y), center_radius_, cv::Scalar(0, 200, 0), -1);
    imshow("calibration", show_img);
    cv::waitKey(500);
}

// this polyfit function is copied from  opencv/modules/contrib/src/polyfit.cpp
// This original code was written by
//  Onkar Raut
//  Graduate Student,
//  University of North Carolina at Charlotte
cv::Mat polyfit(const Mat& src_x, const Mat& src_y, int order)
{
    CV_Assert((src_x.rows>0)&&(src_y.rows>0)&&(src_x.cols>0)&&(src_y.cols>0)&&(order>=1));
    Mat matrix;
    Mat bias = Mat::ones((int)src_x.rows, 1, CV_32FC1);

    Mat input_x = Mat::zeros(src_x.rows, order*src_x.cols, CV_32FC1);

    Mat copy;
    for(int i=1; i<=order;i++){
        copy = src_x.clone();
        pow(copy,i,copy);
        copy.copyTo(input_x(Rect((i-1)*src_x.cols, 0, copy.cols, copy.rows)));
    }

    // Mat input_axis_x, input_axis_y; 
    // cv::hconcat(src_x.col(0), bias, input_axis_x);
    // cv::hconcat(src_x.col(1), bias, input_axis_y);
    // Mat model_x, model_y;
    // cout << "new_mat: " << input_axis_x << endl;
    // cv::solve(input_axis_x, src_y.col(0), model_x, DECOMP_NORMAL);
    // cv::solve(input_axis_y, src_y.col(1), model_y, DECOMP_NORMAL);
    // Mat model;
    // cv::hconcat(model_x, model_y, model);
    // cout << "model_matrix: " << model << endl;
    // Mat calibrated;
    // cv::hconcat(input_axis_x * model.col(0), input_axis_y * model.col(1), calibrated);

    Mat new_mat;
    cv::hconcat(input_x, bias, new_mat);
    cout << "new_mat: " << new_mat << endl;
    cv::solve(new_mat, src_y, matrix, DECOMP_NORMAL);

    cout << "model_matrix: " << matrix << endl;
    Mat calibrated = new_mat * matrix;

    cout << "calibrated: " << calibrated << endl;
    double dist_original = norm(src_x, src_y, NORM_L2);
    cout << "dist_original: " << dist_original << endl;
    double dist_calibrated = norm(calibrated, src_y, NORM_L2);
    cout << "dist_calibrated: " << dist_calibrated << endl;

    return matrix;

}

void PersonalCalibrator::generateModel(vector<Point2f> prediction, vector<Point2f> ground_truth, int order) {

    cv::Mat input_x = cv::Mat((int)prediction.size(), 2, CV_32FC1, prediction.data());
    cv::Mat input_y = cv::Mat((int)ground_truth.size(), 2, CV_32FC1, ground_truth.data());

    cv::FileStorage storage("../output/pred_gt.yml", cv::FileStorage::WRITE);
    storage << "pred" <<input_x;
    storage << "gt" <<input_y;
    storage.release(); 

    // cout << "input_x: " << input_x << endl;
    // cout << "input_y: " << input_y << endl;
    // cv::Mat model_matrix;
    // model_matrix_ = polyfit(input_x, input_y, order);
}

void PersonalCalibrator::saveModel(std::string file_path) {
    cv::FileStorage storage(file_path, cv::FileStorage::WRITE);
    storage << "person_model" <<model_matrix_;
    storage.release();  
}
