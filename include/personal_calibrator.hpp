#ifndef PERSONAL_CALIBRATOR_HPP
#define PERSONAL_CALIBRATOR_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class PersonalCalibrator {

public:
    PersonalCalibrator(int screen_width, int screen_height);
    ~PersonalCalibrator();
    /**
     * generate the random locations for calibration
     * @param num_points number of points to generate
     */
    void generatePoints(int num_points);
    // get the show window ready, it should be full-screen
    void initialWindow();
    // show the next calibration point
    bool showNextPoint();
    // wait for 0.5 second to receive the confirmation (mouse click) from user
    void confirmClicking();
    /**
     * generate a polynomial function for the personal calibration
     * @param prediction prediction from gaze estimation method
     * @param ground_truth calibration points locations on the screen
     * @param order the order of polynomial function, 1 means the linear
     */
    void generateModel(std::vector<cv::Point2f> prediction, std::vector<cv::Point2f> ground_truth, int order=1);
    /**
     * save the personal calibration model
     * @param file_path path to save the model
     */
    void saveModel(std::string file_path);
    /**
     * load the personal calibration model
     * @param file_path path to load the model
     */
    void loadModel(std::string file_path);
    /**
     * return current calibration point location on the screen
     * @return location on the screen
     */
    cv::Point2f getCurrentPoint() {return points_[index_point_];}
    // function to calculate the polynomial function
    void calibratePolynomial();

private:
    // indicator if the user click the mouse or not
    bool is_click_;
    // number of points for personal calibration
    int num_points_;
    // index for the current calibration points
    int index_point_;
    // vector to store the generated calibration points
    std::vector<cv::Point2i> points_;
    int screen_width_, screen_height_, center_radius_; // monitor width and height in pixel
    // personal model
    cv::Mat model_matrix_;
};


#endif //PERSONAL_CALIBRATOR_HPP
