#ifndef OPEN_GAZE_H
#define OPEN_GAZE_H

#include <string>
#include <vector>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <opencv2/opencv.hpp>

#include "input_handler.hpp"
#include "gaze_estimator.hpp"
#include "data.hpp"
#include "personal_calibrator.hpp"

namespace opengaze {

class OpenGaze {
public:
    explicit OpenGaze(int argc, char** argv); //read configuration file
    ~OpenGaze();

    // main function to estimate and show the gaze vector drawn on the input face image.
    void runGazeVisualization(); 

    /**
     * main function to run personal calibration.
     * @param num_calibration_point the numbers of points for calibration.
     */
    void runPersonalCalibration(int num_calibration_point=20);

    // main function to estimate and draw gaze point on the screen.
    void runGazeOnScreen();

    // main function to extract the face image from input image. The face image can then 
    // be used to train a custom gaze estimation model
    // void runDataExtraction();

    cv::Point2f personCalibrated(const float& src_x, const float& src_y, const cv::Mat &model_matrix);


private:
    // visualization
    /**
     * function to draw the gaze vector on the input face image.
     * @param sample the input data includes the gaze vector, head pose etc.
     * @param image the input image contains the face. This function will draw the gaze vector on this input image
     */
    void drawGazeOnFace(opengaze::Sample sample, cv::Mat &image);
    // draw the detected facial landmark
    void drawLandmarks(opengaze::Sample sample, cv::Mat &image);
    // draw the estimated gaze on the top left corner of the input image 
    // to show the relative position on the screen. In this case, 
    //the user can see both the input image and the projected gaze target on the screen. 
    //This function is mainly used for debugging.
    // void drawGazeOnSimScreen(opengaze::Sample sample, cv::Mat &image);
    // estimate and draw gaze point on the screen
    // void drawGazeOnScreen(opengaze::Sample sample, cv::Mat &image);

    // show debug mode will show the gaze draw on the face
    bool show_debug_;

    //class instances
    InputHandler input_handler_;
    GazeEstimator gaze_estimator_;

    // input camera id
    int camera_id_;
    // temporary variables to store the input path, output path, input type
    boost::filesystem::path input_dir_;
    InputHandler::InputType input_type_;
    boost::filesystem::path output_dir_;

    bool is_face_model_;
    bool is_save_video_;
    
    // path to save the personal calibration model
    std::string per_model_save_path_;
};

}

#endif //OPEN_GAZE_H
