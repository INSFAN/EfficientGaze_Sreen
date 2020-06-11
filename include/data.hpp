#ifndef DATA_HPP
#define DATA_HPP

#include <opencv2/opencv.hpp>

namespace opengaze{

/**
 * face and facial landmark detection data
 * @param face_id personal id from tracking across frames
 * @param certainty detection score, 1 is the best, -1 is the worst
 * @param landmarks detected six facial landmarks as four eye corners and two mouth corners
 * @param face_bb detected face bounding box
 */
struct FaceData
{
    unsigned long face_id;
    double certainty;
    cv::Point2f landmarks[98];
    cv::Rect_<int> face_bb;
};
/**
 * eye image related data
 * @param leye_pos/reye_pose 3D eyeball center position for left and right eyes in the original camera coordinate system
 * @param leye_img/reye_img eye image
 * @param leye_rot/reye_rot rotation matrix during the data normalization procedure
 */
struct EyeData
{
    // cv::Mat head_r, head_t; 
    cv::Mat leye_pos, reye_pos; // 

    // normalized eyes
    cv::Mat leye_img, reye_img;
    cv::Mat leye_rot, reye_rot;
};
/**
 * face patch data related to data normalization
 * @param head_r head pose as center of the face
 * @param head_t head translation as center of the face
 * @param face_rot rotation matrix during the data normalization procedure
 * @param face_center 3D face center in the original camera coordinate system
 * @param debug_img use for debug to show the normalized face image
 * @param face_patch normalized face image
 */
struct FacePatchData
{
    cv::Mat head_r, head_t;
    cv::Mat face_rot;
    cv::Mat face_center;
    cv::Mat debug_img;
    cv::Mat face_patch;
};
/**
 * gaze data
 * @param lgaze3d/lgaze3d gaze directions of left and right eyes in the camera coordinate system
 * @param gaze3d gaze direction estimated from face patch in the in the camera coordinate system
 * @param lgaze2d/rgaze2d projected gaze positions on the screen coordinate from left and right eyes
 * @param gaze2d projected gaze positions from face patch on the screen coordinate
 */
struct GazeData
{
    cv::Vec3f lgaze3d, rgaze3d;
    cv::Vec3f gaze3d;
    cv::Point2f lgaze2d, rgaze2d;
    cv::Point2f gaze2d;
};
/**
 * The general output data structure
 * @param face_data store face and facial landmark detection data
 * @param eye_data store data related to eye image input
 * @param face_patch_data normalized face path data
 * @param gaze_data gaze data in 2D and 3D spaces
 */
struct Sample
{
    FaceData face_data;
    EyeData eye_data;
    FacePatchData face_patch_data;
    GazeData gaze_data;
};

}




#endif //DATA_HPP
