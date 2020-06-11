#include "normalizer.hpp"

using namespace cv;
using namespace std;

namespace opengaze {

Normalizer::Normalizer() {
    // parameters for data normalization
    focal_norm_ = 1300;
    distance_norm_ = 550; // 600 500 1000
    eyes_Size_norm_ = cv::Size(320, 64); // 224 448
    face_Size_norm_ = cv::Size(128, 96); // 224 448
    cam_norm_ = (Mat_<float>(3,3) << focal_norm_, 0, eyes_Size_norm_.width/2, 0, focal_norm_, eyes_Size_norm_.height/2.0f, 0, 0, 1.0f);
}

Normalizer::~Normalizer() {}

void Normalizer::setParameters(int focal_length, int distance, int img_w, int img_h){
    // parameters for data normalization
    focal_norm_ = focal_length;
    distance_norm_ = distance; // 600 500 1000
    eyes_Size_norm_ = cv::Size(img_w, img_h); // 224 448
    cam_norm_ = (Mat_<float>(3,3) << focal_norm_, 0, eyes_Size_norm_.width/2, 0, focal_norm_, eyes_Size_norm_.height/2.0f, 0, 0, 1.0f);
}

// convert vector from normalization space to camera coordinate system
cv::Mat Normalizer::cvtToCamera(cv::Point3f input, const Mat cnv_mat) {
    // convert to the original camera coordinate system
    Vec3f gaze_v(input.x, input.y, input.z);
    // apply de-normalization
    Mat gaze_v_cam = cnv_mat.inv() * Mat(gaze_v);
    gaze_v_cam = gaze_v_cam / norm(gaze_v_cam);

    return gaze_v_cam;
}

vector<cv::Mat> Normalizer::normalizeEyesaAndFace(Mat input_image, opengaze::Sample &sample) {
    vector<cv::Mat> images;
    // get the face center in 3D space
    Mat HR;
    cv::Rodrigues(sample.face_patch_data.head_r, HR);
    Mat HT = repeat(sample.face_patch_data.head_t, 1, lmks_index.size());
    Mat Fc;
    add(HR*face_model_mat_, HT, Fc);

    float distance = (float)norm(sample.face_patch_data.face_center); // original distance
    float z_scale = distance_norm_ / distance; // scaling factor
    cv::Mat scaleMat;
    scaleMat = (Mat_<float>(3,3) << 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, z_scale);// scaling matrix
    scaleMat.convertTo(scaleMat, CV_32F);

    // get the look_at matrix
    Mat hRx = HR.col(0);
    Mat forward = sample.face_patch_data.face_center /distance;
    Mat down = forward.cross(hRx);
    down = down / norm(down);
    Mat right = down.cross(forward);
    right = right / norm(right);

    // rotation matrix
    sample.face_patch_data.face_rot = Mat(3, 3, CV_32F);
    right.copyTo(sample.face_patch_data.face_rot.col(0));
    down.copyTo(sample.face_patch_data.face_rot.col(1));
    forward.copyTo(sample.face_patch_data.face_rot.col(2));
    sample.face_patch_data.face_rot = sample.face_patch_data.face_rot.t(); // there is no scaling
    sample.face_patch_data.face_rot.convertTo(sample.face_patch_data.face_rot, CV_32F);
    Mat warpMat = cam_norm_ * (scaleMat * sample.face_patch_data.face_rot) * camera_matrix_.inv();// transformation matrix
    // crop image and copy the equalized image
    Mat eyes_patch, face_patch;

    warpPerspective(input_image, eyes_patch, warpMat, eyes_Size_norm_);
    warpPerspective(input_image, face_patch, warpMat, cv::Size(eyes_Size_norm_.width, 240));
    resize(face_patch, face_patch, face_Size_norm_);

    // debug
    // cv::imshow("face patch", face_patch);
    // cv::imshow("eyes patch", eyes_patch);

    images.emplace_back(eyes_patch);
    images.emplace_back(face_patch);

    return images;
}

// vector<cv::Mat> Normalizer::normalizeEyesaAndFace(cv::Mat input_image, Sample &sample){
//     vector<cv::Mat> images;

//     Mat img_gray;
//     cvtColor(input_image, img_gray, CV_BGR2GRAY);

//     Mat eye_center;
//     Mat* eye_rot;
//     for (int i=0; i<2; ++i) {
//         if (i==0){
//             eye_center = sample.eye_data.leye_pos;
//             eye_rot = &sample.eye_data.leye_rot;
//         }
//         else {
//             eye_center = sample.eye_data.reye_pos;
//             eye_rot = &sample.eye_data.reye_rot;
//         }


//         float distance = (float)norm(eye_center);
//         float z_scale = distance_norm_ / distance;

//         Mat scaleMat;
//         scaleMat = (Mat_<float>(3,3) << 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, z_scale);// scaling matrix
//         scaleMat.convertTo(scaleMat, CV_32F);

//         // get the look_at matrix
//         Mat HR;
//         cv::Rodrigues(sample.face_patch_data.head_r, HR);
//         Mat hRx = HR.col(0);
//         Mat forward = eye_center/distance;
//         Mat down = forward.cross(hRx);
//         down = down / norm(down);
//         Mat right = down.cross(forward);
//         right = right / norm(right);

//         // rotation matrix
//         *eye_rot = Mat(3, 3, CV_32F);
//         right.copyTo(eye_rot->col(0));
//         down.copyTo(eye_rot->col(1));
//         forward.copyTo(eye_rot->col(2));
//         *eye_rot = eye_rot->t(); // there is no scaling

//         Mat warpMat = cam_norm_ * (scaleMat * *eye_rot) * camera_matrix_.inv();// transformation matrix
//         // crop image and copy the equalized image
//         Mat eye_patch, eye_patch_equal;
//         warpPerspective(img_gray, eye_patch, warpMat, roiSize_norm_);
//         equalizeHist(eye_patch, eye_patch_equal);
//         eye_images.push_back(eye_patch_equal);

//     }
//     eye_rot = nullptr;
//     return eye_images;
// }

void Normalizer::loadFaceModel(std::string path) {
    string face_model_file_path = path + "/model/face_model.yml";
    //
    cout << endl << "Loading 3D face model for head pose estimation from : " << face_model_file_path << endl;
    FileStorage fs;
    if (!fs.open(face_model_file_path, FileStorage::READ)) {
        cout << "Cannot load the 3D face model!" << endl;
        exit(EXIT_FAILURE);
    }
    fs["face_model"] >> face_model_mat_;
    face_model_mat_ = face_model_mat_.t();
    for(int p=0; p<13; ++p)
        face_model_.emplace_back(Point3d(face_model_mat_.at<float>(0,p),
                                         face_model_mat_.at<float>(1,p),
                                         face_model_mat_.at<float>(2,p)));                          
    std::cout << "loaded face model successs. "  << std::endl;
    fs.release();
}

// estimate head pose via model fitting
void Normalizer::estimateHeadPose(const Point2f *landmarks, opengaze::Sample &sample) {


    Mat zero_dist = Mat::zeros(1, 5, CV_64F);
    vector<Point2f> landmarks_orig;
    for (int i = 0; i < lmks_index.size(); ++i){
        landmarks_orig.emplace_back(landmarks[lmks_index[i]]);
    }
    // vector<Point2d> landmarks_orig(landmarks,
    //                                landmarks + 6);
    cv::Mat head_r, head_t;
    camera_matrix_.convertTo(camera_matrix_, CV_64F); // input must be double type
    solvePnP(face_model_, landmarks_orig, camera_matrix_, zero_dist, head_r, head_t, false, SOLVEPNP_DLS);
    solvePnP(face_model_, landmarks_orig, camera_matrix_, zero_dist, head_r, head_t, true);
    head_r.convertTo(sample.face_patch_data.head_r, CV_32F);
    head_t.convertTo(sample.face_patch_data.head_t, CV_32F);
    camera_matrix_.convertTo(camera_matrix_, CV_32F);

    // get the face center in 3D space
    Mat HR;
    cv::Rodrigues(sample.face_patch_data.head_r, HR);
    Mat HT = repeat(sample.face_patch_data.head_t, 1, lmks_index.size());
    Mat Fc;
    add(HR*face_model_mat_, HT, Fc);

    // cout << HR << HT << endl;
    //  Gaze-origin (g_o)
    // Mat eyes_center = (Fc.col(1) + Fc.col(2)) / 2.0;
    Mat eyes_center = (Fc.col(10) + Fc.col(11)) / 2.0;

    // Mat face_center = (Fc.col(0) + Fc.col(1) + Fc.col(2) + Fc.col(3) + Fc.col(4) + Fc.col(5)) / 6.0; // face center
    eyes_center.copyTo(sample.face_patch_data.face_center); // copy to output
    sample.face_patch_data.face_center.convertTo(sample.face_patch_data.face_center, CV_32F);

    // Mat le = 0.5*(Fc.col(2) + Fc.col(3)); // left eye
    Mat le = 0.5*(Fc.col(9) + Fc.col(10)); // left eye
    le.copyTo(sample.eye_data.leye_pos);
    sample.eye_data.leye_pos.convertTo(sample.eye_data.leye_pos, CV_32F);
    // Mat re = 0.5*(Fc.col(0) + Fc.col(1)); // right eye
    Mat re = 0.5*(Fc.col(11) + Fc.col(12)); // right eye
    re.copyTo(sample.eye_data.reye_pos);
    sample.eye_data.reye_pos.convertTo(sample.eye_data.reye_pos, CV_32F);

}

void Normalizer::setCameraMatrix(cv::Mat camera_matrix, cv::Mat camera_dist) {
    camera_matrix_ = camera_matrix;
    camera_distortion_ = camera_dist;
    camera_matrix_.convertTo(camera_matrix_, CV_32F);
    camera_distortion_.convertTo(camera_distortion_, CV_32F);
}

}