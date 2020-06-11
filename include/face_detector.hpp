//
// Created by dl on 19-7-19.
//

#ifndef FACE_DETECTOR_HPP
#define FACE_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <stack>
#include "net.h"
#include "data.hpp"
#include <chrono>
using namespace std::chrono;

#define hard_nms 1
#define blending_nms 2 /* mix nms was been proposaled in paper blaze face, aims to minimize the temporal jitter*/

namespace opengaze{

class Timer
{
public:
    std::stack<high_resolution_clock::time_point> tictoc_stack;

    void tic()
    {
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        tictoc_stack.push(t1);
    }

    double toc(std::string msg = "", bool flag = true)
    {
        double diff = duration_cast<milliseconds>(high_resolution_clock::now() - tictoc_stack.top()).count();
        if(msg.size() > 0){
            if (flag)
                printf("%s time elapsed: %f ms\n", msg.c_str(), diff);
        }

        tictoc_stack.pop();
        return diff;
    }
    void reset()
    {
        tictoc_stack = std::stack<high_resolution_clock::time_point>();
    }
};

typedef struct FaceInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;

    float *landmarks;
} FaceInfo;

class FaceDetector
{

public:
    FaceDetector();

    // void Init(const std::string &model_param, const std::string &model_bin);

    ~FaceDetector();

    void load_model(const std::string &model_path);

    int faceDetect(ncnn::Mat &img, std::vector<FaceInfo> &face_list);
    void landmksDetect(ncnn::Mat &img, std::vector<cv::Point2d>& landmks);
    void Detect(cv::Mat &img, std::vector<opengaze::Sample> &output);
private:
   void generateBBox(std::vector<FaceInfo> &bbox_collection, ncnn::Mat scores, ncnn::Mat boxes, float score_threshold, int num_anchors);

    void nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, int type = blending_nms);
    // void create_anchor_retinaface(std::vector<box> &anchor, int w, int h);

private:
    ncnn::Net face_model;

    int num_thread;
    int face_image_w;
    int face_image_h;

    int face_in_w;
    int face_in_h;
    int num_anchors;

    float face_mean_val[3];
    float face_std_val[3];

    int topk;
    float score_threshold;
    float iou_threshold;
    const float center_variance = 0.1;
    const float size_variance = 0.2;

    const std::vector<std::vector<float>> min_boxes = {
            {10, 20},
            {32, 64},
            {128, 256}};
    const std::vector<float> strides = {8.0, 16.0, 32.0};
    std::vector<std::vector<float>> featuremap_size;
    std::vector<std::vector<float>> shrinkage_size;
    std::vector<int> w_h_list;

    std::vector<std::vector<float>> priors = {};

    float _threshold;
    float lmk_mean_val[3];
    float lmk_std_val[3];

    //landmark 
    ncnn::Net landmks_model;
    uint16_t _lmks_num;
    int lmk_image_w;
    int lmk_image_h;

    int lmk_in_w;
    int lmk_in_h;
    const std::vector<int> lmks = {33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16};
};

}


#endif //
