#include <algorithm>
//#include "omp.h"
#include "face_detector.hpp"

#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

namespace opengaze{

FaceDetector::FaceDetector():
           face_mean_val{104, 117, 123},
           face_std_val{0, 0, 0},
           lmk_mean_val{0, 0, 0},
           lmk_std_val{1.0 / 255, 1.0 / 255, 1.0 / 255}
{
    num_thread = 4;
    topk = -1;
    score_threshold = 0.85;
    iou_threshold = 0.4;
    face_in_w = 320;
    face_in_h = 240;
    w_h_list = {face_in_w, face_in_h};

    for (auto size : w_h_list) {
        std::vector<float> fm_item;
        for (float stride : strides) {
            fm_item.push_back(ceil(size / stride));
        }
        featuremap_size.push_back(fm_item);
    }

    for (auto size : w_h_list) {
        shrinkage_size.push_back(strides);
    }

    /* generate prior anchors */
    for (int index = 0; index < strides.size(); index++) {
        float scale_w = face_in_w / shrinkage_size[0][index];
        float scale_h = face_in_h / shrinkage_size[1][index];
        for (int j = 0; j < featuremap_size[1][index]; j++) {
            for (int i = 0; i < featuremap_size[0][index]; i++) {
                float x_center = (i + 0.5) / scale_w;
                float y_center = (j + 0.5) / scale_h;

                for (float k : min_boxes[index]) {
                    float w = k / face_in_w;
                    float h = k / face_in_h;
                    priors.push_back({clip(x_center, 1), clip(y_center, 1), clip(w, 1), clip(h, 1)});
                }
            }
        }
    }
    num_anchors = priors.size();
    /* generate prior anchors finished */

    // face_model.load_param(face_param.data());
    // face_model.load_model(face_bin.data());
    //landmks
    lmk_in_w = 112;
    lmk_in_h = 112;
    _lmks_num = 98;
    // landmks_model.load_param(lmk_param.data());
    // landmks_model.load_model(lmk_bin.data());
}

void FaceDetector::load_model(const std::string &model_path){
    
    const std::string face_param = model_path + "/face.param";
    const std::string face_bin = model_path + "/face.bin";
    const std::string lmk_param = model_path + "/landmks.param";
    const std::string lmk_bin = model_path +  "/landmks.bin";

    face_model.load_param(face_param.data());
    face_model.load_model(face_bin.data());
    landmks_model.load_param(lmk_param.data());
    landmks_model.load_model(lmk_bin.data());

    std::cout << "load face detect sucess .. " << std::endl;
}

FaceDetector::~FaceDetector() {
    face_model.clear(); 
    landmks_model.clear();}

void FaceDetector::Detect(cv::Mat &img, std::vector<opengaze::Sample> &output){


    ncnn::Mat inmat = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows);
    std::vector<FaceInfo> face_info;
    faceDetect(inmat, face_info);

    // std::cout << "face num: " << face_info.size() << std::endl;

    std::vector<cv::Point2d> landmks(98);

    for (int i = 0; i < face_info.size(); i++) {
        auto face = face_info[i];
        // std::cout << "face scores: " << face_info[i].score << endl;
        cv::Point pt1(face.x1, face.y1);
        cv::Point pt2(face.x2, face.y2);

        int face_w = pt2.x - pt1.x + 1;
        int face_h = pt2.y - pt1.y + 1;
        int face_cx = pt1.x + face_w / 2;
        int face_cy = pt1.y + face_h / 2;
        float expand = 1.1;

        int size = std::max(face_w, face_h) * expand;
        int roi_x1 = clip(face_cx - size/2, img.cols-1);
        int roi_x2 = clip(roi_x1 + size, img.cols-1);
        int roi_y1 = clip(face_cy - size/2, img.rows-1);
        int roi_y2 = clip(roi_y1 + size, img.rows-1);

        // std::cout << roi_x1 << " "<< roi_y1 << " "<< roi_x2 << " "<< roi_y2 << std::endl;
        // cv::rectangle(img, cv::Point(roi_x1, roi_y1), cv::Point(roi_x2, roi_y2), cv::Scalar(0, 255, 0), 2);
        cv::Mat face_roi = (img(cv::Rect(roi_x1, roi_y1, roi_x2-roi_x1, roi_y2-roi_y1))).clone();
        ncnn::Mat face_in = ncnn::Mat::from_pixels(face_roi.data, ncnn::Mat::PIXEL_BGR, face_roi.cols, face_roi.rows);
        landmksDetect(face_in, landmks);
        
        // cv::imshow("face", face_roi); 
        // debug
        // for (int i = 0; i < 98; ++i){
        //     // cout << "landmks[i]" <<landmks[i] << " ";
        //     cv::circle(img, cv::Point(landmks[i].x + roi_x1, landmks[i].y + roi_y1), 1, cv::Scalar(0, 255, 225), 2);
        // }
        
        opengaze::Sample temp;
        temp.face_data.certainty = face_info[i].score;
        temp.face_data.face_id = i;
        temp.face_data.face_bb.x = roi_x1;
        temp.face_data.face_bb.y = roi_y1;
        temp.face_data.face_bb.height = roi_y2-roi_y1;
        temp.face_data.face_bb.width = roi_x2-roi_x1;  
        
        // for (int j= 0; j < lmks.size(); ++j){
        //     temp.face_data.landmarks[j] = cv::Point(landmks[lmks[j]].x , landmks[lmks[j]].y);
        // }  

        for (int j= 0; j < 98; ++j){
            temp.face_data.landmarks[j] = cv::Point(landmks[j].x+roi_x1, landmks[j].y+roi_y1 );
        }    
        output.emplace_back(temp);
    }

    // cv::imshow("face detect", img);

}

int FaceDetector::faceDetect(ncnn::Mat &img, std::vector<FaceInfo> &face_list) {
    if (img.empty()) {
        std::cout << "image is empty ,please check!" << std::endl;
        return -1;
    }

    face_image_h = img.h;
    face_image_w = img.w;

    ncnn::Mat in;
    ncnn::resize_bilinear(img, in, face_in_w, face_in_h);
    ncnn::Mat ncnn_img = in;
    ncnn_img.substract_mean_normalize(face_mean_val, 0);

    std::vector<FaceInfo> bbox_collection;
    std::vector<FaceInfo> valid_input;

    ncnn::Extractor ex = face_model.create_extractor();
    ex.set_num_threads(num_thread);
    ex.input("input", ncnn_img);

    ncnn::Mat conf;
    ncnn::Mat loc;
    ncnn::Mat landmks;
    ex.extract("loc", loc);
    ex.extract("conf", conf);
    ex.extract("landmks", landmks);
    generateBBox(bbox_collection, conf, loc, score_threshold, num_anchors);
    nms(bbox_collection, face_list);
    return 0;
}

void FaceDetector::landmksDetect(ncnn::Mat &img, std::vector<cv::Point2d>& landmks)
{
    lmk_image_h = img.h;
    lmk_image_w = img.w;
    float long_side = std::max(lmk_image_h, lmk_image_w);

    ncnn::Mat in;
    ncnn::resize_bilinear(img, in, lmk_in_w, lmk_in_h);
    in.substract_mean_normalize(lmk_mean_val, lmk_std_val);

    ncnn::Extractor ex = landmks_model.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    ex.input("input", in);
    ncnn::Mat feature, lks;

    // ex.extract("mid_feture", feature);
    //landmark
    ex.extract("landmks", lks);

    float *landms = lks.channel(0);

    // #pragma omp parallel for num_threads(2)
    for (int i = 0; i < _lmks_num; ++i)
    {
        cv::Point2d pt;
        pt.x = landms[2*i] * long_side;
        pt.y = landms[2*i+1] * long_side;
        landmks[i]= pt;
        // std::cout << landms[2*i] << " " << landms[2*i+1] << std::endl;
    }
}

void FaceDetector::generateBBox(std::vector<FaceInfo> &bbox_collection, ncnn::Mat scores, ncnn::Mat boxes, float score_threshold, int num_anchors) {
    for (int i = 0; i < num_anchors; i++) {
        if (scores.channel(0)[i * 2 + 1] > score_threshold && scores.channel(0)[i * 2 + 1] < 1.0) {
            FaceInfo rects;
            float x_center = boxes.channel(0)[i * 4] * center_variance * priors[i][2] + priors[i][0];
            float y_center = boxes.channel(0)[i * 4 + 1] * center_variance * priors[i][3] + priors[i][1];
            float w = exp(boxes.channel(0)[i * 4 + 2] * size_variance) * priors[i][2];
            float h = exp(boxes.channel(0)[i * 4 + 3] * size_variance) * priors[i][3];

            rects.x1 = clip(x_center - w / 2.0, 1) * face_image_w;
            rects.y1 = clip(y_center - h / 2.0, 1) * face_image_h;
            rects.x2 = clip(x_center + w / 2.0, 1) * face_image_w;
            rects.y2 = clip(y_center + h / 2.0, 1) * face_image_h;
            rects.score = clip(scores.channel(0)[i * 2 + 1], 1);
            bbox_collection.push_back(rects);
        }
    }
}


void FaceDetector::nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, int type) {
    std::sort(input.begin(), input.end(), [](const FaceInfo &a, const FaceInfo &b) { return a.score > b.score; });

    int box_num = input.size();

    std::vector<int> merged(box_num, 0);

    for (int i = 0; i < box_num; i++) {
        if (merged[i])
            continue;
        std::vector<FaceInfo> buf;

        buf.push_back(input[i]);
        merged[i] = 1;

        float h0 = input[i].y2 - input[i].y1 + 1;
        float w0 = input[i].x2 - input[i].x1 + 1;

        float area0 = h0 * w0;
        

        for (int j = i + 1; j < box_num; j++) {
            if (merged[j])
                continue;

            float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
            float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

            float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
            float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

            float inner_h = inner_y1 - inner_y0 + 1;
            float inner_w = inner_x1 - inner_x0 + 1;

            float inner_area = inner_h * inner_w;

            float h1 = input[j].y2 - input[j].y1 + 1;
            float w1 = input[j].x2 - input[j].x1 + 1;

            float area1 = h1 * w1;

            float score;

            score = inner_area / (area0 + area1 - inner_area);

            if (score > iou_threshold) {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }
        switch (type) {
            case hard_nms: {
                output.push_back(buf[0]);
                break;
            }
            case blending_nms: {
                float total = 0;
                for (int i = 0; i < buf.size(); i++) {
                    total += exp(buf[i].score);
                }
                FaceInfo rects;
                memset(&rects, 0, sizeof(rects));
                for (int i = 0; i < buf.size(); i++) {
                    float rate = exp(buf[i].score) / total;
                    rects.x1 += buf[i].x1 * rate;
                    rects.y1 += buf[i].y1 * rate;
                    rects.x2 += buf[i].x2 * rate;
                    rects.y2 += buf[i].y2 * rate;
                    rects.score += buf[i].score * rate;
                }
                output.push_back(rects);
                break;
            }
            default: {
                printf("wrong type of nms.");
                exit(-1);
            }
        }
    }
}

}
