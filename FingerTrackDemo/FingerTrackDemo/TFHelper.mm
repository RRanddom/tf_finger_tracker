//
//  TFHelper.m
//  Demo
//
//  Created by zjcneil on 2019/1/30.
//  Copyright © 2019 zjcneil. All rights reserved.
//

#import "TFHelper.h"
#include <queue>
#include <iomanip>
#include <iostream>

#import <AssertMacros.h>
#import <AssetsLibrary/AssetsLibrary.h>

#import <AVFoundation/AVFoundation.h>

#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/op_resolver.h"

#include <iostream>

#define LOG(x) std::cerr


static NSString* model_file_name = @"finger_track";
static NSString* model_file_type = @"tflite";

static const int wanted_input_width = 640;
static const int wanted_input_height = 480;
static const int wanted_input_channels = 3;
//1,480,640,3

@implementation TFHelper


static float hi_prob_thresh = .25;

static float32_t X[15][20] = {
    {-0.95, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95},
    {-0.95, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95},
    {-0.95, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95},
    {-0.95, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95},
    {-0.95, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95},
    {-0.95, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95},
    {-0.95, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95},
    {-0.95, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95},
    {-0.95, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95},
    {-0.95, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95},
    {-0.95, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95},
    {-0.95, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95},
    {-0.95, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95},
    {-0.95, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95},
    {-0.95, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95}
};

static float32_t Y[15][20] = {
    {-0.9333,-0.9333,-0.9333,-0.9333,-0.9333,-0.9333,-0.9333,-0.9333,-0.9333,-0.9333,-0.9333,-0.9333,-0.9333,-0.9333,-0.9333,-0.9333,-0.9333,-0.9333,-0.9333,-0.9333},
    {-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8},
    {-0.6667,-0.6667,-0.6667,-0.6667,-0.6667,-0.6667,-0.6667,-0.6667,-0.6667,-0.6667,-0.6667,-0.6667,-0.6667,-0.6667,-0.6667,-0.6667,-0.6667,-0.6667,-0.6667,-0.6667},
    {-0.5333,-0.5333,-0.5333,-0.5333,-0.5333,-0.5333,-0.5333,-0.5333,-0.5333,-0.5333,-0.5333,-0.5333,-0.5333,-0.5333,-0.5333,-0.5333,-0.5333,-0.5333,-0.5333,-0.5333},
    {-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4},
    {-0.2667,-0.2667,-0.2667,-0.2667,-0.2667,-0.2667,-0.2667,-0.2667,-0.2667,-0.2667,-0.2667,-0.2667,-0.2667,-0.2667,-0.2667,-0.2667,-0.2667,-0.2667,-0.2667,-0.2667},
    {-0.1333,-0.1333,-0.1333,-0.1333,-0.1333,-0.1333,-0.1333,-0.1333,-0.1333,-0.1333,-0.1333,-0.1333,-0.1333,-0.1333,-0.1333,-0.1333,-0.1333,-0.1333,-0.1333,-0.1333},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0.1333,0.1333,0.1333,0.1333,0.1333,0.1333,0.1333,0.1333,0.1333,0.1333,0.1333,0.1333,0.1333,0.1333,0.1333,0.1333,0.1333,0.1333,0.1333,0.1333},
    {0.2667,0.2667,0.2667,0.2667,0.2667,0.2667,0.2667,0.2667,0.2667,0.2667,0.2667,0.2667,0.2667,0.2667,0.2667,0.2667,0.2667,0.2667,0.2667,0.2667},
    {0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4},
    {0.5333,0.5333,0.5333,0.5333,0.5333,0.5333,0.5333,0.5333,0.5333,0.5333,0.5333,0.5333,0.5333,0.5333,0.5333,0.5333,0.5333,0.5333,0.5333,0.5333},
    {0.6667,0.6667,0.6667,0.6667,0.6667,0.6667,0.6667,0.6667,0.6667,0.6667,0.6667,0.6667,0.6667,0.6667,0.6667,0.6667,0.6667,0.6667,0.6667,0.6667},
    {0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8},
    {0.9333,0.9333,0.9333,0.9333,0.9333,0.9333,0.9333,0.9333,0.9333,0.9333,0.9333,0.9333,0.9333,0.9333,0.9333,0.9333,0.9333,0.9333,0.9333,0.9333}
};


+ (instancetype) sharedInstance {
    static TFHelper *sharedInstance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedInstance = [[self alloc] init];
    });
    return sharedInstance;
}

- (instancetype) init {
    if (self = [super init]) {
        [self loadModel];
    }return self;
}

static NSString* FilePathForResourceName(NSString* name, NSString* extension) {
    NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
    if (file_path == NULL) {
        LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "." << [extension UTF8String]
        << "' in bundle.";
    }
    return file_path;
}

- (void) loadModel {
    NSString *graph_path = FilePathForResourceName(model_file_name, model_file_type);
    model = tflite::FlatBufferModel::BuildFromFile([graph_path UTF8String]);
    if (!model) {
        LOG(FATAL) << "Failed to mmap model " << graph_path <<std::endl;
    }
    LOG(INFO) << "Loaded Model:" << graph_path << std::endl;
    model->error_reporter();
    LOG(INFO) << "resolved reporter" << std::endl;
    
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    
    if (!interpreter) {
        LOG(FATAL) << "Failed to construct interpreter" << std::endl;
    }
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        LOG(FATAL) << "Failed to allocate tensors!" << std::endl;
    }
    LOG(INFO) << "Everything works well!" <<std::endl;
}

void dsnt(int height, int width, float32_t *heatmaps, int *X_coords, int *Y_coords) {
    // 1. norm the heatmaps.
    float32_t exp_sum[] = {0,0,0};
    int dim = 3;
    for (int h=0; h<height; h++) {
        auto a_row = heatmaps + h * dim * width;
        for (int w=0; w<width; w++) {
            auto a_pos = a_row + w * dim;
            for (int d=0; d<dim; d++) {
                auto the_element = a_pos[d];
                a_pos[d] = exp(the_element);
                exp_sum[d] += a_pos[d];

            }
        }
    }

    for (int d=0; d<dim; d++) {
        exp_sum[d] = MAX(exp_sum[d], 1e-12);
    }
    
    
    float32_t max_prob[3] = {0,0,0};
    for (int h=0; h<height; h++) {
        auto a_row = heatmaps + h * dim * width;
        for (int w=0; w<width; w++) {
            auto a_pos = a_row + w * dim;
            for (int d=0; d<dim; d++) {
                a_pos[d] /= exp_sum[d];
                if (a_pos[d] > max_prob[d]) {
                    max_prob[d] = a_pos[d];
                }
            }
        }
    }
    
    if (max_prob[0] < hi_prob_thresh) {
        return;
    }
    
    // 2. coordinate transform
    float32_t x_coords[] = {0,0,0};
    float32_t y_coords[] = {0,0,0};

    for (int h=0; h<height; h++) {
        auto X_row = X[h];
        auto Y_row = Y[h];
        auto _row = heatmaps + h * width * dim;
        for (int w=0; w<width; w++) {
            auto x_element = X_row[w];
            auto y_element = Y_row[w];
            auto _ele = _row + w*dim;
            for (int d=0; d<dim; d++) {
                x_coords[d] += (_ele[d] * x_element);
                y_coords[d] += (_ele[d] * y_element);
            }
        }
    }

    for (int d=0; d<dim; d++) {
        auto x_coord = x_coords[d];
        auto y_coord = y_coords[d];

        int x_real_coord = (int)round( (x_coord+1)/2.0 * wanted_input_width);
        int y_real_coord = (int)round( (y_coord+1)/2.0 * wanted_input_height);

        X_coords[d] = x_real_coord;
        Y_coords[d] = y_real_coord;
    }
}

- (void) inferImage:(const cv::Mat &)inputImage
           heatmap1:(cv::Mat&)heatmap1
           heatmap2:(cv::Mat&)heatmap2
           heatmap3:(cv::Mat&)heatmap3
             result:(cv::Mat&)result
          keypoints:(int *)keypoints {
    
    assert(inputImage.rows == wanted_input_height);
    assert(inputImage.cols == wanted_input_width);
    assert(inputImage.channels() == wanted_input_channels);
    assert(inputImage.type() == CV_32FC3);
    
    auto network_input = interpreter->inputs()[0];
    float32_t *network_input_ptr = interpreter->typed_tensor<float32_t>(network_input);
    const float *source_data = (float*) inputImage.data;
    
    std::memcpy(network_input_ptr, source_data, wanted_input_width*wanted_input_height*wanted_input_channels*sizeof(float32_t));
    if (interpreter->Invoke() != kTfLiteOk) {
        LOG(FATAL) << "Failed to invoke!";
    }
    float32_t* network_output = interpreter->typed_output_tensor<float32_t>(0);
    
    auto output_h = 15;
    auto output_w = 20;
    auto output_dim = 3;
    //#type: float32[1,15,20,3]
    
    int X_coords[3] = {0,0,0};
    int Y_coords[3] = {0,0,0};
    
    float32_t heatmaps[output_w*output_h*output_dim];
    std::memcpy(heatmaps, network_output, output_h*output_w*output_dim*sizeof(float32_t));
    
    dsnt(output_h, output_w, heatmaps, X_coords, Y_coords);
    
    //    if (validate_coords(X_coords, Y_coords)) {
    //        // there is a rectangle.
    //        auto lineColor = cv::Scalar(110, 220, 0);
    //        cv::line(input_dummy, cv::Point(X_coords[0], Y_coords[0]), cv::Point(X_coords[1], Y_coords[1]), lineColor, 5);
    //        cv::line(input_dummy, cv::Point(X_coords[1], Y_coords[1]), cv::Point(X_coords[3], Y_coords[3]), lineColor, 5);
    //        cv::line(input_dummy, cv::Point(X_coords[3], Y_coords[3]), cv::Point(X_coords[2], Y_coords[2]), lineColor, 5);
    //        cv::line(input_dummy, cv::Point(X_coords[2], Y_coords[2]), cv::Point(X_coords[0], Y_coords[0]), lineColor, 5);
    //    }{
    //        // there is no rectangle,
    //    }
    
    cv::Mat input_dummy = inputImage.clone();
    auto dotColor = cv::Scalar(110, 220, 0);
    
    cv::circle(input_dummy, cv::Point(X_coords[0], Y_coords[0]), 3, dotColor, -1);
    cv::circle(input_dummy, cv::Point(X_coords[1], Y_coords[1]), 3, dotColor, -1);
    cv::circle(input_dummy, cv::Point(X_coords[2], Y_coords[2]), 3, dotColor, -1);
    
    for (int i=0; i<3; i++) {
        keypoints[i*2] = X_coords[i];
        keypoints[i*2+1] = Y_coords[i];
    }

    cv::Mat dummy = cv::Mat(output_h, output_w, CV_32FC3, heatmaps);
    std::vector<cv::Mat> heatmaps_activations(3);
    cv::split(dummy, heatmaps_activations);

    auto p1 = heatmaps_activations[0];
    auto p2 = heatmaps_activations[1];
    auto p3 = heatmaps_activations[2];

//    cv::Mat heatmaps_sum = p1+p2+p3;
    cv::Mat heatmaps_sum = p1+p2+p3;

    cv::Mat _heatmap;
    cv::Mat grayscale;

    heatmaps_sum.convertTo(grayscale, CV_8UC3 , 255, 0);
//    cv::applyColorMap(grayscale, _heatmap, cv::COLORMAP_JET);

    cv::Mat grayscale1;
    cv::Mat grayscale2;
    cv::Mat grayscale3;
    
    p1.convertTo(grayscale1, CV_8UC3, 255, 0);
    p2.convertTo(grayscale2, CV_8UC3, 255, 0);
    p3.convertTo(grayscale3, CV_8UC3, 255, 0);

    heatmap1 = grayscale1.clone();
    heatmap2 = grayscale2.clone();
    heatmap3 = grayscale3.clone();
    
    cv::Mat image_to_show;
    input_dummy.convertTo(image_to_show, CV_8UC3);
    result = image_to_show.clone();
}

@end
