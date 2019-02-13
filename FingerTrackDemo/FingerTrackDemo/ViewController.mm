//
//  ViewController.m
//  FingerTrackDemo
//
//  Created by zjcneil on 2019/2/13.
//  Copyright © 2019 zjcneil. All rights reserved.
//

#import "ViewController.h"

#import "TFHelper.h"
#import "OpenCVUtil.h"

#import <opencv2/opencv.hpp>
#import <opencv2/highgui/cap_ios.h>

@interface ViewController () <CvVideoCameraDelegate>

@property (weak, nonatomic) IBOutlet UIImageView *rawView;
@property (weak, nonatomic) IBOutlet UIImageView *processedView;
@property (weak, nonatomic) IBOutlet UILabel *displayView;
@property (nonatomic, strong) CvVideoCamera* videoCamera;
@property (nonatomic, assign) NSTimeInterval lastTime;

@end

#define LOG_CV_MAT_TYPE(mat)
#define VIDEO_SIZE AVCaptureSessionPreset1280x720
#define HW_RATIO (3/4.0)

@implementation ViewController

- (CvVideoCamera *)videoCamera {
    if (!_videoCamera) {
        _videoCamera = [[CvVideoCamera alloc] initWithParentView:nil];
        _videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
        _videoCamera.defaultAVCaptureSessionPreset = VIDEO_SIZE;
        _videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
        _videoCamera.rotateVideo = YES;
        _videoCamera.defaultFPS = 30;
        _videoCamera.grayscaleMode = NO;
        _videoCamera.delegate = self;
    }
    return _videoCamera;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    
    CGFloat containerViewWidth = self.view.frame.size.width;
    CGFloat left_padding = 80;
    CGFloat top_padding = 25;
    
    CGFloat imageViewWidth = (containerViewWidth - left_padding*2);
    
    self.rawView.frame = CGRectMake(left_padding, top_padding, imageViewWidth, imageViewWidth*HW_RATIO);
    self.processedView.frame = CGRectMake(left_padding, top_padding*2+imageViewWidth*HW_RATIO, imageViewWidth, imageViewWidth*HW_RATIO);
}

- (void)viewWillAppear:(BOOL)animated {
    [self startCapture];
}

- (void)viewWillDisappear:(BOOL)animated {
    [self stopCapture];
}

- (void)startCapture {
    self.lastTime = [[NSDate date] timeIntervalSince1970];
    [self.videoCamera start];
}

- (void)stopCapture {
    [self.videoCamera stop];
}

- (void)processImage:(cv::Mat&)bgraImage {
    cv::Mat& rawBgraImage = bgraImage;
    LOG_CV_MAT_TYPE(rawBgraImage);
    assert(rawBgraImage.type() == CV_8UC4);
    
    int origin_height = rawBgraImage.rows;
    int origin_width = rawBgraImage.cols;
    
    //     1. crop
    cv::Rect bounds(0,0,origin_width,origin_height);
    cv::Rect crop;
    if (1.0*origin_height/origin_width > HW_RATIO) { // crop the vertical direction.
        auto height_cropped = origin_width * HW_RATIO;
        auto top_padding = (int)(origin_height - height_cropped)/2;
        auto _x = cv::Rect(0,top_padding,origin_width, height_cropped);
        crop = _x;
    }else { // crop the horizontal direction.
        auto width_cropped = origin_height / HW_RATIO;
        auto left_padding = (int)(origin_width - width_cropped)/2;
        auto _x = cv::Rect(left_padding,0,width_cropped,origin_height);
        crop = _x;
    }
    auto cropped_Image = rawBgraImage(bounds & crop);
    
    // 2. resize.
    int height = 480;
    int width = 640;
    cv::Size size(width, height);
    cv::Mat resized_image;
    cv::resize(cropped_Image, resized_image, size, 0, 0, cv::INTER_LINEAR);
    //clip
    
    cv::Mat rgb_image;
    cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGRA2RGB);
    LOG_CV_MAT_TYPE(rgb_image);
    assert(rgb_image.type() == CV_8UC3);
    
    cv::Mat float_rgbImage;
    rgb_image.convertTo(float_rgbImage, CV_32FC3);
    LOG_CV_MAT_TYPE(float_rgbImage);
    
    NSTimeInterval startTime = [[NSDate date] timeIntervalSince1970];
    
    cv::Mat result_image;
    cv::Mat heatmap;
    
    
    [[TFHelper sharedInstance] inferImage:float_rgbImage heatmap:heatmap];
    
    NSTimeInterval current_time = [[NSDate date] timeIntervalSince1970];
    NSTimeInterval total_inference_time = current_time - startTime;
    
    NSUInteger FPS = (NSUInteger) 1.0/(current_time - self.lastTime);
    
    NSString *debugInfo = [NSString stringWithFormat:@"%.3f second \n FPS: %lu", total_inference_time, (unsigned long)FPS];

//    UIImage *image1 = [OpenCVUtil UIImageFromCVMat:result_image];
    UIImage *image2 = [OpenCVUtil UIImageFromCVMat:heatmap];
    
    UIImage *resized = [OpenCVUtil UIImageFromCVMat:resized_image];
    self.lastTime = current_time;
    dispatch_async(dispatch_get_main_queue(), ^{
        self.displayView.text = debugInfo;
        [self.rawView setImage:resized];
        [self.processedView setImage:image2];
    });
}

@end
