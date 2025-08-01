//
// Created by yuan on 2025/8/1.
//

#ifndef VIDEO_DETECTION_DETECTOR_YOLO_H
#define VIDEO_DETECTION_DETECTOR_YOLO_H

#include "detector.h"
#include "otl_pipeline.h"
#include "TopsInference/TopsInferRuntime.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>

class YoloDetector : public Detector
{
public:
    YoloDetector(int devId = 0);
    virtual ~YoloDetector();
public:
    int preprocess(std::vector<FrameInfo> &frames) override;
    int forward(std::vector<FrameInfo> &frames) override;
    int postprocess(std::vector<FrameInfo> &frames) override;

private:
    // TopsInference components
    TopsInference::IEngine* m_engine;
    TopsInference::handler_t m_handler;

    // Model parameters
    int m_deviceId;
    int m_inputWidth;
    int m_inputHeight;
    int m_numClasses;
    float m_confThreshold;
    float m_nmsThreshold;

    // Preprocessed data storage
    std::vector<cv::Mat> m_preprocessedImages;
    std::vector<float> m_inputData;
    std::vector<float> m_outputData;

    // Helper methods
    bool initializeEngine(const std::string& modelPath);
    cv::Mat preprocessImage(const cv::Mat& image);
    std::vector<cv::Rect> postprocessDetections(const float* output, int imageWidth, int imageHeight);
    void cleanup();
};

using YoloDetectorPtr = std::shared_ptr<YoloDetector>;


#endif //VIDEO_DETECTION_DETECTOR_YOLO_H