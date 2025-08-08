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
#include <numeric>
#include <mutex>

class YoloDetector : public Detector
{
public:
    YoloDetector(int devId = 0);
    virtual ~YoloDetector();
public:
    int initialize() override;
    int preprocess(std::vector<FrameInfo> &frames) override;
    int forward(std::vector<FrameInfo> &frames) override;
    int postprocess(std::vector<FrameInfo> &frames) override;

private:
    // TopsInference components (following yolov5_ref.cpp pattern)
    TopsInference::IEngine* m_engine;
    TopsInference::handler_t m_handler;

    // Model parameters
    int m_deviceId;
    int m_inputWidth;
    int m_inputHeight;
    float m_confThreshold;
    float m_nmsThreshold;

    // Engine and model related members
    std::string m_modelPath;

    // Shape information (following yolov5_ref.cpp pattern)
    struct ShapeInfo {
        std::string name;
        std::vector<int> dims;
        int dtype;
        int dtype_size;
        int volume;
        int mem_size;
        ShapeInfo() {}
        ShapeInfo(const char *tensor_name, std::vector<int> &_dims, int dtype, int _dtype_size, int _mem_size)
                : name(tensor_name), dims(_dims), dtype(dtype), dtype_size(_dtype_size), mem_size(_mem_size) {
            volume = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
        }
    };
    std::vector<ShapeInfo> m_inputShapes;
    std::vector<ShapeInfo> m_outputShapes;

    // 互斥锁用于多线程同步
    std::mutex m_mutex;

    std::vector<ShapeInfo> getInputsShape();
    std::vector<ShapeInfo> getOutputsShape();
    std::vector<void*> allocHostMemory(std::vector<ShapeInfo> &shapes_info, int times, bool verbose);
    void freeHostMemory(std::vector<void*> &datum);
    int get_dtype_size(TopsInference::DataType dtype);
    void cleanup();
};

using YoloDetectorPtr = std::shared_ptr<YoloDetector>;


#endif //VIDEO_DETECTION_DETECTOR_YOLO_H