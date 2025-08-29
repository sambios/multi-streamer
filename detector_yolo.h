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
    YoloDetector(int devId = 0, std::string modelPath = "");
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
    void allocHostMemory(FrameInfo &info, bool isInput, std::vector<ShapeInfo> &shapes_info, int times, bool verbose);
    void freeHostMemory(FrameInfo &info, bool isInput);

    void syncInputs(FrameInfo& info, bool isH2D = true)
    {
        if (isH2D)
        {
            for (int i = 0;i < info.netHostInputs.size(); ++i)
            {
                auto ret = topsMemcpyHtoD(info.netDeviceInputs[i], info.netHostInputs[i], info.netInputsSize[i]);
                assert(topsSuccess == ret);
            }
        }else {
            for (int i = 0;i < info.netHostInputs.size(); ++i)
            {
                auto ret = topsMemcpyDtoH(info.netHostInputs[i], info.netDeviceInputs[i], info.netInputsSize[i]);
                assert(topsSuccess == ret);
            }
        }
    }

    void syncOutputs(FrameInfo& info, bool isD2H = true)
    {
        if (isD2H)
        {
            for (int i = 0;i < info.netHostOutputs.size(); ++i)
            {
                auto ret = topsMemcpyDtoH(info.netHostOutputs[i], info.netDeviceOutputs[i], info.netOutputsSize[i]);
                assert(topsSuccess == ret);
            }
        }else {
            for (int i = 0;i < info.netHostOutputs.size(); ++i)
            {
                auto ret = topsMemcpyHtoD(info.netDeviceOutputs[i], info.netHostInputs[i], info.netOutputsSize[i]);
                assert(topsSuccess == ret);
            }
        }
    }

    int get_dtype_size(TopsInference::DataType dtype);
    void cleanup();
};

using YoloDetectorPtr = std::shared_ptr<YoloDetector>;


#endif //VIDEO_DETECTION_DETECTOR_YOLO_H