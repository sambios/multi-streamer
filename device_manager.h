//
// Created by yuan on 2025/7/31.
//

#ifndef DEVICE_MANAGER_H
#define DEVICE_MANAGER_H

#include "detector.h"
#include "otl_pipeline.h"

class FrameInfo;

class DeviceManager
{
    std::unordered_map<int, YoloDetectorPtr> mMapDetectors;
    std::unordered_map<int, std::shared_ptr<otl::InferencePipe<FrameInfo>> > m_mapInferPipe;
public:
    DeviceManager();
    virtual ~DeviceManager();

    YoloDetectorPtr getDetector(int devId);
    std::shared_ptr<otl::InferencePipe<FrameInfo>> getInferPipe(int devId);
};

using DeviceManagerPtr = std::shared_ptr<DeviceManager>;


#endif //DEVICE_MANAGER_H