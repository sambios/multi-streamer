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
    std::unordered_map<int, DetectorPtr> mMapDetectors;
    std::unordered_map<int, std::shared_ptr<otl::InferencePipe<FrameInfo>> > m_mapInferPipe;
public:
    DeviceManager();
    virtual ~DeviceManager();

    DetectorPtr getDetector(int devId, std::string modelPath);
    std::shared_ptr<otl::InferencePipe<FrameInfo>> getInferPipe(int devId, std::string modelPath);
};

using DeviceManagerPtr = std::shared_ptr<DeviceManager>;


#endif //DEVICE_MANAGER_H