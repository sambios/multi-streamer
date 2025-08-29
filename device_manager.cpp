//
// Created by yuan on 2025/7/31.
//

#include "device_manager.h"

DeviceManager::DeviceManager() {
    std::cout << "DeviceManager ctor" << std::endl;
}

DeviceManager::~DeviceManager() {
    std::cout << "DeviceManager dtor" << std::endl;
}


DetectorPtr DeviceManager::getDetector(int devId, std::string modelPath) {
    if (mMapDetectors.find(devId) != mMapDetectors.end()) {
        return mMapDetectors[devId];
    }

    //DetectorPtr detector = std::make_shared<Detector>(devId);
    DetectorPtr detector = Detector::createDetector(devId, modelPath);
    mMapDetectors[devId] = detector;

    return detector;
}

std::shared_ptr<otl::InferencePipe<FrameInfo>> DeviceManager::getInferPipe(int devId, std::string modelPath) {
    // 查找已经创建的
    if (m_mapInferPipe.find(devId) != m_mapInferPipe.end()) {
        return m_mapInferPipe[devId];
    }

    // 重新创建新的实例
    auto detector = getDetector(devId, modelPath);
    otl::DetectorParam param;
    param.batch_num = 1;
    param.inference_thread_num = 1;
    param.preprocess_thread_num = 8;
    param.postprocess_thread_num = 8;

    std::shared_ptr<otl::InferencePipe<FrameInfo>>  inferPipe = std::make_shared<otl::InferencePipe<FrameInfo>>();
    inferPipe->init(param, detector);
    m_mapInferPipe[devId] = inferPipe;

    return inferPipe;
}