//
// Created by yuan on 2025/8/1.
//

#ifndef VIDEO_DETECTION_DETECTOR_DUMMY_H
#define VIDEO_DETECTION_DETECTOR_DUMMY_H

#include "detector.h"

class DummyDetector : public Detector
{
public:
    DummyDetector(int devId = 0, std::string modelPath = "");
    virtual ~DummyDetector();
public:
    int initialize() override;
    int preprocess(std::vector<FrameInfo> &frames) override;
    int forward(std::vector<FrameInfo> &frames) override;
    int postprocess(std::vector<FrameInfo> &frames) override;
};

#endif //VIDEO_DETECTION_DETECTOR_DUMMY_H