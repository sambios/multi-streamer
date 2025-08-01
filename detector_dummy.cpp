//
// Created by yuan on 2025/8/1.
//

#include "detector_dummy.h"


    DummyDetector::DummyDetector(int devId) {

    }

    DummyDetector::~DummyDetector() {

    }

    int DummyDetector::preprocess(std::vector<FrameInfo> &frames) {
        return 0;
    }

    int DummyDetector::forward(std::vector<FrameInfo> &frames) {
        return 0;
    }

    int DummyDetector::postprocess(std::vector<FrameInfo> &frames) {
        return 0;
    }
