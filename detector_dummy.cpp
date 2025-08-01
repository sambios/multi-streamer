//
// Created by yuan on 2025/8/1.
//

#include "detector_dummy.h"
#include "otl_ffmpeg.h"

    DummyDetector::DummyDetector(int devId) {

        std::cout << "DummyDetector ctor" << std::endl;

    }

    DummyDetector::~DummyDetector() {
        std::cout << "DummyDetector dtor" << std::endl;
    }

    int DummyDetector::preprocess(std::vector<FrameInfo> &frames) {
        return 0;
    }

    int DummyDetector::forward(std::vector<FrameInfo> &frames) {
        return 0;
    }

    int DummyDetector::postprocess(std::vector<FrameInfo> &frames) {
         // Continue with original postprocessing logic
        for (auto frameInfo : frames) {
            if (m_pfnDetectFinish) {
                m_pfnDetectFinish(frameInfo);
            }

            if (m_nextInferPipe != nullptr) {
                m_nextInferPipe->push_frame(&frameInfo);
            } else {
                // Stop pipeline
                av_packet_unref(frameInfo.pkt);
                av_packet_free(&frameInfo.pkt);

                av_frame_unref(frameInfo.frame);
                av_frame_free(&frameInfo.frame);
            }
        }
        return 0;
    }


std::shared_ptr<Detector> Detector::createDetector(int devId) {
    std::shared_ptr<Detector> detector;
    detector.reset(new DummyDetector(devId));
    return detector;
}