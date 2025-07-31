//
// Created by yuan on 2025/7/31.
//

#include "detector.h"
#include "otl_ffmpeg.h"

YoloDetector::YoloDetector(int id) {
    std::cout << "YoloDetector: ctor" << std::endl;

}

YoloDetector::~YoloDetector() {
    std::cout << "YoloDetector:dtor" << std::endl;
}


int YoloDetector::preprocess(std::vector<FrameInfo>& frames) {
    //std::cout << __func__ << std::endl;
    return 0;
}

int YoloDetector::forward(std::vector<FrameInfo>& frames) {
    //std::cout << __func__ << std::endl;
    return 0;
}

int YoloDetector::postprocess(std::vector<FrameInfo>& frames) {

    for (auto frameInfo : frames) {
        if (m_pfnDetectFinish) {
            m_pfnDetectFinish(frameInfo);
        }

        if (m_nextInferPipe != nullptr) {
            m_nextInferPipe->push_frame(&frameInfo);
        }else {
            // Stop pipeline
            av_packet_unref(frameInfo.pkt);
            av_packet_free(&frameInfo.pkt);

            av_frame_unref(frameInfo.frame);
            av_frame_free(&frameInfo.frame);
        }
    }




    return 0;
}
