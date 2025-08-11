//
// Created by yuan on 2025/7/31.
//

#ifndef VIDEO_DETECTION_DETECTOR_H
#define VIDEO_DETECTION_DETECTOR_H

#include "otl_pipeline.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include "otl_baseclass.h"

// declare some structures
class Streamer;
struct AVPacket;
struct AVFrame;


struct FrameInfo
{
    int channelId;
    AVPacket *pkt;
    AVFrame *frame;
    std::shared_ptr<Streamer> streamer;
    std::vector<void*> netInputs;
    std::vector<void*> netOutputs;
    otl::Detection detection;
};





class Detector : public otl::DetectorDelegate<FrameInfo>
{
public:
    static std::shared_ptr<Detector> createDetector(int devId);
    virtual ~Detector(){}
};

using DetectorPtr = std::shared_ptr<Detector>;



#endif //VIDEO_DETECTION_DETECTOR_H