//
// Created by yuan on 2025/7/31.
//

#ifndef VIDEO_DETECTION_DETECTOR_H
#define VIDEO_DETECTION_DETECTOR_H

#include "otl_pipeline.h"

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
};

class YoloDetector : public otl::DetectorDelegate<FrameInfo>
{
public:
    YoloDetector(int devId = 0);
    virtual ~YoloDetector();
public:
    int preprocess(std::vector<FrameInfo> &frames) override;
    int forward(std::vector<FrameInfo> &frames) override;
    int postprocess(std::vector<FrameInfo> &frames) override;
};

using YoloDetectorPtr = std::shared_ptr<YoloDetector>;


#endif //VIDEO_DETECTION_DETECTOR_H