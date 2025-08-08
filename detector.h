//
// Created by yuan on 2025/7/31.
//

#ifndef VIDEO_DETECTION_DETECTOR_H
#define VIDEO_DETECTION_DETECTOR_H

#include "otl_pipeline.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <bits/regex_constants.h>

// declare some structures
class Streamer;
struct AVPacket;
struct AVFrame;

struct Serialable
{
    virtual void fromByteBuffer(otl::ByteBuffer *buf) = 0;
    virtual std::shared_ptr<otl::ByteBuffer> toByteBuffer() = 0;
    virtual ~Serialable() = default;
};

struct Bbox
{
    int classId;
    float confidence;
    cv::Rect rect;
};

class YoloDetection : public Serialable {
        std::vector<Bbox> m_bboxes;
    int8_t m_type;
public:
        YoloDetection()
        {
            m_type = 0;
        }

        std::shared_ptr<otl::ByteBuffer> toByteBuffer() override
        {
            std::shared_ptr<otl::ByteBuffer> buf = std::make_shared<otl::ByteBuffer>();
            buf->push_back(m_type);
            buf->push_back(m_bboxes.size());
            for (auto o : m_bboxes)
            {
                buf->push_back(o.rect.x);
                buf->push_back(o.rect.y);
                buf->push_back(o.rect.x + o.rect.width);
                buf->push_back(o.rect.y + o.rect.height);
                buf->push_back(o.confidence);
                buf->push_back(o.classId);
            }

            return buf;
        }

        void fromByteBuffer(otl::ByteBuffer* buf) override
        {
            int8_t type;
            buf->pop_front(type);
            m_type = type;

            uint32_t size = 0;
            buf->pop_front(size);
            for (int i = 0; i < size; ++i)
            {
                Bbox o;
                int x1, y1, x2, y2;
                buf->pop_front(x1);
                buf->pop_front(y1);
                buf->pop_front(x2);
                buf->pop_front(y2);
                o.rect = cv::Rect(x1, y1, x2 - x1, y2 - y1);
                buf->pop_front(o.confidence);
                buf->pop_front(o.classId);
                m_bboxes.push_back(o);
            }
        }

        void clear()
        {
            m_bboxes.clear();
        }

        void push_back(const Bbox& box)
        {
            m_bboxes.push_back(box);
        }

        size_t size()
        {
            return m_bboxes.size();
        }

    };


struct FrameInfo
{
    int channelId;
    AVPacket *pkt;
    AVFrame *frame;
    std::shared_ptr<Streamer> streamer;
    std::vector<void*> netInputs;
    std::vector<void*> netOutputs;
    YoloDetection detection;
};





class Detector : public otl::DetectorDelegate<FrameInfo>
{
public:
    static std::shared_ptr<Detector> createDetector(int devId);
    virtual ~Detector(){}
};

using DetectorPtr = std::shared_ptr<Detector>;



#endif //VIDEO_DETECTION_DETECTOR_H