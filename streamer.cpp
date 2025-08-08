#include "streamer.h"
#include <iostream>
#include <chrono>

#include "otl_string.h"
#include "stream_sei.h"

Streamer::Streamer(DeviceManagerPtr ptr) {
    m_fpsStat = otl::StatTool::create(5);
    m_detectorManager = ptr;
}

Streamer::~Streamer() {
    stop();
}


bool Streamer::init(const Config& config) {
    m_config = config;
    m_decoder = std::make_unique<otl::StreamDecoder>(config.decodeId);
    m_decoder->setObserver(this);


    m_detector = m_detectorManager->getDetector(config.devId);
    m_inferPipe = m_detectorManager->getInferPipe(config.devId);



    return true;
}

bool Streamer::start() {
    if (m_running) {
        return false;
    }

    if (m_decoder->openStream(m_config.inputUrl) != 0) {
        std::cout << "OpenStream " << m_config.inputUrl << " failed!" << std::endl;
        return false;
    }

    //---- stream operations -----//
    m_decoder->setAvformatOpenedCallback([this](const AVFormatContext* ifmtCtx)
    {
        if (m_output == nullptr) {
            m_output = std::make_unique<otl::FfmpegOutputer>();
            m_output->openOutputStream(m_config.outputUrl, ifmtCtx);
        }
    });

    m_decoder->setAvformatClosedCallback([this]()
    {
        if (m_output != nullptr) {
            m_output->closeOutputStream();
        }
    });


    //------- Delegate callback --------//
    m_detector->set_detected_callback([](FrameInfo& frameInfo)
    {
        if (frameInfo.streamer && frameInfo.streamer->m_output)
        {
            if (frameInfo.detection.size() > 0)
            {
                auto detect_bbuf = frameInfo.detection.toByteBuffer();
                auto base64_str = otl::base64Enc(detect_bbuf->data(), detect_bbuf->size());
                AVPacket* sei_pkt = av_packet_alloc();

                av_packet_copy_props(sei_pkt, frameInfo.pkt);

                //AVCodecID codec_id = frameInfo.streamer->get_video_codec_id();

                //if (codec_id == AV_CODEC_ID_H264)
                {
                    auto packet_size = otl::h264SeiCalcPacketSize(base64_str.length());
                    AVBufferRef* buf = av_buffer_alloc(packet_size << 1);
                    //assert(packet_size < 16384);
                    int real_size = otl::h264SeiPacketWrite(buf->data, true, (uint8_t*)base64_str.data(),
                                                            base64_str.length());
                    sei_pkt->data = buf->data;
                    sei_pkt->size = real_size;
                    sei_pkt->buf = buf;
                }

                frameInfo.streamer->m_output->inputPacket(sei_pkt);
                frameInfo.streamer->m_output->inputPacket(frameInfo.pkt);

                av_packet_unref(sei_pkt);
                av_packet_free(&sei_pkt);
            }
            frameInfo.streamer->m_output->inputPacket(frameInfo.pkt);
        }
    });

    m_running = true;
    return true;
}

void Streamer::stop() {
    m_running = false;
    m_decoder->closeStream();
}

Streamer::Stats Streamer::getStats() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_stats.fps = m_fpsStat->getSpeed();
    return m_stats;
}

void Streamer::onDecodedAVFrame(const AVPacket* pkt, const AVFrame* pFrame) {
    //1. statistic
    m_fpsStat->update();

    if (m_config.detectEnabled) {
        // 2. Post Frame to queue
        FrameInfo frame;
        frame.pkt = av_packet_alloc();
        av_packet_copy_props(frame.pkt, pkt);
        av_packet_ref(frame.pkt, pkt);

        frame.frame = av_frame_alloc();
        av_frame_ref(frame.frame, pFrame);

        frame.streamer = get_shared_ptr();

        m_inferPipe->push_frame(&frame);
    }else {
        //3. directly output
        if (m_output)
            m_output->inputPacket(pkt);
    }
}
