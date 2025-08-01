#include "streamer.h"
#include <iostream>
#include <chrono>

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

    if (m_decoder->openStream(config.inputUrl) != 0) {
        std::cout << "OpenStream " << config.inputUrl << " failed!" << std::endl;
        return false;
    }

    //---- stream operations -----//
    m_decoder->setAvformatOpenedCallback([this, config](const AVFormatContext* ifmtCtx)
    {
        if (m_output == nullptr) {
            m_output = std::make_unique<otl::FfmpegOutputer>();
            m_output->openOutputStream(config.outputUrl, ifmtCtx);
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
        if (frameInfo.streamer && frameInfo.streamer->m_output) {
            // 检测完毕，根据检测结果，上传视频流
            frameInfo.streamer->m_output->inputPacket(frameInfo.pkt);
        }
    });


    return true;
}

bool Streamer::start(FrameCallback callback) {
    if (m_running) {
        return false;
    }

    m_frameCallback = std::move(callback);
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
    m_fpsStat->update();

    if (m_frameCallback) m_frameCallback(pkt, pFrame);

    FrameInfo frame;
    frame.pkt = av_packet_alloc();
    av_packet_copy_props(frame.pkt, pkt);
    //printf("%d, pts = %d\n", m_config.id, pkt->pts);
    av_packet_ref(frame.pkt, pkt);

    frame.frame = av_frame_alloc();
    av_frame_ref(frame.frame, pFrame);

    frame.streamer = get_shared_ptr();

    m_inferPipe->push_frame(&frame);
}
