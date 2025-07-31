#include "streamer.h"
#include <iostream>
#include <chrono>

Streamer::Streamer() {
    m_fpsStat = otl::StatTool::create(5);
}

Streamer::~Streamer() {
    stop();
}


bool Streamer::init(const Config& config) {
    m_config = config;
    m_decoder = std::make_unique<otl::StreamDecoder>(config.id);
    m_decoder->setObserver(this);
    if (m_decoder->openStream(config.inputUrl) != 0) {
        std::cout << "OpenStream " << config.inputUrl << " failed!" << std::endl;
        return false;
    }

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

    if (m_output) {
        m_output->inputPacket(pkt);
    }
}
