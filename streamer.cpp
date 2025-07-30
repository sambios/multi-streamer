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
    config_ = config;
    decoder_ = std::make_unique<otl::StreamDecoder>(config.id);
    decoder_->setObserver(this);
    if (decoder_->openStream(config.input_url) != 0) {
        std::cout << "OpenStream " << config.input_url << " failed!" << std::endl;
        return false;
    }

    decoder_->setAvformatOpenedCallback([this, config](const AVFormatContext* ifmtCtx)
    {
        if (output_ == nullptr) {
            output_ = std::make_unique<otl::FfmpegOutputer>();
            output_->openOutputStream(config.output_url, ifmtCtx);
        }
    });

    decoder_->setAvformatClosedCallback([this]()
    {
        if (output_ != nullptr) {
            output_->closeOutputStream();
        }
    });

    return true;
}

bool Streamer::start(FrameCallback callback) {
    if (running_) {
        return false;
    }

    frame_callback_ = std::move(callback);
    running_ = true;
    return true;
}

void Streamer::stop() {
    running_ = false;
    decoder_->closeStream();
}

Streamer::Stats Streamer::getStats() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.fps = m_fpsStat->getSpeed();
    return stats_;
}

void Streamer::onDecodedAVFrame(const AVPacket* pkt, const AVFrame* pFrame) {
    m_fpsStat->update();

    if (frame_callback_) frame_callback_(pkt, pFrame);

    if (output_) {
        output_->inputPacket(pkt);
    }
}
