#pragma once

#include <string>
#include <thread>
#include <atomic>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <opencv2/opencv.hpp>

#include "otl_timer.h"
#include "stream_decode.h"
#include "stream_pusher.h"

class Streamer : public otl::StreamDecoderEvents {
public:
    struct Config {
        int id;
        std::string inputUrl;        // URL or file path of the video stream
        int frameDropInterval = 0;   // 0 = no frame drop
        std::string outputUrl;
    };

    otl::StatToolPtr m_fpsStat;
    struct Stats {
       double fps;
    };

    using FrameCallback = std::function<void(const AVPacket *pkt, const AVFrame *pFrame)>;
    
    Streamer();
    ~Streamer();
    
    // Non-copyable
    Streamer(const Streamer&) = delete;
    Streamer& operator=(const Streamer&) = delete;

    bool init(const Config& config);
    bool start(FrameCallback callback);
    void stop();
    bool isRunning() const { return m_running; }
    Stats getStats();

protected:
    void onDecodedAVFrame(const AVPacket *pkt, const AVFrame *pFrame) override;

private:
    Config m_config;
    std::unique_ptr<otl::StreamDecoder> m_decoder;
    std::unique_ptr<otl::FfmpegOutputer> m_output;
    std::atomic<bool> m_running{false};
    FrameCallback m_frameCallback;
    mutable std::mutex m_mutex;
    Stats m_stats;
};
