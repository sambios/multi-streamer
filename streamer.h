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
        std::string input_url;        // URL or file path of the video stream
        int frame_drop_interval = 0;   // 0 = no frame drop
        std::string output_url;
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
    bool isRunning() const { return running_; }
    Stats getStats();

protected:
    void onDecodedAVFrame(const AVPacket *pkt, const AVFrame *pFrame) override;

private:
    Config config_;
    std::unique_ptr<otl::StreamDecoder> decoder_;
    std::unique_ptr<otl::FfmpegOutputer> output_;
    std::atomic<bool> running_{false};
    FrameCallback frame_callback_;
    mutable std::mutex mutex_;
    Stats stats_;
};
