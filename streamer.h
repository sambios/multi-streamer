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
#include "stream_decode_hw.h"
#include "stream_pusher.h"
#include "device_manager.h"
#include "detector.h"


class Streamer : public otl::StreamDecoderEvents, public std::enable_shared_from_this<Streamer>{
public:
    struct Config {
        int devId;
        int channelId;
        int decodeId;
        std::string inputUrl;        // URL or file path of the video stream
        int frameDropInterval = 0;   // 0 = no frame drop
        std::string outputUrl;
        bool detectEnabled;
    };

    otl::StatToolPtr m_fpsStat;
    struct Stats {
       double fps;
    };

    using FrameCallback = std::function<void(const AVPacket *pkt, const AVFrame *pFrame)>;
    
    Streamer(DeviceManagerPtr ptr);
    ~Streamer();
    
    // Non-copyable
    Streamer(const Streamer&) = delete;
    Streamer& operator=(const Streamer&) = delete;

    bool init(const Config& config);
    bool start();
    void stop();
    bool isRunning() const { return m_running; }
    Stats getStats();

    inline std::shared_ptr<Streamer> get_shared_ptr() {
        // 通过 shared_from_this() 生成指向自身的 shared_ptr
        return shared_from_this();
    }

    int getChannelId() {
        return m_config.channelId;
    }

    int getDevId() {
        return m_config.devId;
    }

protected:
    void onDecodedAVFrame(const AVPacket *pkt, const AVFrame *pFrame) override;

private:
    Config m_config;
    std::unique_ptr<otl::StreamDecoder> m_decoder;
    std::unique_ptr<otl::FfmpegOutputer> m_output;
    std::shared_ptr<Detector> m_detector;
    std::shared_ptr<otl::InferencePipe<FrameInfo>> m_inferPipe;
    std::atomic<bool> m_running{false};
    DeviceManagerPtr m_detectorManager;
    mutable std::mutex m_mutex;
    Stats m_stats;

};
