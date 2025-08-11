#pragma once

#include <string>
#include <thread>
#include <atomic>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <opencv2/opencv.hpp>
#include <cstring>   // strlen
#include <strings.h> // strcasecmp, strncasecmp

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

    // Expose video codec id for decisions in callbacks
    AVCodecID get_video_codec_id() {
        return m_decoder ? m_decoder->getVideoCodecId() : AV_CODEC_ID_NONE;
    }

    // Decide whether output prefers AVCC (length-prefixed) rather than Annex B.
    // Heuristic based on output URL/extension: MP4/MOV/FLV/RTMP typically use AVCC.
    bool preferAVCC() const {
        auto &url = m_config.outputUrl;
        auto ends_with = [&](const char* suf){
            size_t n = strlen(suf);
            if (url.size() < n) return false;
            return strcasecmp(url.c_str() + url.size() - n, suf) == 0;
        };
        auto starts_with = [&](const char* pre){
            size_t n = strlen(pre);
            if (url.size() < n) return false;
            return strncasecmp(url.c_str(), pre, n) == 0;
        };
        if (ends_with(".mp4") || ends_with(".mov") || ends_with(".m4v") ||
            ends_with(".flv") || starts_with("rtmp://")) {
            return true;
        }
        return false; // default to Annex B (e.g., MPEG-TS)
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
