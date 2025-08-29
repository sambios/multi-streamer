#include "streamer.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <cstring>
#include <algorithm>

#include "otl_string.h"
#include "stream_sei.h"
extern "C" {
#include <libavutil/rational.h>
#include <libavutil/avutil.h>
}

// Simple overlay utilities for YUV420P frames
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
static void draw_rect_y(uint8_t* y, int w, int h, int linesize, int x1, int y1, int x2, int y2, int thickness, uint8_t yval)
{
    x1 = clampi(x1, 0, w - 1); x2 = clampi(x2, 0, w - 1);
    y1 = clampi(y1, 0, h - 1); y2 = clampi(y2, 0, h - 1);
    if (x1 > x2) std::swap(x1, x2);
    if (y1 > y2) std::swap(y1, y2);
    for (int t = 0; t < thickness; ++t) {
        int yt1 = clampi(y1 + t, 0, h - 1);
        int yt2 = clampi(y2 - t, 0, h - 1);
        // top and bottom
        memset(y + yt1 * linesize + x1, yval, x2 - x1 + 1);
        memset(y + yt2 * linesize + x1, yval, x2 - x1 + 1);
        // left and right
        for (int yy = y1; yy <= y2; ++yy) {
            int yyy = clampi(yy, 0, h - 1);
            if (x1 + t < w) y[yyy * linesize + x1 + t] = yval;
            if (x2 - t >= 0) y[yyy * linesize + x2 - t] = yval;
        }
    }
}

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


    m_detector = m_detectorManager->getDetector(config.devId, config.modelPath);
    m_inferPipe = m_detectorManager->getInferPipe(config.devId, config.modelPath);

    if (m_config.encodeEnabled) {
        std::string codecName = "h264";
        m_encoder = otl::CreateStreamEncoder(codecName);
        otl::EncodeParam p;
        p.codecName = codecName;
        p.width = 1280; p.height = 720;
        p.timeBase = {1, 90000};
        p.frameRate = {30, 1};
        p.pixFmt = AV_PIX_FMT_YUV420P;
        p.gopSize = 60;
        p.maxBFrames = 0;
        p.bitRate = 3'000'000;
        p.preferHardware = true;
        m_encoder->init(&p);
        // set encoder timing for PTS generation
        m_encTimeBase = m_encoder->getTimeBase();
        m_encFrameRate = p.frameRate;
        m_nextPts = 0;
    }

    return true;
}

bool Streamer::start() {
    if (m_running) {
        return false;
    }

    // If we will encode, prepare the output sink early with explicit codec parameters from encoder
    if (m_config.encodeEnabled && m_output == nullptr && m_encoder) {
        const AVCodecParameters* cpar = m_encoder->getCodecParameters();
        if (cpar) {
            AVRational encTb = m_encoder->getTimeBase();
            m_output = std::make_unique<otl::FfmpegOutputer>();
            m_output->openOutputStreamWithCodec(m_config.outputUrl, cpar, encTb);
        }
    }

    AVDictionary *opts=NULL;
    //av_dict_set(&opts, "vf", "scale=640:640:force_original_aspect_ratio=decrease,pad=640:640:(ow-iw)/2:(oh-ih)/2,format=rgb24", 0);
    av_dict_set(&opts, "pp_set", "1920x1080:0:0:3840x2160", 0);
    if (m_decoder->openStream(m_config.inputUrl, true,  opts) != 0) {
        std::cout << "OpenStream " << m_config.inputUrl << " failed!" << std::endl;
        return false;
    }
    av_dict_free(&opts);

    //---- stream operations -----//
    m_decoder->setAvformatOpenedCallback([this](const AVFormatContext* ifmtCtx)
    {
        // For pass-through (no re-encode), derive output stream from input format
        if (!m_config.encodeEnabled) {
            if (m_output == nullptr) {
                m_output = std::make_unique<otl::FfmpegOutputer>();
                m_output->openOutputStream(m_config.outputUrl, ifmtCtx);
            }
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
            if (frameInfo.streamer->m_config.encodeEnabled) {
                // 1) Overlay detection bboxes on Y plane (YUV420P/YUVJ420P) before encoding
                if (frameInfo.frame &&
                    (frameInfo.frame->format == AV_PIX_FMT_YUV420P || frameInfo.frame->format == AV_PIX_FMT_YUVJ420P) &&
                    !frameInfo.detection.bboxes().empty()) {
                    uint8_t* y = frameInfo.frame->data[0];
                    int ls = frameInfo.frame->linesize[0];
                    int W = frameInfo.frame->width;
                    int H = frameInfo.frame->height;
                    for (const otl::Bbox& b : frameInfo.detection.bboxes()) {
                        // If coords look normalized (<=1), scale to pixels
                        auto norm = (b.x2 <= 1.0f && b.y2 <= 1.0f);
                        int x1 = norm ? (int)(b.x1 * W) : (int)b.x1;
                        int y1 = norm ? (int)(b.y1 * H) : (int)b.y1;
                        int x2 = norm ? (int)(b.x2 * W) : (int)b.x2;
                        int y2 = norm ? (int)(b.y2 * H) : (int)b.y2;
                        draw_rect_y(y, W, H, ls, x1, y1, x2, y2, 2, 235);
                    }
                }

                // 2) Assign monotonically increasing PTS in encoder time base and encode
                if (frameInfo.frame) {
                    int64_t step = av_rescale_q(1, av_inv_q(frameInfo.streamer->m_encFrameRate), frameInfo.streamer->m_encTimeBase);
                    if (step <= 0) step = 1;
                    if (frameInfo.frame->pts == AV_NOPTS_VALUE || frameInfo.frame->pts < frameInfo.streamer->m_nextPts) {
                        frameInfo.frame->pts = frameInfo.streamer->m_nextPts;
                    }
                    frameInfo.streamer->m_nextPts = frameInfo.frame->pts + step;
                }
                // Encode frame to packets via vector API
                std::vector<AVPacket*> pkts;
                auto ret = frameInfo.streamer->m_encoder->encode(frameInfo.frame, pkts);
                if (ret == 0) {
                    if (!frameInfo.streamer->m_output) {
                        const AVCodecParameters* cpar = frameInfo.streamer->m_encoder->getCodecParameters();
                        if (cpar) {
                            AVRational encTb = frameInfo.streamer->m_encoder->getTimeBase();
                            frameInfo.streamer->m_output = std::make_unique<otl::FfmpegOutputer>();
                            frameInfo.streamer->m_output->openOutputStreamWithCodec(frameInfo.streamer->m_config.outputUrl, cpar, encTb);
                        }
                    }
                    for (AVPacket* out : pkts) {
                        if (!out) continue;
                        if (out->stream_index < 0) out->stream_index = 0; // single-stream
                        frameInfo.streamer->m_output->inputPacket(out);
                        frameInfo.streamer->m_encoder->freePacket(out);
                    }
                }
                // When encoding path is used, we skip SEI injection (bbox already meant to be fused into image)
                return;
            }

            if (frameInfo.detection.size() > 0)
            {
                auto detect_bbuf = frameInfo.detection.toByteBuffer();
                auto base64_str = otl::base64Enc(detect_bbuf->data(), detect_bbuf->size());
                //std::cout << "SEI:" << base64_str << std::endl;
                // Build SEI NAL unit buffer
                AVPacket* sei_pkt = av_packet_alloc();
                av_packet_copy_props(sei_pkt, frameInfo.pkt);
                // Decide codec and packaging (Annex B vs AVCC)
                AVCodecID codec_id = frameInfo.streamer->get_video_codec_id();
                bool isAnnexb = !frameInfo.streamer->preferAVCC();

                if (codec_id == AV_CODEC_ID_H264)
                {
                    auto packet_size = otl::h264SeiCalcPacketSize((uint32_t)base64_str.length(), isAnnexb, 4);
                    AVBufferRef* buf = av_buffer_alloc(packet_size);
                    int real_size = otl::h264SeiPacketWrite(buf->data, isAnnexb, (uint8_t*)base64_str.data(),
                                                            (uint32_t)base64_str.length());
                    sei_pkt->data = buf->data;
                    sei_pkt->size = real_size;
                    sei_pkt->buf = buf;
                }
                else if (codec_id == AV_CODEC_ID_H265)
                {
                    // Use H.264 calc as a close upper bound, allocate a bit more for 2-byte HEVC header
                    auto base_size = otl::h264SeiCalcPacketSize((uint32_t)base64_str.length(), isAnnexb, 4);
                    AVBufferRef* buf = av_buffer_alloc(base_size + 16);
                    int real_size = otl::h265SeiPacketWrite(buf->data, isAnnexb, (uint8_t*)base64_str.data(),
                                                            (uint32_t)base64_str.length());
                    sei_pkt->data = buf->data;
                    sei_pkt->size = real_size;
                    sei_pkt->buf = buf;
                }
                else
                {
                    // Unknown codec; pass through original packet only
                    av_packet_unref(sei_pkt);
                    av_packet_free(&sei_pkt);
                    frameInfo.streamer->m_output->inputPacket(frameInfo.pkt);
                    return;
                }

                // ensure stream index matches
                sei_pkt->stream_index = frameInfo.pkt->stream_index;

                // Merge SEI and original packet into a single packet (recommended for AVCC/MP4/FLV/RTMP).
                // For Annex B, merging is also safe and keeps SEI preceding the corresponding VCL NAL.
                AVPacket *merged = av_packet_alloc();
                av_packet_copy_props(merged, frameInfo.pkt);
                merged->stream_index = frameInfo.pkt->stream_index;

                // Allocate combined buffer: [SEI][Original]
                int merged_size = sei_pkt->size + frameInfo.pkt->size;
                AVBufferRef *merged_buf = av_buffer_alloc(merged_size);
                memcpy(merged_buf->data, sei_pkt->data, sei_pkt->size);
                memcpy(merged_buf->data + sei_pkt->size, frameInfo.pkt->data, frameInfo.pkt->size);
                merged->data = merged_buf->data;
                merged->size = merged_size;
                merged->buf = merged_buf;

                frameInfo.streamer->m_output->inputPacket(merged);

                av_packet_unref(sei_pkt);
                av_packet_free(&sei_pkt);
                av_packet_unref(merged);
                av_packet_free(&merged);
                // Original pkt will be sent only via merged path
                return;
            }else {
                frameInfo.streamer->m_output->inputPacket(frameInfo.pkt);
            }
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
        if (pFrame->width == 0 || pFrame->height == 0)
        {
            std::cout << "ERROR: width=height=0\n" << std::endl;
        }
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
