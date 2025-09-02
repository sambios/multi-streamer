#include <iostream>
#include <signal.h>
#include <atomic>
#include <fstream>
#include <filesystem>


#include <nlohmann/json.hpp>

#include "device_manager.h"
#include "otl.h"
#include "streamer.h"

// 使用 nlohmann/json 库
using json = nlohmann::json;

static std::atomic<bool> g_running{true};

static void signalHandler(int signum) {
    g_running = false;
}

int main(int argc, char* argv[]) {

    otl::log::LogConfig logConfig;
    logConfig.enableConsole = true;
    logConfig.level = otl::log::LOG_TRACE;
    otl::log::init(logConfig);

    otl::TimerQueuePtr timerQueue = otl::TimerQueue::create();

    // 解析 config.json 文件
    std::string configFilePath;
    if (std::filesystem::exists("config.json")) {
        configFilePath = "config.json";
    }else if (std::filesystem::exists("../../multi-streamer/config.json")) {
        configFilePath = "../../multi-streamer/config.json";
    }

    std::ifstream configFile(configFilePath);
    if (!configFile.is_open()) {
        std::cerr << "Failed to open config.json" << std::endl;
        return -1;
    }

    json configsJson;
    try {
        configFile >> configsJson;
    } catch (const json::exception& e) {
        std::cerr << "Error parsing config.json: " << e.what() << std::endl;
        return -1;
    }

    std::vector<std::shared_ptr<Streamer>> streamers;
    // 从 JSON 对象中提取配置信息
    int channelId = 0;
    int card_nums = configsJson["dev_num"].get<int>();
    bool detect_enabled = configsJson["detect_enabled"].get<bool>();
    std::string model_path = otl::replaceHomeDirectory(configsJson["model_path"].get<std::string>());
    std::string pp_str = configsJson["ppset_info"].get<std::string>();
    bool ppset_enabled = configsJson["ppset_enabled"].get<bool>();
    float pp_scale = configsJson["pp_scale"].get<float>();

    for (int devId = 0; devId < card_nums; ++devId) {

        // 读取卡的运行配置
        std::vector<Streamer::Config> configs;
        std::string devNodeStr = otl::formatString("configs_dev%d", devId);
        
        for (const auto& configObj : configsJson[devNodeStr]) {
            // 获取基础配置
            std::string inputUrl = otl::replaceHomeDirectory(configObj["input_url"].get<std::string>());
            int frameDropInterval = configObj["frame_drop_interval"].get<int>();
            std::string outputUrlBase = otl::replaceHomeDirectory(configObj["output_url_base"].get<std::string>());
            int repeatCount = configObj["client_count"].get<int>();
            
            // 解析基础输出URL以提取IP和端口
            std::string baseIp;
            int basePort;
            std::string protocol;
            
            // 简单的URL解析 (假设格式为 udp://ip:port)
            size_t protocolPos = outputUrlBase.find("://");
            if (protocolPos != std::string::npos) {
                protocol = outputUrlBase.substr(0, protocolPos + 3);
                std::string remaining = outputUrlBase.substr(protocolPos + 3);
                
                size_t colonPos = remaining.find_last_of(':');
                if (colonPos != std::string::npos) {
                    baseIp = remaining.substr(0, colonPos);
                    basePort = std::stoi(remaining.substr(colonPos + 1));
                } else {
                    // 如果没有端口，使用默认端口
                    baseIp = remaining;
                    basePort = 9000;
                }
            } else {
                // 如果格式不正确，使用默认值
                protocol = "udp://";
                baseIp = "127.0.0.1";
                basePort = 9000;
            }
            
            // 根据重复次数创建多个配置
            for (int i = 0; i < repeatCount; ++i) {
                Streamer::Config config;
                config.devId = OTL_MAKE_INT32(devId, i);
                config.channelId = channelId++;
                config.inputUrl = inputUrl;
                config.frameDropInterval = frameDropInterval;
                config.decodeId = OTL_MAKE_INT32(devId, i % 2);
                config.detectEnabled = detect_enabled;
                config.encodeEnabled = false;
                config.modelPath = model_path;
                config.pp_str = pp_str;
                config.ppset_enabled = ppset_enabled;
                config.pp_scale = pp_scale;
                
                // 生成递增的输出URL
                config.outputUrl = protocol + baseIp + ":" + std::to_string(basePort + i);
                
                configs.push_back(config);
                
                std::cout << "Generated config - Channel " << config.channelId 
                         << ": " << config.inputUrl << " -> " << config.outputUrl << std::endl;
            }
        }

        //开始启动PIPE，一个卡一个PIPE实例
        DeviceManagerPtr detectorManager = std::make_shared<DeviceManager>();

        // 初始化并启动流
        for (const auto& config : configs) {
            std::shared_ptr<Streamer> streamer = std::make_shared<Streamer>(detectorManager);
            if (!streamer->init(config)) {
                std::cerr << "Failed to initialize streamer" << std::endl;
                return -1;
            }

            streamer->start();

            streamers.push_back(streamer);
        }
    }



    // 等待用户输入以停止
    uint64_t timerId;
    timerQueue->createTimer(1000, 1, [streamers] (){
        double total = 0.0;
        for (auto strm : streamers) {
            auto stats = strm->getStats();
            total += stats.fps;
            auto stat = strm->getPipeStatus();
            OTL_LOGI("video_detection", "%d:(%d/%d %.2f, %d/%d %.2f, %d/%d %.2f)", strm->getChannelId(),
                stat.preprocess_queue_current, stat.preprocess_queue_size, stat.preprocess_fps,
                stat.forward_queue_current, stat.forward_queue_size, stat.forward_fps,
                stat.postprocess_queue_current, stat.postprocess_queue_size, stat.postprocess_fps);
        }

        std::cout << otl::formatString("total:ch-num=%d FPS=%.2f\n", streamers.size(), total);
        std::cout.flush();
    }, -1, &timerId);

    timerQueue->runLoop();


    return 0;
}
