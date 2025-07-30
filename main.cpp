#include <iostream>
#include <signal.h>
#include <atomic>
#include <fstream>
#include <nlohmann/json.hpp>

#include "otl_timer.h"
#include "streamer.h"

// 使用 nlohmann/json 库
using json = nlohmann::json;

static std::atomic<bool> g_running{true};

static void signalHandler(int signum) {
    g_running = false;
}

int main(int argc, char* argv[]) {

    otl::TimerQueuePtr tqp = otl::TimerQueue::create();

    // 解析 config.json 文件
    std::ifstream config_file("config.json");
    if (!config_file.is_open()) {
        std::cerr << "Failed to open config.json" << std::endl;
        return -1;
    }

    json configs_json;
    try {
        config_file >> configs_json;
    } catch (const json::exception& e) {
        std::cerr << "Error parsing config.json: " << e.what() << std::endl;
        return -1;
    }

    // 从 JSON 对象中提取配置信息
    int i = 0;
    std::vector<Streamer::Config> configs;
    for (const auto& config_obj : configs_json["configs"]) {
        Streamer::Config config;
        config.id = i++;
        config.input_url = config_obj["input_url"].get<std::string>();
        config.output_url = config_obj["output_url"].get<std::string>();
        config.frame_drop_interval = config_obj["frame_drop_interval"].get<int>();
        configs.push_back(config);
    }

    std::vector<std::shared_ptr<Streamer>> streamers;

    // 初始化并启动流
    for (const auto& config : configs) {

        std::shared_ptr<Streamer> streamer = std::make_shared<Streamer>();
        if (!streamer->init(config)) {
            std::cerr << "Failed to initialize streamer" << std::endl;
            return -1;
        }

        streamer->start([](const AVPacket *pkt, const AVFrame *frame) {
            //std::cout << "Received frame: " << frame->width << "x" << frame->height << std::endl;

        });

        streamers.push_back(streamer);
    }

    // 等待用户输入以停止
    uint64_t timer_id;
    tqp->create_timer(1000, 1, [streamers] (){
        int ch = 0;
        auto stats = streamers[0]->getStats();
        std::cout << "total fps ="
        << std::setiosflags(std::ios::fixed) << std::setprecision(1) << stats.fps
        <<  ",ch=" << ch << ":fps=" << stats.fps << std::endl;
    }, 1, &timer_id);

    tqp->run_loop();


    return 0;
}