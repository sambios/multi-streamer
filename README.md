# Multi-Streamer
## 一、docker基础环境准备
### （一）/data/demo目录结构
```
demo
├─ multi-streamer
├─ otl
└─ test_car_person_1080P.h264
```

### （二）运行docker
需将主机的/data/目录挂载到docker，执行以下命令：
```bash
docker run -itd --name yolo-demo \
--volume=/data:/data \
--ipc=host --ulimit core=-1 --security-opt seccomp=unconfined \
--network host --privileged \
all_in_one_x86_image:v1 bash
```

### （三）进入docker准备demo运行环境
执行命令进入docker容器：
```bash
docker exec -it yolo-demo bash
```

### （四）配置python运行版本
将python运行版本配置为python3.10，执行以下两条命令：
```bash
ln -sf /usr/local/bin/python3.10 /usr/bin/python
ln -sf /usr/local/bin/pip3.10 /usr/local/bin/pip3
```

### （五）更新pip源和apt源
1. 更新pip源：
```bash
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
2. 更新apt源：
```bash
cat /etc/apt/sources.list.bak > /etc/apt/sources.list
apt update
```

### （六）安装依赖包
1. 安装python依赖包：
```bash
pip3 install flask
```
2. 安装系统依赖包：
```bash
apt -y install libavdevice-dev libavdevice58 nlohmann-json3-dev
```

## 三、VPU 多路解码demo（multi-streamer）
### （一）准备源视频文件
1. 进入docker容器（若已在容器内可跳过此步）：
```bash
docker exec -it yolo-demo bash
```
2. 转换视频格式生成源视频文件：
```bash
/opt/tops/bin/ffmpeg -i enrigin_yolo/test_car_person_1080P.mp4 -c:v copy -an test_car_person_1080P.h264
```

### （二）编译multi-streamer
1. 进入docker容器（若已在容器内可跳过此步）：
```bash
docker exec -it yolo-demo bash
```
2. 复制文件并配置环境变量：
```bash
cd /data/demo
cp codec_sample/hwcontext.c codec_sample/pixfmt.h /usr/include/x86_64-linux-gnu/libavutil/
export VCE_SHRED_LIB=/opt/tops/lib/libh2enc.so
```
3. 创建编译目录并执行编译：
```bash
cd multi-streamer
mkdir build 
cd build
cmake ..
make -j10
```

### （三）运行demo
1. 检查或修改配置文件 `multi-streamer/config.json`，配置内容如下：
```json
{
"dev_num": 1,  #使用一张卡
"configs_dev0": [
{
"input_url": "/data/demo/test_car_person_1080P.h264",  # 适配流文件
"frame_drop_interval": 2,
"output_url_base": "udp://127.0.0.1:9000",  #暂时未生效
"client_count":2  # 2路并发,最大128
}
]
}
```
2. 运行程序：
```bash
cd multi-streamer/build
./video_detection
```

### （四）测试输出结果
```
open url=udp://127.0.0.1:9102,format_name=h264
Output #0, h264, to 'udp://127.0.0.1:9102':
Stream #0:0: Video: h264 (Main), 1 reference frame, yuv420p(progressive, left), 1920x1080 (0x0) [SAR 1:1 DAR 16:9], q=2-31
create decoderID = 0, hwdevicectx /dev/gcu0vid0
Decoder: h264_vsv_decoder, device type: vsv.
hw_pix_fmt 198
[h264_mp4toannexb @ 0x7fa96c74b660] The input looks like it is Annex B already
create video decoder ok!
id=0, ffmpeg delayed frames: 0
ch-num=128 FPS=-nan
ch-num=128 FPS=3086.27
ch-num=128 FPS=3079.11
ch-num=128 FPS=3076.79
ch-num=128 FPS=3075.61
ch-num=128 FPS=3071.94
ch-num=128 FPS=3072.05
ch-num=128 FPS=3072.01
ch-num=128 FPS=3071.99
ch-num=128 FPS=3072.03
ch-num=128 FPS=3072.05
ch-num=128FPS=3072.03
ch-num=128 FPS=3072.12
```

### （五）D10VPU使用率和状态监控
1. 监控命令：
```bash
ersmi-dmon-sv
```
2. 监控输出结果：
```
(base)[root4090-10-16-201-1~]#ersmi-dmon-sv
----Enrigin System Management Interface
------------- Enrigin Tech, All Rights Reserved.2024 Copyright(C)
*Dev VPU0_DECO VPU0_DEC1 VPU0_DEC2 VPU0_DEC3 VPU0_ENC VPU1_DECO VPU1_DEC1 VPU1_DEC2 VPU1_DEC3 VPU1_ENC
*Idx 号 号 号 号 0 号 号 号 号 0
0 65 65 65 65 0 65 65 65 65 0
0 65 65 65 65 0 65 65 65 65 0
0 65 65 65 65 0 65 65 65 65 0
0 64 64 64 64 0 64 64 64 64 0
0 63 63 63 63 0 63 63 63 63 0
0 65 65 65 65 0 65 65 65 65 0
0 65 65 65 65 0 65 65 65 65 0
```

### （五）D10使用率和状态监控
1. 监控命令：
```bash
ersmi -dmon
```
2. 监控输出结果：
```
(base) [root@4090-10-16-201-1 ~]# ersmi -dmon
------------------------- Enrigin System Management Interface ------------------------
---------------- Enrigin Tech, All Rights Reserved. 2024 Copyright (C) ---------------

| *Dev Idx | Pwr (W) | DTemp (°C) | Sip (%) | DUsed (%) | Dpm | MUsed | Mem (MiB) | Dclk (MHz) | Mclk (MHz) | VOLT (V) | TxPCI (MiB/s) | RxPCI (MiB/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 51 | 48. | 100.0 | 100.0 | Active | 2.0 | 57340 | 1000 | 6400 | 0.852 | 0 | 59% |
| 0 | 51 | 48. | 100.0 | 100.0 | Active | 2.0 | 57340 | 1000 | 6400 | 0.852 | 0 | 62. |
| 0 | 51 | 48. | 100.0 | 100.0 | Active | 2.0 | 57340 | 1000 | 6400 | 0.860 | 2692 | 10561 |
| 0 | 51 | 48. | 100.0 | 100.0 | Active | 2.0 | 57340 | 1000 | 6400 | 0.850 | 0 | 1798 |
```这条消息已经在编辑器中准备就绪。你想如何调整这篇文档?请随时告诉我。


