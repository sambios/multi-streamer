//
// Created by yuan on 2025/8/1.
//

//
// Created by yuan on 2025/7/31.
//

#include "detector_yolo.h"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <string>
#include <numeric>
#include "otl.h"
#include "otl_ffmpeg.h"
#define TOPS_CHECK(func) {auto ret = func; assert(topsSuccess == ret);}

thread_local bool isDeviceSetted = false;

// COCO class labels mapping (from yolov5_ref.cpp)
std::map<int, std::string> label_map = {
    {0, "person"},         {1, "bicycle"},       {2, "car"},
    {3, "motorbike"},      {4,"aeroplane"},      {5, "bus"},
    {6, "train"},          {7, "truck"},         {8, "boat"},
    {9, "traffic_light"},  {10, "fire_hydrant"}, {11, "stop_sign"},
    {12, "parking_meter"}, {13, "bench"},        {14, "bird"},
    {15, "cat"},           {16, "dog"},          {17, "horse"},
    {18, "sheep"},         {19, "cow"},          {20, "elephant"},
    {21, "bear"},          {22, "zebra"},        {23, "giraffe"},
    {24, "backpack"},      {25, "umbrella"},     {26, "handbag"},
    {27, "tie"},           {28, "suitcase"},     {29, "frisbee"},
    {30, "skis"},          {31, "snowboard"},    {32, "sports_ball"},
    {33, "kite"},          {34, "baseball_bat"}, {35, "baseball_glove"},
    {36, "skateboard"},    {37, "surfboard"},    {38, "tennis_racket"},
    {39, "bottle"},        {40, "wine_glass"},   {41, "cup"},
    {42, "fork"},          {43, "knife"},        {44, "spoon"},
    {45, "bowl"},          {46, "banana"},       {47, "apple"},
    {48, "sandwich"},      {49, "orange"},       {50, "broccoli"},
    {51, "carrot"},        {52, "hot_dog"},      {53, "pizza"},
    {54, "donut"},         {55, "cake"},         {56, "chair"},
    {57, "sofa"},          {58, "pottedplant"},  {59, "bed"},
    {60, "diningtable"},   {61, "toilet"},       {62, "tvmonitor"},
    {63, "laptop"},        {64, "mouse"},        {65, "remote"},
    {66, "keyboard"},      {67, "cell_phone"},   {68, "microwave"},
    {69, "oven"},          {70, "toaster"},      {71, "sink"},
    {72, "refrigerator"},  {73, "book"},         {74, "clock"},
    {75, "vase"},          {76, "scissors"},     {77, "teddy_bear"},
    {78, "hair_drier"},    {79, "toothbrush"},
};

// Helper functions from yolov5_ref.cpp
void clamp(float &val, const float low, const float high) {
    if (val > high) {
        val = high;
    } else if (val < low) {
        val = low;
    }
}

void sort(int n, const std::vector<float> x, std::vector<int> indices) {
    int i, j;
    for (i = 0; i < n; i++)
        for (j = i + 1; j < n; j++) {
            if (x[indices[j]] > x[indices[i]]) {
                int index_tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = index_tmp;
            }
        }
}

bool nonMaximumSuppression(const std::vector<cv::Rect> rects,
                          const std::vector<float> score,
                          float overlap_threshold,
                          std::vector<int> &index_out) {
    int num_boxes = rects.size();
    int i, j;
    std::vector<float> box_area(num_boxes);
    std::vector<int> indices(num_boxes);
    std::vector<int> is_suppressed(num_boxes);

    for (i = 0; i < num_boxes; i++) {
        indices[i] = i;
        is_suppressed[i] = 0;
        box_area[i] = (float)((rects[i].width + 1) * (rects[i].height + 1));
    }

    sort(num_boxes, score, indices);

    for (i = 0; i < num_boxes; i++) {
        if (!is_suppressed[indices[i]]) {
            for (j = i + 1; j < num_boxes; j++) {
                if (!is_suppressed[indices[j]]) {
                    int x1max = std::max(rects[indices[i]].x, rects[indices[j]].x);
                    int x2min = std::min(rects[indices[i]].x + rects[indices[i]].width,
                                        rects[indices[j]].x + rects[indices[j]].width);
                    int y1max = std::max(rects[indices[i]].y, rects[indices[j]].y);
                    int y2min = std::min(rects[indices[i]].y + rects[indices[i]].height,
                                        rects[indices[j]].y + rects[indices[j]].height);
                    int overlap_w = x2min - x1max + 1;
                    int overlap_h = y2min - y1max + 1;
                    if (overlap_w > 0 && overlap_h > 0) {
                        float iou = (overlap_w * overlap_h) /
                                   (box_area[indices[j]] + box_area[indices[i]] -
                                    overlap_w * overlap_h);
                        if (iou > overlap_threshold) {
                            is_suppressed[indices[j]] = 1;
                        }
                    }
                }
            }
        }
    }

    for (i = 0; i < num_boxes; i++) {
        if (!is_suppressed[i])
            index_out.push_back(i);
    }

    return true;
}

YoloDetector::YoloDetector(int devId, std::string modelPath) : m_deviceId(OTL_GET_INT32_HIGH16(devId)), m_engine(nullptr),
                                       m_confThreshold(0.45f), m_nmsThreshold(0.25f){
   m_modelPath = modelPath;
}

YoloDetector::~YoloDetector() {
    std::cout << "YoloDetector: dtor" << std::endl;
    cleanup();
}

// Helper methods from yolov5_ref.cpp
int YoloDetector::get_dtype_size(TopsInference::DataType dtype) {
    switch (dtype) {
        case TopsInference::DataType::TIF_FP32:
            return 4;
        case TopsInference::DataType::TIF_FP16:
        case TopsInference::DataType::TIF_BF16:
            return 2;
        case TopsInference::DataType::TIF_INT32:
        case TopsInference::DataType::TIF_UINT32:
            return 4;
        case TopsInference::DataType::TIF_INT16:
        case TopsInference::DataType::TIF_UINT16:
            return 2;
        case TopsInference::DataType::TIF_INT8:
        case TopsInference::DataType::TIF_UINT8:
            return 1;
        case TopsInference::DataType::TIF_FP64:
        case TopsInference::DataType::TIF_INT64:
        case TopsInference::DataType::TIF_UINT64:
            return 8;
        default:
            return 4;
    }
}

std::vector<YoloDetector::ShapeInfo> YoloDetector::getInputsShape() {
    std::vector<ShapeInfo> shapes_info;
    int num = m_engine->getInputNum();
    for (int i = 0; i < num; i++) {
        auto name = m_engine->getInputName(i);
        auto Dims = m_engine->getInputShape(i);
        auto dtype = m_engine->getInputDataType(i);

        std::vector<int> shape;
        int dtype_size = get_dtype_size(dtype);
        int mem_size = dtype_size;
        for (int j = 0; j < Dims.nbDims; j++) {
            shape.push_back(Dims.dimension[j]);
            mem_size *= Dims.dimension[j];
        }
        shapes_info.push_back(ShapeInfo(name, shape, dtype, dtype_size, mem_size));
    }
    return shapes_info;
}

std::vector<YoloDetector::ShapeInfo> YoloDetector::getOutputsShape() {
    std::vector<ShapeInfo> shapes_info;
    int num = m_engine->getOutputNum();
    for (int i = 0; i < num; i++) {
        auto name = m_engine->getOutputName(i);
        auto Dims = m_engine->getOutputShape(i);
        auto dtype = m_engine->getOutputDataType(i);

        std::vector<int> shape;
        int dtype_size = get_dtype_size(dtype);
        int mem_size = dtype_size;
        for (int j = 0; j < Dims.nbDims; j++) {
            shape.push_back(Dims.dimension[j]);
            mem_size *= Dims.dimension[j];
        }
        shapes_info.push_back(ShapeInfo(name, shape, dtype, dtype_size, mem_size));
    }
    return shapes_info;
}

void YoloDetector::allocHostMemory(FrameInfo &info, bool isInput, std::vector<ShapeInfo> &shapes_info, int times, bool verbose) {
    std::vector<void*> datum;
    for (auto &shape_info : shapes_info) {
        int dataSize = shape_info.mem_size * times;
        char *data = new char[dataSize];
        if (isInput)
        {
            info.netHostInputs.push_back((void*)data);
            info.netInputsSize.push_back(dataSize);
        }
        else
        {
            info.netHostOutputs.push_back(data);
            info.netOutputsSize.push_back(dataSize);
        }

        if (verbose) {
            std::cout << "Allocated host memory size: " << shape_info.mem_size * times << std::endl;
        }

        //Device memory
        topsDevice_t *devMem = nullptr;
        auto ret = topsMalloc(&devMem, shape_info.mem_size * times);
        assert(ret == topsSuccess);
        if (isInput) info.netDeviceInputs.push_back(devMem);
        else info.netDeviceOutputs.push_back(devMem);
    }
}

void YoloDetector::freeHostMemory(FrameInfo &info, bool isInput) {
    if (isInput)
    {
        for (auto &data : info.netHostInputs) {
            delete[] (char*)data;
        }
        for (auto data : info.netDeviceInputs) {
            TOPS_CHECK(topsFree(data));
        }
        info.netHostInputs.clear();
    }else
    {
        for (auto &data : info.netHostOutputs) {
            delete[] (char*)data;
        }
        for (auto data : info.netDeviceOutputs) {
            TOPS_CHECK(topsFree(data));
        }
        info.netHostOutputs.clear();
    }
}

int YoloDetector::initialize() {

    // Initialize TopsInference
    TopsInference::topsInference_init();

    // Set device
    if (!isDeviceSetted)
    {
        std::vector<uint32_t> cluster_ids = {0};
        m_handler = TopsInference::set_device(m_deviceId, cluster_ids.data(), cluster_ids.size());
        if (!m_handler) {
            std::cerr << "threadid=" << std::this_thread::get_id() << ", Failed to set device " << m_deviceId << std::endl;
            return -1;
        }
        isDeviceSetted = true;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_engine != nullptr) return 0;
    try {
        // Initialize TopsInference
        //TopsInference::topsInference_init();

        // Set device
        //std::vector<uint32_t> cluster_ids = {0};
        //m_handler = TopsInference::set_device(m_deviceId, cluster_ids.data(), cluster_ids.size());
        //if (!m_handler) {
        //    std::cerr << "threadid=" << std::this_thread::get_id() << ", Failed to set device " << m_deviceId << std::endl;
        //    return -1;
        //}

        // Generate .exec file path from ONNX model path
        fs::path onnxPath(m_modelPath);
        std::string execPath = onnxPath.string() + ".exec";
        
        std::cout << "Looking for pre-built engine: " << execPath << std::endl;
        
        // Check if .exec file exists in current directory
        if (fs::exists(execPath)) {
            std::cout << "Found pre-built engine file: " << execPath << std::endl;
            std::cout << "Loading pre-built engine..." << std::endl;
            
            // Create engine and load from .exec file
            m_engine = TopsInference::create_engine();
            if (!m_engine) {
                std::cerr << "Failed to create engine for loading" << std::endl;
                return -1;
            }
            
            // Load pre-built engine
            auto ret = m_engine->loadExecutable(execPath.c_str());
            if (!ret) {
                std::cerr << "Failed to load pre-built engine from: " << execPath << std::endl;
                TopsInference::release_engine(m_engine);
                m_engine = nullptr;
                return -1;
            }
            
            std::cout << "Successfully loaded pre-built engine from: " << execPath << std::endl;
            
        } else {
            std::cout << "Pre-built engine not found, building from ONNX model: " << m_modelPath << std::endl;
            
            // Create parser and read model
            TopsInference::IParser* parser = TopsInference::create_parser(TopsInference::TIF_ONNX);
            if (!parser) {
                std::cerr << "Failed to create parser" << std::endl;
                return -1;
            }

            TopsInference::INetwork* network = parser->readModel(m_modelPath.c_str());
            if (!network) {
                std::cerr << "Failed to read model: " << m_modelPath << std::endl;
                TopsInference::release_parser(parser);
                exit(-1);
            }

            // Create optimizer and build engine
            TopsInference::IOptimizer* optimizer = TopsInference::create_optimizer();
            if (!optimizer) {
                std::cerr << "Failed to create optimizer" << std::endl;
                TopsInference::release_network(network);
                TopsInference::release_parser(parser);
                return -1;
            }

            // 指定模型推理为FP16和FP32混合精度
            optimizer->getConfig()->setBuildFlag(TopsInference::BuildFlag::TIF_KTYPE_MIX_FP16);

            std::cout << "Building engine from ONNX model..." << std::endl;
            m_engine = optimizer->build(network);
            if (!m_engine) {
                std::cerr << "Failed to build engine" << std::endl;
                TopsInference::release_optimizer(optimizer);
                TopsInference::release_network(network);
                TopsInference::release_parser(parser);
                return -1;
            }
            
            std::cout << "Engine built successfully, saving to: " << execPath << std::endl;
            
            // Save the built engine for future use
            auto status = m_engine->saveExecutable(execPath.c_str());
            if (!status) {
                std::cout << "Warning: Failed to save engine to " << execPath << ", but continuing..." << std::endl;
            } else {
                std::cout << "Engine saved successfully to: " << execPath << std::endl;
            }

            // Cleanup temporary objects
            TopsInference::release_optimizer(optimizer);
            TopsInference::release_network(network);
            TopsInference::release_parser(parser);
        }
        
        // Get input/output shapes (works for both loaded and built engines)
        TopsInference::Dims inputShape = m_engine->getInputShape(0);
        if (inputShape.nbDims >= 4) {
            m_inputHeight = inputShape.dimension[2];
            m_inputWidth = inputShape.dimension[3];
        }

        // Initialize shape information (following yolov5_ref.cpp pattern)
        m_inputShapes = getInputsShape();
        m_outputShapes = getOutputsShape();
        
        std::cout << "Input shapes:" << std::endl;
        for (const auto& shape : m_inputShapes) {
            std::cout << "  " << shape.name << ": [";
            for (size_t i = 0; i < shape.dims.size(); ++i) {
                std::cout << shape.dims[i];
                if (i < shape.dims.size() - 1) std::cout << ",";
            }
            std::cout << "] mem_size: " << shape.mem_size << std::endl;
        }
        
        std::cout << "Output shapes:" << std::endl;
        for (const auto& shape : m_outputShapes) {
            std::cout << "  " << shape.name << ": [";
            for (size_t i = 0; i < shape.dims.size(); ++i) {
                std::cout << shape.dims[i];
                if (i < shape.dims.size() - 1) std::cout << ",";
            }
            std::cout << "] mem_size: " << shape.mem_size << std::endl;
        }

        std::cout << "YOLO11n engine initialized successfully" << std::endl;
        std::cout << "Input size: " << m_inputWidth << "x" << m_inputHeight << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception in initializeEngine: " << e.what() << std::endl;
        return -1;
    }
}



int YoloDetector::preprocess(std::vector<FrameInfo>& frameInfos) {
    //std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_engine || m_inputShapes.empty()) {
        std::cerr << "Engine(dev=" << m_deviceId << ") not initialized" << std::endl;
        return -1;
    }

    for (auto& frameInfo : frameInfos)
    {
        int batchSize = 1;
        //frameInfo.netHostInputs = allocHostMemory(m_inputShapes, 1, false);
        //frameInfo.netDeviceInputs = allocDeviceMemory(m_inputShapes, 1, false);
        allocHostMemory(frameInfo, true, m_inputShapes, 1, false);
        //freeHostMemory(frameInfo.netHostOutputs);
        //frameInfo.netHostOutputs = allocHostMemory(m_outputShapes, 1, false);
        allocHostMemory(frameInfo, false, m_outputShapes, 1, false);

        // Shape is [n, c, h, w]
        int inputW = m_inputShapes[0].dims[3];
        int inputH = m_inputShapes[0].dims[2];
        for (int shapeIdx = 0; shapeIdx < m_inputShapes.size(); ++shapeIdx)
        {
            for (int i = 0; i < batchSize; ++i)
            {
                // Convert AVFrame to cv::Mat
                cv::Mat image;
                if (frameInfo.frame)
                {
                    // Simple AVFrame to cv::Mat conversion for YUV420P format
                    AVFrame* frame = frameInfo.frame;
                    int width = frame->width;
                    int height = frame->height;

                    if (frame->format == AV_PIX_FMT_YUV420P)
                    {
                        // Create cv::Mat from YUV420P data
                        cv::Mat yuv(height + height / 2, width, CV_8UC1, frame->data[0], frame->linesize[0]);
                        cv::cvtColor(yuv, image, cv::COLOR_YUV2BGR_I420);
                    }else if (frame->format == AV_PIX_FMT_BGR24)
                    {
                        cv::Mat bgr24(height, width, CV_8UC3, frame->data[0]);
                        image = bgr24;
                    }
                }
                else
                {
                    std::cerr << "Invalid frame at index " << i << std::endl;
                    return -1;
                }

                // Letterbox preprocessing (following yolov5_ref.cpp pattern)
                int maxLen = std::max(image.cols, image.rows);
                cv::Mat image2 = cv::Mat::zeros(cv::Size(maxLen, maxLen), CV_8UC3);
                image.copyTo(image2(cv::Rect(0, 0, image.cols, image.rows)));

                cv::Mat input;
                cv::resize(image2, input, cv::Size(inputW, inputH));

                // Convert BGR to RGB and normalize to [0, 1]
                cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
                input.convertTo(input, CV_32F, 1.0f / 255.0f);

                // Convert HWC to CHW format and copy to input buffer
                int planeSize = input.cols * input.rows;
                // 使用当前缓冲区而不是nextBuffer
                auto* begin = static_cast<float*>((void*)((char*)frameInfo.netHostInputs[shapeIdx] + m_inputShapes[shapeIdx].mem_size * i));
                cv::Mat b(cv::Size(input.cols, input.rows), CV_32FC1, begin);
                cv::Mat g(cv::Size(input.cols, input.rows), CV_32FC1, begin + planeSize);
                cv::Mat r(cv::Size(input.cols, input.rows), CV_32FC1, begin + (planeSize << 1));
                cv::Mat rgb[3] = {r, g, b};
                cv::split(input, rgb);
            }
        }

        syncInputs(frameInfo);
    }

    return 0;
}

int YoloDetector::forward(std::vector<FrameInfo>& frameInfos) {
    //std::lock_guard<std::mutex> lock(m_mutex);

    if (!m_engine && m_inputShapes.empty()) {
        std::cerr << "Engine(dev=" << m_deviceId << ") not initialized" << std::endl;
        return -1;
    }

   for (auto& frameInfo : frameInfos)
   {
       // Run inference using runWithBatch (following yolov5_ref.cpp pattern)
       auto success = m_engine->runWithBatch(
           1,
           frameInfo.netDeviceInputs.data(),
           frameInfo.netDeviceOutputs.data(),
           TopsInference::BufferType::TIF_ENGINE_RSC_IN_DEVICE_OUT_DEVICE);

       if (!success) {
           std::cerr << "engine runWithBatch failed." << std::endl;
           return -1;
       }
   }

    return 0;
}



int YoloDetector::postprocess(std::vector<FrameInfo>& frameInfos) {

    for (auto& frameInfo : frameInfos)
    {
        syncOutputs(frameInfo);
        if (frameInfo.netHostOutputs.empty()) {
            std::cerr << "No output data available" << std::endl;
            return -1;
        }

        // Shape is [n, c, h, w]
        int inputW = m_inputShapes[0].dims[3];
        int inputH = m_inputShapes[0].dims[2];

        for (int bactchIdx = 0; bactchIdx < 1; ++bactchIdx)
        {
            // OutputData is column-major, [[batch1_output1, batch2_output1,...],[batch1_output2, batch2_output2,...]...]
            int outputShapeIdx = 0; // yolov8m has only one output shape.
            //[1,84,8400]
            float* output1 = (float*)frameInfo.netHostOutputs[outputShapeIdx] + m_outputShapes[outputShapeIdx].volume * bactchIdx;

            int output_h = m_outputShapes[outputShapeIdx].dims[1];
            int output_w = m_outputShapes[outputShapeIdx].dims[2];
            int maxLen = MAX(frameInfo.frame->width, frameInfo.frame->height);
            float x_factor, y_factor;
            x_factor = y_factor = maxLen / static_cast<float>(inputW);

            cv::Mat dout(output_h, output_w, CV_32F, (float*)output1);
            cv::Mat det_output = dout.t(); // 8400x84

            std::vector<cv::Rect> boxes;
            std::vector<int> classIds;
            std::vector<float> confidences;

            for (int i = 0; i < det_output.rows; i++)
            {
                cv::Mat classes_scores = det_output.row(i).colRange(4, 84);
                cv::Point classIdPoint;
                double score;
                cv::minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

                // 置信度 0～1之间
                if (score > m_confThreshold)
                {
                    float cx = det_output.at<float>(i, 0);
                    float cy = det_output.at<float>(i, 1);
                    float ow = det_output.at<float>(i, 2);
                    float oh = det_output.at<float>(i, 3);
                    int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
                    int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
                    int width = static_cast<int>(ow * x_factor);
                    int height = static_cast<int>(oh * y_factor);
                    cv::Rect box;
                    box.x = x;
                    box.y = y;
                    box.width = width;
                    box.height = height;

                    boxes.push_back(box);
                    classIds.push_back(classIdPoint.x);
                    confidences.push_back(score);
                }
            }

            // NMS
            std::vector<int> indexes;
            cv::dnn::NMSBoxes(boxes, confidences, m_nmsThreshold, m_confThreshold, indexes);

            //printf("detect num: %d\n", (int)indexes.size());
            for (size_t i = 0; i < indexes.size(); i++)
            {
                int index = indexes[i];
                int cls_id = classIds[index];
                otl::Bbox bbox;
                bbox.x1 = boxes[index].x;
                bbox.y1 = boxes[index].y;
                bbox.x2 = bbox.x1 + boxes[index].width;
                bbox.y2 = bbox.y1 + boxes[index].height;
                bbox.classId = cls_id;

                //detection.label = label_map[cls_id];
                bbox.confidence = confidences[index];
                //printf("box[%d, %d, %d, %d], conf:%f, cls:%d\n", boxes[index].x, boxes[index].y, boxes[index].width, boxes[index].height, confidences[index], cls_id);
                frameInfo.detection.push_back(bbox);
            }

            // for debug
            //std::cout << "size of result: " << dets.size() << std::endl;

        }
    }

    // Continue with original postprocessing logic
    for (auto frameInfo : frameInfos) {
        if (m_pfnDetectFinish) {
            m_pfnDetectFinish(frameInfo);
        }

        if (m_nextInferPipe != nullptr) {
            m_nextInferPipe->push_frame(&frameInfo);
        } else {
            //Free input/output
            freeHostMemory(frameInfo, true);
            freeHostMemory(frameInfo, false);

            // Stop pipeline
            av_packet_unref(frameInfo.pkt);
            av_packet_free(&frameInfo.pkt);

            av_frame_unref(frameInfo.frame);
            av_frame_free(&frameInfo.frame);
        }
    }

    return 0;
}

void YoloDetector::cleanup() {
    if (m_engine) {
        TopsInference::release_engine(m_engine);
        m_engine = nullptr;
    }

    if (m_handler) {
        TopsInference::release_device(m_handler);
        m_handler = nullptr;
    }

    TopsInference::topsInference_finish();
}


std::shared_ptr<Detector> Detector::createDetector(int devId, std::string modelPath) {
    std::shared_ptr<Detector> detector;
    detector.reset(new YoloDetector(devId, modelPath));
    return detector;
}