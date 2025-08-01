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

YoloDetector::YoloDetector(int devId) : m_deviceId(devId), m_engine(nullptr),
                                       m_confThreshold(0.45f), m_nmsThreshold(0.25f),
                                       m_currentBuffer(0) {
    // Initialize TopsInference
    TopsInference::topsInference_init();
    
    // Set device
    std::vector<uint32_t> cluster_ids = {0};
    m_handler = TopsInference::set_device(m_deviceId, cluster_ids.data(), cluster_ids.size());
    
    if (!m_handler) {
        std::cerr << "Failed to set device " << m_deviceId << std::endl;
        return;
    }
    
    // Initialize engine with default model path
    std::string modelPath = "yolov5s.onnx"; // Default model path
    if (!initializeEngine(modelPath)) {
        std::cerr << "Failed to initialize engine" << std::endl;
    }
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

std::vector<void*> YoloDetector::allocHostMemory(std::vector<ShapeInfo> &shapes_info, int times, bool verbose) {
    std::vector<void*> datum;
    for (auto &shape_info : shapes_info) {
        char *data = new char[shape_info.mem_size * times];
        datum.push_back((void*)data);
        if (verbose) {
            std::cout << "Allocated memory size: " << shape_info.mem_size * times << std::endl;
        }
    }
    return datum;
}

void YoloDetector::freeHostMemory(std::vector<void*> &datum) {
    for (auto &data : datum) {
        delete[] (char*)data;
    }
    datum.clear();
}

bool YoloDetector::initializeEngine(const std::string& modelPath) {
    try {
        // Generate .exec file path from ONNX model path
        fs::path onnxPath(modelPath);
        std::string execPath = onnxPath.filename().string() + ".exec";
        
        std::cout << "Looking for pre-built engine: " << execPath << std::endl;
        
        // Check if .exec file exists in current directory
        if (fs::exists(execPath)) {
            std::cout << "Found pre-built engine file: " << execPath << std::endl;
            std::cout << "Loading pre-built engine..." << std::endl;
            
            // Create engine and load from .exec file
            m_engine = TopsInference::create_engine();
            if (!m_engine) {
                std::cerr << "Failed to create engine for loading" << std::endl;
                return false;
            }
            
            // Load pre-built engine
            auto ret = m_engine->loadExecutable(execPath.c_str());
            if (!ret) {
                std::cerr << "Failed to load pre-built engine from: " << execPath << std::endl;
                TopsInference::release_engine(m_engine);
                m_engine = nullptr;
                return false;
            }
            
            std::cout << "Successfully loaded pre-built engine from: " << execPath << std::endl;
            
        } else {
            std::cout << "Pre-built engine not found, building from ONNX model: " << modelPath << std::endl;
            
            // Create parser and read model
            TopsInference::IParser* parser = TopsInference::create_parser(TopsInference::TIF_ONNX);
            if (!parser) {
                std::cerr << "Failed to create parser" << std::endl;
                return false;
            }

            TopsInference::INetwork* network = parser->readModel(modelPath.c_str());
            if (!network) {
                std::cerr << "Failed to read model: " << modelPath << std::endl;
                TopsInference::release_parser(parser);
                return false;
            }

            // Create optimizer and build engine
            TopsInference::IOptimizer* optimizer = TopsInference::create_optimizer();
            if (!optimizer) {
                std::cerr << "Failed to create optimizer" << std::endl;
                TopsInference::release_network(network);
                TopsInference::release_parser(parser);
                return false;
            }

            std::cout << "Building engine from ONNX model..." << std::endl;
            m_engine = optimizer->build(network);
            if (!m_engine) {
                std::cerr << "Failed to build engine" << std::endl;
                TopsInference::release_optimizer(optimizer);
                TopsInference::release_network(network);
                TopsInference::release_parser(parser);
                return false;
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

        std::cout << "YOLOv5 engine initialized successfully" << std::endl;
        std::cout << "Input size: " << m_inputWidth << "x" << m_inputHeight << std::endl;

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in initializeEngine: " << e.what() << std::endl;
        return false;
    }
}



int YoloDetector::preprocess(std::vector<FrameInfo>& frames) {
    if (!m_engine) {
        std::cerr << "Engine not initialized" << std::endl;
        return -1;
    }

    int batchSize = frames.size();
     
    // 加锁保护共享资源
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // 分配当前缓冲区的输入内存，而不是下一个缓冲区
    // 这确保了我们在forward中使用的输入数据是已准备好的
    
    // Free previous input memory for current buffer
    freeHostMemory(m_netInputs[m_currentBuffer]);
    
    // Allocate host memory for batch inputs (following yolov5_ref.cpp pattern)
    m_netInputs[m_currentBuffer] = allocHostMemory(m_inputShapes, batchSize, false);
    
    // Shape is [n, c, h, w]
    int inputW = m_inputShapes[0].dims[3];
    int inputH = m_inputShapes[0].dims[2];

    for (int shapeIdx = 0; shapeIdx < m_inputShapes.size(); ++shapeIdx) {
        for (int i = 0; i < batchSize; ++i) {
            // Convert AVFrame to cv::Mat
            cv::Mat image;
            if (frames[i].frame) {
                // Simple AVFrame to cv::Mat conversion for YUV420P format
                AVFrame* frame = frames[i].frame;
                int width = frame->width;
                int height = frame->height;
                
                // Create cv::Mat from YUV420P data
                cv::Mat yuv(height + height/2, width, CV_8UC1, frame->data[0], frame->linesize[0]);
                cv::cvtColor(yuv, image, cv::COLOR_YUV2BGR_I420);
            } else {
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
            float *begin = static_cast<float *>((void*)((char*)m_netInputs[m_currentBuffer][shapeIdx] + m_inputShapes[shapeIdx].mem_size*i));
            cv::Mat b(cv::Size(input.cols, input.rows), CV_32FC1, begin);
            cv::Mat g(cv::Size(input.cols, input.rows), CV_32FC1, begin + planeSize);
            cv::Mat r(cv::Size(input.cols, input.rows), CV_32FC1, begin + (planeSize << 1));
            cv::Mat rgb[3] = {r, g, b};
            cv::split(input, rgb);
        }
    }

    return 0;
}

int YoloDetector::forward(std::vector<FrameInfo>& frames) {
    if (!m_engine) {
        std::cerr << "Engine not ready" << std::endl;
        return -1;
    }

    int batch_size = frames.size();
     
    // 加锁保护共享资源
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // 确保当前索引合法
    int nextBuffer = 1 - m_currentBuffer;
    
    // 先分配新的输出内存
    freeHostMemory(m_netOutputs[nextBuffer]);
    m_netOutputs[nextBuffer] = allocHostMemory(m_outputShapes, batch_size, false);
    
    // 只有在新内存分配成功后，才切换缓冲区
    m_currentBuffer = nextBuffer;

    // Run inference using runWithBatch (following yolov5_ref.cpp pattern)
    auto success = m_engine->runWithBatch(
        batch_size, 
        m_netInputs[m_currentBuffer].data(),
        m_netOutputs[m_currentBuffer].data(),
        TopsInference::BufferType::TIF_ENGINE_RSC_IN_HOST_OUT_HOST);
        
    if (!success) {
        std::cerr << "engine runWithBatch failed." << std::endl;
        return -1;
    }

    return 0;
}



int YoloDetector::postprocess(std::vector<FrameInfo>& frames) {
    // 加锁保护共享资源
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_netOutputs[m_currentBuffer].empty()) {
        std::cerr << "No output data available" << std::endl;
        return -1;
    }
    
    // Shape is [n, c, h, w]
    int inputW = m_inputShapes[0].dims[3];
    int inputH = m_inputShapes[0].dims[2];

    // 保存当前处理的批次大小
    int currentBatchSize = frames.size();
    
    // 检查是否有足够的输出形状信息
    if (m_outputShapes.empty()) {
        std::cerr << "Output shapes not initialized" << std::endl;
        return -1;
    }
    
    // 检查输出索引是否有效
    int outputShapeIdx = 0; // yolov5-v6.2 has only one output shape
    if (outputShapeIdx >= m_outputShapes.size() || outputShapeIdx >= m_netOutputs[m_currentBuffer].size()) {
        std::cerr << "Invalid output shape index" << std::endl;
        return -1;
    }
    
    for (int batchIdx = 0; batchIdx < currentBatchSize; ++batchIdx) {
        // 边界检查，确保不越界
        if (batchIdx >= currentBatchSize) {
            std::cerr << "Batch index " << batchIdx << " exceeds batch size " << currentBatchSize << std::endl;
            continue;
        }
        
        // Clear previous detections
        frames[batchIdx].bboxes.clear();
        
        // 计算单个样本输出所需的内存大小
        size_t singleOutputSize = m_outputShapes[outputShapeIdx].volume;
        
        // 检查内存越界
        size_t maxElements = m_outputShapes[outputShapeIdx].mem_size / sizeof(float);
        if (batchIdx * singleOutputSize + singleOutputSize > maxElements) {
            std::cerr << "Memory access out of bounds: batchIdx=" << batchIdx 
                     << ", volume=" << singleOutputSize 
                     << ", max=" << maxElements << std::endl;
            continue;
        }
        
        // 计算安全的内存偏移量，确保不会越界
        size_t safeOffset = std::min(batchIdx * singleOutputSize, maxElements - singleOutputSize);
        float* output1 = (float*)m_netOutputs[m_currentBuffer][outputShapeIdx] + safeOffset;
       
        std::vector<cv::Rect> selected_boxes;
        std::vector<float> confidence;
        std::vector<int> class_id;

        // YOLOv5 output format: [1, 25200, 85] where 85 = 4(bbox) + 1(conf) + 80(classes)
        int num_detections = m_outputShapes[outputShapeIdx].dims[1]; // 25200
        int detection_size = m_outputShapes[outputShapeIdx].dims[2]; // 85
        
        for (int i = 0; i < num_detections; ++i) {
            float* detection = output1 + i * detection_size;
            
            // Extract bbox coordinates and confidence
            float center_x = detection[0];
            float center_y = detection[1];
            float width = detection[2];
            float height = detection[3];
            float objectness = detection[4];
            
            // Skip low confidence detections
            if (objectness < m_confThreshold) continue;
            
            // Find best class
            float max_class_score = 0.0f;
            int best_class_id = 0;
            for (int c = 0; c < m_numClasses; ++c) {
                float class_score = detection[5 + c];
                if (class_score > max_class_score) {
                    max_class_score = class_score;
                    best_class_id = c;
                }
            }
            
            float final_score = objectness * max_class_score;
            if (final_score < m_confThreshold) continue;
            
            // Convert center format to corner format
            int x = static_cast<int>(center_x - width / 2);
            int y = static_cast<int>(center_y - height / 2);
            int w = static_cast<int>(width);
            int h = static_cast<int>(height);
            
            selected_boxes.push_back(cv::Rect(x, y, w, h));
            confidence.push_back(final_score);
            class_id.push_back(best_class_id);
        }
        
        // No object detected
        if (selected_boxes.size() == 0) {
            //std::cout << "no bbox over score threshold detected." << std::endl;
            continue;
        }
        
        // Apply NMS using reference implementation
        std::vector<int> indexes;
        nonMaximumSuppression(selected_boxes, confidence, m_nmsThreshold, indexes);
        
        // Convert detections to Bbox format and scale coordinates
        for (int id : indexes) {
            auto result_box = selected_boxes[id];
            
            // Scale coordinates back to original image dimensions
            int originalWidth = frames[batchIdx].frame->width;
            int originalHeight = frames[batchIdx].frame->height;
            
            // Calculate scale factors (letterbox preprocessing)
            int maxLen = std::max(originalWidth, originalHeight);
            float scale = static_cast<float>(maxLen) / inputW;
            
            result_box.x = static_cast<int>(result_box.x * scale);
            result_box.y = static_cast<int>(result_box.y * scale);
            result_box.width = static_cast<int>(result_box.width * scale);
            result_box.height = static_cast<int>(result_box.height * scale);
            
            // Clamp to image boundaries
            result_box.x = std::max(0, std::min(result_box.x, originalWidth));
            result_box.y = std::max(0, std::min(result_box.y, originalHeight));
            result_box.width = std::max(0, std::min(result_box.width, originalWidth - result_box.x));
            result_box.height = std::max(0, std::min(result_box.height, originalHeight - result_box.y));
            
            Bbox det_bbox;
            det_bbox.rect = result_box;
            det_bbox.confidence = confidence[id];
            det_bbox.classId = class_id[id];
            frames[batchIdx].bboxes.push_back(det_bbox);
            
            //std::string label = (label_map.find(class_id[id]) != label_map.end()) ? label_map[class_id[id]] : "unknown";
            //std::cout << "cls: " << label << " conf: " << confidence[id]
            //          << " (" << result_box.x << "," << result_box.y << "," 
            //          << result_box.width << "," << result_box.height << ")" << std::endl;
        }
        
        //std::cout << "Frame " << batchIdx << ": Detected " << frames[batchIdx].bboxes.size() << " objects" << std::endl;
    }

    // Continue with original postprocessing logic
    for (auto frameInfo : frames) {
        if (m_pfnDetectFinish) {
            m_pfnDetectFinish(frameInfo);
        }

        if (m_nextInferPipe != nullptr) {
            m_nextInferPipe->push_frame(&frameInfo);
        } else {
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
    // Free host memory (following yolov5_ref.cpp pattern)
    freeHostMemory(m_netInputs[0]);
    freeHostMemory(m_netInputs[1]);
    freeHostMemory(m_netOutputs[0]);
    freeHostMemory(m_netOutputs[1]);
    
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


std::shared_ptr<Detector> Detector::createDetector(int devId) {
    std::shared_ptr<Detector> detector;
    detector.reset(new YoloDetector(devId));
    return detector;
}