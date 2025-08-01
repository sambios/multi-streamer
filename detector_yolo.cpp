//
// Created by yuan on 2025/8/1.
//

//
// Created by yuan on 2025/7/31.
//

#include "detector_yolo.h"
#include "otl_ffmpeg.h"
#include <iostream>
#include <algorithm>
#include <cmath>

YoloDetector::YoloDetector(int devId)
    : m_engine(nullptr)
    , m_handler(nullptr)
    , m_deviceId(devId)
    , m_inputWidth(640)
    , m_inputHeight(640)
    , m_numClasses(80)
    , m_confThreshold(0.5f)
    , m_nmsThreshold(0.4f) {

    std::cout << "YoloDetector: ctor, device ID: " << m_deviceId << std::endl;

    // Initialize TopsInference
    TopsInference::topsInference_init();

    // Create GCU device handler
    uint32_t cluster_id[] = {0};
    m_handler = TopsInference::set_device(m_deviceId, cluster_id);

    // Initialize engine with default model path (can be configured)
    // Note: User should provide actual model path
    std::string modelPath = "./models/yolov5s.onnx"; // Default path
    if (!initializeEngine(modelPath)) {
        std::cerr << "Failed to initialize YOLOv5 engine" << std::endl;
    }
}

YoloDetector::~YoloDetector() {
    std::cout << "YoloDetector: dtor" << std::endl;
    cleanup();
}

bool YoloDetector::initializeEngine(const std::string& modelPath) {
    try {
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

        m_engine = optimizer->build(network);
        if (!m_engine) {
            std::cerr << "Failed to build engine" << std::endl;
            TopsInference::release_optimizer(optimizer);
            TopsInference::release_network(network);
            TopsInference::release_parser(parser);
            return false;
        }

        // Get input/output shapes
        TopsInference::Dims inputShape = m_engine->getInputShape(0);
        if (inputShape.nbDims >= 4) {
            m_inputHeight = inputShape.dimension[2];
            m_inputWidth = inputShape.dimension[3];
        }

        std::cout << "YOLOv5 engine initialized successfully" << std::endl;
        std::cout << "Input size: " << m_inputWidth << "x" << m_inputHeight << std::endl;

        // Cleanup temporary objects
        TopsInference::release_optimizer(optimizer);
        TopsInference::release_network(network);
        TopsInference::release_parser(parser);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in initializeEngine: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat YoloDetector::preprocessImage(const cv::Mat& image) {
    cv::Mat resized, normalized;

    // Resize to model input size
    cv::resize(image, resized, cv::Size(m_inputWidth, m_inputHeight));

    // Convert BGR to RGB and normalize to [0, 1]
    cv::cvtColor(resized, normalized, cv::COLOR_BGR2RGB);
    normalized.convertTo(normalized, CV_32F, 1.0/255.0);

    return normalized;
}

int YoloDetector::preprocess(std::vector<FrameInfo>& frames) {
    if (!m_engine) {
        std::cerr << "Engine not initialized" << std::endl;
        return -1;
    }

    m_preprocessedImages.clear();
    m_inputData.clear();

    // Calculate total input size
    size_t batchSize = frames.size();
    size_t inputSize = batchSize * 3 * m_inputHeight * m_inputWidth;
    m_inputData.resize(inputSize);

    for (size_t i = 0; i < frames.size(); ++i) {
        // Convert AVFrame to cv::Mat
        cv::Mat image;
        if (frames[i].frame) {
            // Convert AVFrame to cv::Mat (assuming YUV420P format)
            int width = frames[i].frame->width;
            int height = frames[i].frame->height;

            cv::Mat yuv(height + height/2, width, CV_8UC1, frames[i].frame->data[0], frames[i].frame->linesize[0]);
            cv::cvtColor(yuv, image, cv::COLOR_YUV2BGR_I420);
        } else {
            std::cerr << "Invalid frame data" << std::endl;
            continue;
        }

        // Preprocess image
        cv::Mat preprocessed = preprocessImage(image);
        m_preprocessedImages.push_back(preprocessed);

        // Copy to input buffer (CHW format)
        float* inputPtr = m_inputData.data() + i * 3 * m_inputHeight * m_inputWidth;

        // Convert HWC to CHW format
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < m_inputHeight; ++h) {
                for (int w = 0; w < m_inputWidth; ++w) {
                    inputPtr[c * m_inputHeight * m_inputWidth + h * m_inputWidth + w] =
                        preprocessed.at<cv::Vec3f>(h, w)[c];
                }
            }
        }
    }

    return 0;
}

int YoloDetector::forward(std::vector<FrameInfo>& frames) {
    if (!m_engine || m_inputData.empty()) {
        std::cerr << "Engine not ready or no input data" << std::endl;
        return -1;
    }

    try {
        size_t batchSize = frames.size();

        // Create input tensor
        std::vector<TopsInference::TensorPtr_t> inputTensors;
        TopsInference::TensorPtr_t inputTensor = TopsInference::create_tensor();

        inputTensor->setOpaque(reinterpret_cast<void*>(m_inputData.data()));
        inputTensor->setDeviceType(TopsInference::DataDeviceType::HOST);

        TopsInference::Dims inputShape = m_engine->getInputShape(0);
        inputShape.dimension[0] = batchSize;
        inputTensor->setDims(inputShape);

        inputTensors.push_back(inputTensor);

        // Create output tensor
        std::vector<TopsInference::TensorPtr_t> outputTensors;
        TopsInference::Dims outputShape = m_engine->getMaxOutputShape(0);
        outputShape.dimension[0] = batchSize;

        int64_t outputSize = 1;
        for (size_t i = 0; i < outputShape.nbDims; ++i) {
            outputSize *= outputShape.dimension[i];
        }

        m_outputData.resize(outputSize);

        TopsInference::TensorPtr_t outputTensor = TopsInference::create_tensor();
        outputTensor->setOpaque(reinterpret_cast<void*>(m_outputData.data()));
        outputTensor->setDeviceType(TopsInference::DataDeviceType::HOST);
        outputTensor->setDims(outputShape);

        outputTensors.push_back(outputTensor);

        // Run inference
        TopsInference::TIFStatus status = m_engine->runV2(
            inputTensors.data(),
            outputTensors.data()
        );

        if (status != TopsInference::TIF_OK) {
            std::cerr << "Inference failed with status: " << status << std::endl;

            // Cleanup tensors
            TopsInference::destroy_tensor(inputTensor);
            TopsInference::destroy_tensor(outputTensor);
            return -1;
        }

        // Cleanup tensors
        TopsInference::destroy_tensor(inputTensor);
        TopsInference::destroy_tensor(outputTensor);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception in forward: " << e.what() << std::endl;
        return -1;
    }
}

std::vector<cv::Rect> YoloDetector::postprocessDetections(const float* output, int imageWidth, int imageHeight) {
    std::vector<cv::Rect> detections;

    // YOLOv5 output format: [batch, num_detections, 85] where 85 = 4(bbox) + 1(conf) + 80(classes)
    const int numDetections = 25200; // Typical for 640x640 input
    const int outputStride = 4 + 1 + m_numClasses; // bbox + confidence + classes

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    for (int i = 0; i < numDetections; ++i) {
        const float* detection = output + i * outputStride;

        float confidence = detection[4];
        if (confidence < m_confThreshold) continue;

        // Find best class
        int bestClassId = 0;
        float bestClassScore = detection[5];
        for (int c = 1; c < m_numClasses; ++c) {
            if (detection[5 + c] > bestClassScore) {
                bestClassScore = detection[5 + c];
                bestClassId = c;
            }
        }

        float finalScore = confidence * bestClassScore;
        if (finalScore < m_confThreshold) continue;

        // Convert from center format to corner format
        float centerX = detection[0];
        float centerY = detection[1];
        float width = detection[2];
        float height = detection[3];

        // Scale to original image size
        float scaleX = static_cast<float>(imageWidth) / m_inputWidth;
        float scaleY = static_cast<float>(imageHeight) / m_inputHeight;

        int x = static_cast<int>((centerX - width/2) * scaleX);
        int y = static_cast<int>((centerY - height/2) * scaleY);
        int w = static_cast<int>(width * scaleX);
        int h = static_cast<int>(height * scaleY);

        boxes.push_back(cv::Rect(x, y, w, h));
        confidences.push_back(finalScore);
        classIds.push_back(bestClassId);
    }

    // Apply NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, m_confThreshold, m_nmsThreshold, indices);

    for (int idx : indices) {
        detections.push_back(boxes[idx]);
    }

    return detections;
}

int YoloDetector::postprocess(std::vector<FrameInfo>& frames) {
    if (m_outputData.empty()) {
        std::cerr << "No output data available" << std::endl;
        return -1;
    }

    // Process detections for each frame
    for (size_t i = 0; i < frames.size(); ++i) {
        if (i >= m_preprocessedImages.size()) continue;

        // Get original image dimensions
        int originalWidth = frames[i].frame->width;
        int originalHeight = frames[i].frame->height;

        // Get output for this frame
        const float* frameOutput = m_outputData.data() + i * (25200 * (4 + 1 + m_numClasses));

        // Postprocess detections
        std::vector<cv::Rect> detections = postprocessDetections(frameOutput, originalWidth, originalHeight);

        std::cout << "Frame " << i << ": Detected " << detections.size() << " objects" << std::endl;

        // Here you can store detections in FrameInfo or process them as needed
        // For now, just print detection count
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
