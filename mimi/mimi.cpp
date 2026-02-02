// Header file for input output functions
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <chrono>
#include "onnxruntime/onnxruntime_cxx_api.h"

void get_tensors(Ort::Session& session, Ort::MemoryInfo& memory_info, std::vector<void*> allocated_buffers,
        std::vector<Ort::Value>& input_tensors, std::vector<const char*>& input_node_names,
        std::vector<Ort::Value>& output_tensors, std::vector<const char*>& output_node_names) {
    for (int i = 0; i < session.GetInputCount(); i++) {
        auto name = session.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions()).get();
        char* buffer = new char[strlen(name) + 1];
        allocated_buffers.push_back(buffer);
        strcpy(buffer, name);
        input_node_names.emplace_back(buffer);

        auto type = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
        auto shape = type.GetShape();
        if (type.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
            char* buffer = new char[type.GetElementCount()];
            allocated_buffers.push_back(buffer);
            memset(buffer, 1, type.GetElementCount());
            input_tensors.emplace_back(Ort::Value::CreateTensor<bool>(memory_info, (bool*) buffer, type.GetElementCount(), shape.data(), shape.size()));
        } else if (type.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            int64_t* buffer = new int64_t[type.GetElementCount()];
            allocated_buffers.push_back(buffer);
            memset(buffer, 0, type.GetElementCount() * sizeof(int64_t));
            input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info, buffer, type.GetElementCount(), shape.data(), shape.size()));
        } else if (type.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            float* buffer = new float[type.GetElementCount()];
            allocated_buffers.push_back(buffer);
            memset(buffer, 0, type.GetElementCount() * sizeof(float));
            input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, buffer, type.GetElementCount(), shape.data(), shape.size()));
        } else {
            std::cerr << "Unsupported input tensor type: " << type.GetElementType() << std::endl;
            exit(-1);
        }
        // auto last = (input_tensors.end() - 1)->GetTensorTypeAndShapeInfo();
        // std::cout << "Input tensor " << buffer << " shape: " << last.GetElementCount() << " type: " << last.GetElementType() << std::endl;
    }
    for (int i = 0; i < session.GetOutputCount(); i++) {
        auto name = session.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions()).get();
        char* buffer = new char[strlen(name) + 1];
        allocated_buffers.push_back(buffer);
        strcpy(buffer, name);
        output_node_names.emplace_back(buffer);

        auto type = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
        auto shape = type.GetShape();
        if (type.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
            char* buffer = new char[type.GetElementCount()];
            allocated_buffers.push_back(buffer);
            memset(buffer, 1, type.GetElementCount());
            output_tensors.emplace_back(Ort::Value::CreateTensor<bool>(memory_info, (bool*) buffer, type.GetElementCount(), shape.data(), shape.size()));
        } else if (type.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            int64_t* buffer = new int64_t[type.GetElementCount()];
            allocated_buffers.push_back(buffer);
            memset(buffer, 0, type.GetElementCount() * sizeof(int64_t));
            output_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info, buffer, type.GetElementCount(), shape.data(), shape.size()));
        } else if (type.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            float* buffer = new float[type.GetElementCount()];
            allocated_buffers.push_back(buffer);
            memset(buffer, 0, type.GetElementCount() * sizeof(float));
            output_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, buffer, type.GetElementCount(), shape.data(), shape.size()));
        } else {
            std::cerr << "Unsupported input tensor type: " << type.GetElementType() << std::endl;
            exit(-1);
        }

        // std::cout << "Output tensor " << buffer << std::endl;
    }
}

// Main function: entry point for execution
int main() {
    // Writing print statement to print hello world
    std::cout << "Mimi ONNX Runtime Inference" << std::endl;

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Env env;
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session encoder(env, "/data/mimi_encoder.onnx", options);
    Ort::Session decoder(env, "/data/mimi_decoder.onnx", options);
    Ort::RunOptions run_options;

    std::ifstream pcm_stream("/data/music.pcm", std::ios::binary);
    if (!pcm_stream) {
        std::cerr << "Failed to open input.pcm file." << std::endl;
        return -1;
    }

    std::vector<const char*> encoder_inputs;
    std::vector<const char*> encoder_outputs;
    std::vector<Ort::Value> encoder_tensors;
    std::vector<Ort::Value> encoder_output_tensors;

    std::vector<const char*> decoder_inputs;
    std::vector<const char*> decoder_outputs;
    std::vector<Ort::Value> decoder_tensors;
    std::vector<Ort::Value> decoder_output_tensors;

    std::vector<void*> allocated_buffers;
    
    get_tensors(encoder, memory_info, allocated_buffers, encoder_tensors, encoder_inputs, encoder_output_tensors, encoder_outputs);
    get_tensors(decoder, memory_info, allocated_buffers, decoder_tensors, decoder_inputs, decoder_output_tensors, decoder_outputs);

    short pcm_data;
    float inputData[1920];
    int offset = 0;
    int64_t encoder_shape[3] = {1, 1, 1920};
    int64_t decoder_shape[3] = {1, 8, 1};
    int64_t decoder_codes[8];
    int max_elapsed = 0;

    std::vector<short> output;
    while (pcm_stream.read(reinterpret_cast<char*>(&pcm_data), sizeof(short))) {
        // little endian
        inputData[offset++] = static_cast<float>(pcm_data) / 32768.0f;

        if (offset == 1920) {
            offset = 0;
            auto start = std::chrono::system_clock::now();

            memcpy(encoder_tensors[0].GetTensorMutableData<float>(), inputData, 1920 * sizeof(float));
            encoder_tensors[0] = Ort::Value::CreateTensor<float>(memory_info, inputData, 1920, encoder_shape, 3);
            encoder.Run(run_options, encoder_inputs.data(), encoder_tensors.data(), encoder_inputs.size(), encoder_outputs.data(), encoder_output_tensors.data(), encoder_output_tensors.size());

            int64_t* codes = (int64_t*) encoder_output_tensors[0].GetTensorData<int64_t>();
            // std::cout << "Codes " << codes[0] << " " << codes[1] << " " << codes[2] << " " << codes[3] << " " << codes[4] << " " << codes[5] << " " << codes[6] << " " << codes[7] << std::endl;
            for (int i = 1; i < encoder_tensors.size(); i++) {
                std::swap(encoder_tensors[i], encoder_output_tensors[i]);
            }
            for (int i = 0; i < 8; i++) {
                decoder_codes[i] = codes[i];
            }
            decoder_tensors[0] = Ort::Value::CreateTensor<int64_t>(memory_info, decoder_codes, 8, decoder_shape, 3);
            decoder.Run(run_options, decoder_inputs.data(), decoder_tensors.data(), decoder_inputs.size(), decoder_outputs.data(), decoder_output_tensors.data(), decoder_output_tensors.size());

            for (int i = 1; i < decoder_tensors.size(); i++) {
                std::swap(decoder_tensors[i], decoder_output_tensors[i]);
            }

            // std::cout << "Decoder output tensor shape: " << decoder_output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount() << " type: " << decoder_output_tensors[0].GetTensorTypeAndShapeInfo().GetElementType() << std::endl;
            float* data = (float*) decoder_output_tensors[0].GetTensorData<float>();
            for (int i = 0; i < 1920; i++) {
                float sample = data[i];
                if (sample > 1.0f) sample = 1.0f;
                if (sample < -1.0f) sample = -1.0f;
                short int_sample = (short)(sample * 32768.0f);
                output.push_back(int_sample);
            }

            auto end = std::chrono::system_clock::now();

            std::chrono::duration<double> elapsed_seconds = end-start;
            int elapsed_ms = (int) (elapsed_seconds.count() * 1000);
            if (elapsed_ms > max_elapsed) {
                max_elapsed = elapsed_ms;
            }
            std::cout << " Elapsed time: " << elapsed_ms << "ms" << std::endl;
        }
    }

    pcm_stream.close();
    for (auto buffer : allocated_buffers) {
        delete[] buffer;
    }

    std::cout << "Max elapsed time: " << max_elapsed << "ms" << std::endl;

    std::ofstream output_pcm("/data/output.pcm", std::ios::binary);
    output_pcm.write((char*) output.data(), output.size() * sizeof(short));
    output_pcm.close();

    return 0;
}