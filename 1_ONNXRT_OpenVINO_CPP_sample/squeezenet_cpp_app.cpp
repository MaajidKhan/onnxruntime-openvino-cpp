// https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
// https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_cxx_api.h
#include <onnxruntime_cxx_api.h>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

std::vector<std::string> readLabels(std::string& labelFilepath)
{
    std::vector<std::string> labels;
    std::string line;
    std::ifstream fp(labelFilepath);
    while (std::getline(fp, line))
    {
        labels.push_back(line);
    }
    return labels;
}

int main(int argc, char* argv[])
{
    bool useOPENVINO{true};
    const char* useOPENVINOFlag = "--use_openvino";
    const char* useCPUFlag = "--use_cpu";
    if (argc == 1)
    {
        useOPENVINO = false;
    }
    else if ((argc == 2) && (strcmp(argv[1], useOPENVINOFlag) == 0))
    {
        useOPENVINO = true;
    }
    else if ((argc == 2) && (strcmp(argv[1], useCPUFlag) == 0))
    {
        useOPENVINO = false;
    }
    else if ((argc == 2) && (strcmp(argv[1], useOPENVINOFlag) != 0))
    {
        useOPENVINO = false;
    }
    else
    {
        throw std::runtime_error{"Too many arguments."};
    }

    if (useOPENVINO)
    {
        std::cout << "Inference Execution Provider: OPENVINO" << std::endl;
    }
    else
    {
        std::cout << "Inference Execution Provider: CPU" << std::endl;
    }

    std::string instanceName{"image-classification-inference"};

    //Full path to the model
    std::string modelFilepath{"/home/maajid/project_files/ov_onnxruntime_issues_2021_Jan1/onnxruntime/build/Linux/Debug/squeezenet1.1-7.onnx"};

    //Full path to the image you want to run the inference on
    std::string imageFilepath{"/home/maajid/project_files/ov_onnxruntime_issues_2021_Jan1/onnxruntime/build/Linux/Debug/cars.jpg"};

    //Full path to the 1000 class labels from ImageNet dataset for squeezenet model
    std::string labelFilepath{"/home/maajid/project_files/ov_onnxruntime_issues_2021_Jan1/onnxruntime/build/Linux/Debug/synset.txt"};

    std::vector<std::string> labels{readLabels(labelFilepath)};

    // https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L123
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                 instanceName.c_str());
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);

    //Appending OpenVINO Execution Provider API
    if (useOPENVINO)
    {
        // Using OPENVINO backend
        // https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/providers/openvino/openvino_provider_factory.h
        OrtOpenVINOProviderOptions options;
        options.device_type = "CPU_FP32";
        //options.num_of_threads = 6;
        //options.enable_vpu_fast_compile = true;
        sessionOptions.AppendExecutionProvider_OpenVINO(options);
    }

    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node
    // removals) ORT_ENABLE_EXTENDED -> To enable extended optimizations
    // (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible optimizations
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_DISABLE_ALL);

    //Creation: The Ort::Session is created here
    Ort::Session session(env, modelFilepath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();

    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;

    const char* inputName = session.GetInputName(0, allocator);
    std::cout << "Input Name: " << inputName << std::endl;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    std::cout << "Input Type: " << inputType << std::endl;

    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    std::cout << "Input Dimensions: " << inputDims << std::endl;

    const char* outputName = session.GetOutputName(0, allocator);
    std::cout << "Output Name: " << outputName << std::endl;

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    std::cout << "Output Type: " << outputType << std::endl;

    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    std::cout << "Output Dimensions: " << outputDims << std::endl;
    //pre-processing the Image
    // step 1: Read an image in HWC BGR UINT8 format.
    cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);

    // step 2: Resize the image.
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
    cv::resize(imageBGR, resizedImageBGR,
               cv::Size(inputDims.at(2), inputDims.at(3)),
               cv::InterpolationFlags::INTER_CUBIC);

    // step 3: Convert the image to HWC RGB UINT8 format.
    cv::cvtColor(resizedImageBGR, resizedImageRGB,
                 cv::ColorConversionCodes::COLOR_BGR2RGB);
    // step 4: Convert the image to HWC RGB float format by dividing each pixel by 255.
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

    // step 5: Split the RGB channels from the image.   
    cv::Mat channels[3];
    cv::split(resizedImage, channels);

    //step 6: Normalize each channel.
    // Normalization per channel
    // Normalization parameters obtained from
    // https://github.com/onnx/models/tree/master/vision/classification/squeezenet
    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;

    //step 7: Merge the RGB channels back to the image.
    cv::merge(channels, 3, resizedImage);

    // step 8: Convert the image to CHW RGB float format.
    // HWC to CHW
    cv::dnn::blobFromImage(resizedImage, preprocessedImage);


    //Run Inference

    /* To run inference using ONNX Runtime, the user is responsible for creating and managing the 
    input and output buffers. These buffers could be created and managed via std::vector.
    The linear-format input data should be copied to the buffer for ONNX Runtime inference. */

    size_t inputTensorSize = vectorProduct(inputDims);
    std::vector<float> inputTensorValues(inputTensorSize);
    inputTensorValues.assign(preprocessedImage.begin<float>(),
                             preprocessedImage.end<float>());

    size_t outputTensorSize = vectorProduct(outputDims);
    assert(("Output tensor size should equal to the label set size.",
            labels.size() == outputTensorSize));
    std::vector<float> outputTensorValues(outputTensorSize);


    /* Once the buffers were created, they would be used for creating instances of Ort::Value 
    which is the tensor format for ONNX Runtime. There could be multiple inputs for a neural network, 
    so we have to prepare an array of Ort::Value instances for inputs and outputs respectively even if 
    we only have one input and one output. */

    std::vector<const char*> inputNames{inputName};
    std::vector<const char*> outputNames{outputName};
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    /*
    Creating ONNX Runtime inference sessions, querying input and output names, 
    dimensions, and types are trivial.
    Setup inputs & outputs: The input & output tensors are created here. */

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
        inputDims.size()));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, outputTensorValues.data(), outputTensorSize,
        outputDims.data(), outputDims.size()));

    /* To run inference, we provide the run options, an array of input names corresponding to the 
    inputs in the input tensor, an array of input tensor, number of inputs, an array of output names 
    corresponding to the the outputs in the output tensor, an array of output tensor, number of outputs.
    https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_cxx_api.h#L353 */

    session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                inputTensors.data(), 1, outputNames.data(),
                outputTensors.data(), 1);

    int predId = 0;
    float activation = 0;
    float maxActivation = std::numeric_limits<float>::lowest();
    float expSum = 0;
    /* The inference result could be found in the buffer for the output tensors, 
    which are usually the buffer from std::vector instances. */
    for (int i = 0; i < labels.size(); i++)
    {
        activation = outputTensorValues.at(i);
        expSum += std::exp(activation);
        if (activation > maxActivation)
        {
            predId = i;
            maxActivation = activation;
        }
    }
    std::cout << "Predicted Label ID: " << predId << std::endl;
    std::cout << "Predicted Label: " << labels.at(predId) << std::endl;
    std::cout << "Uncalibrated Confidence: " << std::exp(maxActivation) / expSum
              << std::endl;

    // Measure latency
    int numTests{100};
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();

    //Run: Running the session is done in the Run() method:
    for (int i = 0; i < numTests; i++)
    {
        session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                    inputTensors.data(), 1, outputNames.data(),
                    outputTensors.data(), 1);
    }
    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    std::cout << "Minimum Inference Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       begin)
                         .count() /
                     static_cast<float>(numTests)
              << " ms" << std::endl;
}
