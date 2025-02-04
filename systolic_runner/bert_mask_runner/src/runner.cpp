// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

#include <assert.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <errno.h>
#include <systolic/systolic_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#ifdef USE_CUSTOM_OP_LIBRARY
#include "custom_op_library.h"
#endif

#include "tensor_helper.h"
#include "cmd_args.h"

unsigned long long read_cycles()
{
    unsigned long long cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
}

int main(int argc, char* argv[]) {
  setbuf(stdout, NULL);
  printf("Loaded runner program\n");

  cxxopts::ParseResult cmd = parse(argc, argv);
  //*************************************************************************
  // initialize  enviroment...one enviroment per process
  // enviroment maintains thread pools and other state info
  Ort::Env env(static_cast<OrtLoggingLevel>(cmd["debug"].as<int>()), "test");

  // initialize session options if needed
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  if (cmd.count("trace")) {
    session_options.EnableProfiling(cmd["trace"].as<std::string>().c_str());
  }
  
  printf("Using systolic in mode %d\n", cmd["execution"].as<int>());

  // If onnxruntime.dll is built with CUDA enabled, we can uncomment out this line to use CUDA for this
  // session (we also need to include cuda_provider_factory.h above which defines it)
  // #include "cuda_provider_factory.h"
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Systolic(session_options, /*use_arena=*/ 1, /*accelerator_mode=*/ (char) cmd["execution"].as<int>()));

  // Sets graph optimization level
  // Available levels are
  // 0: ORT_DISABLE_ALL -> To disable all optimizations
  // 1: ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
  // 2: ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
  // 99: ORT_ENABLE_ALL -> To Enable All possible opitmizations
  session_options.SetGraphOptimizationLevel(static_cast<GraphOptimizationLevel>(cmd["optimization"].as<int>()));

  if (cmd.count("save_model")) {
    session_options.SetOptimizedModelFilePath(cmd["save_model"].as<std::string>().c_str());
  }

#ifdef USE_CUSTOM_OP_LIBRARY
  if (cmd.count("kernel")) {
    printf("Loading custom kernel\n");
    Ort::ThrowOnError(RegisterCustomOps((OrtSessionOptions*) session_options, OrtGetApiBase()));
  }
#endif

  //*************************************************************************
  // create session and load model into memory
  // using squeezenet version 1.3
  // URL = https://github.com/onnx/models/tree/master/squeezenet

  const char* model_path = cmd["model"].as<std::string>().c_str();

  printf("Using Onnxruntime C++ API\n");
  Ort::Session session(env, model_path, session_options);

  //*************************************************************************
  // print model input layer (node names, types, shape etc.)
  Ort::AllocatorWithDefaultOptions allocator;

  // print number of model input nodes
  size_t num_input_nodes = session.GetInputCount();
  std::vector<const char*> input_node_names(num_input_nodes);
  std::vector<std::vector<int64_t>> input_node_dims;

  printf("Number of inputs = %zu\n", num_input_nodes);
  // iterate over all input nodes
  for (size_t i = 0; i < num_input_nodes; i++) {
    // print input node names
    char* input_name = session.GetInputName(i, allocator);
    printf("Input %ld : name=%s, ", i, input_name);
    input_node_names[i] = input_name;

    // print input node types
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("type=%d, ", type);

    // print input shapes/dims
    input_node_dims.push_back(tensor_info.GetShape());
    printf("num_dims=%zu: [", input_node_dims[i].size());
    for (size_t j = 0; j < input_node_dims[i].size(); j++) {
      printf("%jd, ", input_node_dims[i][j]);
    }
    printf("]\n");
  }

  if (num_input_nodes != 3) {
    printf("ERROR: Graph has the wrong number of input nodes defined.\n");
    return -1;
  }

  // Results should be...
  // Number of inputs = 1
  // Input 0 : name = data_0
  // Input 0 : type = 1
  // Input 0 : num_dims = 4
  // Input 0 : dim 0 = 1
  // Input 0 : dim 1 = 3
  // Input 0 : dim 2 = 224
  // Input 0 : dim 3 = 224

  size_t num_output_nodes = session.GetOutputCount();
  printf("Number of outputs = %zu\n", num_output_nodes);
  std::vector<const char*> output_node_names(num_output_nodes);
  for (size_t i = 0; i < num_output_nodes; i++) {
    // print input node names
    char* output_name = session.GetOutputName(i, allocator);
    printf("Output %ld : name=%s, ", i, output_name);
    output_node_names[i] = output_name;

    // print input node types
    Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("type=%d, ", type);

    // print input shapes/dims
    std::vector<int64_t> output_node_dims = tensor_info.GetShape();
    printf("num_dims=%zu: [", output_node_dims.size());
    for (size_t j = 0; j < output_node_dims.size(); j++) {
      printf("%jd, ", output_node_dims[j]);
    }
    printf("]\n");
  }
  
  //*************************************************************************
  // Similar operations to get output node information.
  // Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
  // OrtSessionGetOutputTypeInfo() as shown above.

  //*************************************************************************
  // Score the model using sample data, and inspect values

  
  int64_t dims[2]; // {BATCH, SEQUENCE}
  std::ifstream fs;
  fs.open("inputs/dims.data", std::ios::in | std::ios::binary);
  fs.read((char *)&dims, sizeof(int64_t) * 2);
  fs.close();
  fs.clear();

  int64_t BATCH = dims[0];
  int64_t SEQUENCE = dims[1];

  printf("BERT batch & seq dims: %ld, %ld\n", BATCH, SEQUENCE);

  input_node_dims[0][0] = BATCH;
  input_node_dims[0][1] = SEQUENCE;
  input_node_dims[1][0] = BATCH;
  input_node_dims[1][1] = SEQUENCE;
  input_node_dims[2][0] = BATCH;
  input_node_dims[2][1] = SEQUENCE;


  size_t input_tensor_size = BATCH * SEQUENCE;

  std::vector<int64_t> input_ids_val(input_tensor_size);
  std::vector<int64_t> attention_mask_val(input_tensor_size);
  std::vector<int64_t> token_type_ids_val(input_tensor_size);

  fs.open("inputs/input_ids.data", std::ios::in | std::ios::binary);
  fs.read((char *)input_ids_val.data(), sizeof(int64_t) * input_tensor_size);
  fs.close();
  fs.clear();
  printf("The first input id vals are: %ld, %ld, %ld\n", input_ids_val[0], input_ids_val[1], input_ids_val[2]);

  fs.open("inputs/attention_mask.data", std::ios::in | std::ios::binary);
  fs.read((char *)attention_mask_val.data(), sizeof(int64_t) * input_tensor_size);
  fs.close();
  fs.clear();
  printf("The first attention mask val are: %ld, %ld, %ld\n", attention_mask_val[0], attention_mask_val[1], attention_mask_val[2]);
  
  fs.open("inputs/token_type_ids.data", std::ios::in | std::ios::binary);
  fs.read((char *)token_type_ids_val.data(), sizeof(int64_t) * input_tensor_size);
  fs.close();
  fs.clear();
  printf("The first token type ids are: %ld, %ld, %ld\n", token_type_ids_val[0], token_type_ids_val[1], token_type_ids_val[2]);


  // initialize input data with values in [0.0, 1.0]
  // for (unsigned int i = 0; i < input_tensor_size; i++)
  //   input_tensor_values[i] = (float)i / (input_tensor_size + 1);

  // create input tensor object from data values
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);


  std::vector<Ort::Value> bert_inputs;
  bert_inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory_info,
                                                       input_ids_val.data(), input_tensor_size,
                                                       input_node_dims[0].data(), input_node_dims[0].size()));
  bert_inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory_info,
                                                       attention_mask_val.data(), input_tensor_size,
                                                       input_node_dims[1].data(), input_node_dims[1].size()));
  bert_inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory_info,
                                                       token_type_ids_val.data(), input_tensor_size,
                                                       input_node_dims[2].data(), input_node_dims[2].size()));

  printf("Starting inference\n");

  auto pre_inference_cycles = read_cycles();

  // score model & input tensor, get back output tensor
  auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), bert_inputs.data(), bert_inputs.size(), output_node_names.data(), output_node_names.size());
  auto post_inference_cycles = read_cycles();

  printf("Number of output tensors %lld\n", output_tensors.size());

  for (int i = 0; i < output_tensors.size(); i++) {
    printf("Dimensions\n");
    std::vector<int64_t> output_node_dims = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
    printf("num_dims=%zu: [", output_node_dims.size());
    for (size_t j = 0; j < output_node_dims.size(); j++) {
      printf("%jd, ", output_node_dims[j]);
    }
    printf("]\n");
  }

  float *probs = output_tensors[0].GetTensorMutableData<float>();
  FILE *f = fopen("outputs/output.data", "wb");
  if (f == nullptr) {
    printf("Failed to open output file\n");
    printf("Error %d \n", errno);
    return 1;
  }
  fwrite(probs, sizeof(float), output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount(), f);
  fclose(f);

  printf("Done! Inference took %llu cycles \n", (post_inference_cycles - pre_inference_cycles));
  return 0;
}
