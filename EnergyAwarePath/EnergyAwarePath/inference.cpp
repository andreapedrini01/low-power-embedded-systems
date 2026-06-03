#include "inference.h"
#include "config.h"
#include "model.h"
#include "model_exit1.h"

#include <TinyMLShield.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

namespace {
  tflite::ErrorReporter*    errorReporter = nullptr;
  const tflite::Model*      tflModel      = nullptr;
  tflite::MicroInterpreter* interpreter   = nullptr;
  TfLiteTensor*             tflInput      = nullptr;
  TfLiteTensor*             tflOutput     = nullptr;
}

namespace exit1 {
  const tflite::Model*      tflModel  = nullptr;
  tflite::MicroInterpreter* interp    = nullptr;
  TfLiteTensor*             input     = nullptr;
  TfLiteTensor*             output    = nullptr;
  float scale_in  = 0.003922f;
  int   zero_in   = -128;
  float scale_out = 0.003906f;
  int   zero_out  = -128;
}

alignas(16) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
alignas(16) static uint8_t tensor_arena_exit1[TENSOR_ARENA_EXIT1_SIZE];

static ExitLevel last_exit = EXIT_FULL;

float INPUT_SCALE  = 0.003922f;
int   INPUT_ZERO   = -128;
float OUTPUT_SCALE = 0.003906f;
int   OUTPUT_ZERO  = -128;

bool inference_init() {
  static tflite::MicroErrorReporter microErrorReporter;
  errorReporter = &microErrorReporter;
  
  tflModel = tflite::GetModel(model_data);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema mismatch: ");
    Serial.println(tflModel->version());
    return false;
  }
  
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter staticInterp(
      tflModel, resolver, tensor_arena, TENSOR_ARENA_SIZE, errorReporter);
  interpreter = &staticInterp;
  
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed");
    return false;
  }
  
  tflInput  = interpreter->input(0);
  tflOutput = interpreter->output(0);
  
  Serial.println("Model loaded");
  Serial.print("  Arena: ");
  Serial.print(interpreter->arena_used_bytes());
  Serial.print(" / ");
  Serial.println(TENSOR_ARENA_SIZE);
  
  INPUT_SCALE  = tflInput->params.scale;
  INPUT_ZERO   = tflInput->params.zero_point;
  OUTPUT_SCALE = tflOutput->params.scale;
  OUTPUT_ZERO  = tflOutput->params.zero_point;
  
  return true;
}

float inference_predict(const float features[10], unsigned long &latency_us) {
  int8_t* input_data = tflInput->data.int8;
  for (int i = 0; i < NUM_FEATURES; i++) {
    float quantized = features[i] / INPUT_SCALE + INPUT_ZERO;
    if (quantized < -128.0f) quantized = -128.0f;
    if (quantized > 127.0f) quantized = 127.0f;
    input_data[i] = (int8_t)quantized;
  }
  
  unsigned long t_start = micros();
  TfLiteStatus status = interpreter->Invoke();
  unsigned long t_end = micros();
  latency_us = t_end - t_start;
  
  if (status != kTfLiteOk) {
    Serial.println("Inference failed");
    return -1.0f;
  }
  
  int output_quant = (int)tflOutput->data.int8[0];
  float result = (float)(output_quant - OUTPUT_ZERO) * OUTPUT_SCALE;
  
  if (result < 0.0f) result = 0.0f;
  if (result > 1.0f) result = 1.0f;
  
  return result;
}

bool inference_init_exit1() {
  if (errorReporter == nullptr) {
    Serial.println("Call inference_init() first");
    return false;
  }
  
  exit1::tflModel = tflite::GetModel(model_exit1_data);
  if (exit1::tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Exit1 schema mismatch");
    return false;
  }
  
  static tflite::AllOpsResolver resolver_exit1;
  static tflite::MicroInterpreter staticInterp(
      exit1::tflModel, resolver_exit1, tensor_arena_exit1, 
      TENSOR_ARENA_EXIT1_SIZE, errorReporter);
  exit1::interp = &staticInterp;
  
  if (exit1::interp->AllocateTensors() != kTfLiteOk) {
    Serial.println("Exit1 AllocateTensors failed");
    return false;
  }
  
  exit1::input  = exit1::interp->input(0);
  exit1::output = exit1::interp->output(0);
  
  Serial.println("Exit1 loaded");
  Serial.print("  Arena: ");
  Serial.print(exit1::interp->arena_used_bytes());
  Serial.print(" / ");
  Serial.println(TENSOR_ARENA_EXIT1_SIZE);
  
  exit1::scale_in  = exit1::input->params.scale;
  exit1::zero_in   = exit1::input->params.zero_point;
  exit1::scale_out = exit1::output->params.scale;
  exit1::zero_out  = exit1::output->params.zero_point;
  
  return true;
}

float inference_predict_exit1(const float features[10], unsigned long &latency_us) {
  if (exit1::interp == nullptr) {
    Serial.println("Exit1 not initialized");
    latency_us = 0;
    return -1.0f;
  }
  
  int8_t* input_data = exit1::input->data.int8;
  for (int i = 0; i < NUM_FEATURES; i++) {
    float quantized = features[i] / exit1::scale_in + exit1::zero_in;
    if (quantized < -128.0f) quantized = -128.0f;
    if (quantized > 127.0f) quantized = 127.0f;
    input_data[i] = (int8_t)quantized;
  }
  
  unsigned long t_start = micros();
  TfLiteStatus status = exit1::interp->Invoke();
  unsigned long t_end = micros();
  latency_us = t_end - t_start;
  
  if (status != kTfLiteOk) {
    Serial.println("Exit1 inference failed");
    return -1.0f;
  }
  
  int output_quant = (int)exit1::output->data.int8[0];
  float result = (float)(output_quant - exit1::zero_out) * exit1::scale_out;
  
  if (result < 0.0f) result = 0.0f;
  if (result > 1.0f) result = 1.0f;
  
  return result;
}

ExitLevel inference_last_exit_level() {
  return last_exit;
}

float inference_predict_adaptive(const float features[10], float budget,
                                 float oracle_cost, unsigned long &latency_us) {
  if (budget >= BUDGET_HIGH_THRESHOLD) {
    last_exit = EXIT_FULL;
    return inference_predict(features, latency_us);
  } 
  else if (budget >= BUDGET_LOW_THRESHOLD) {
    last_exit = EXIT_MIDDLE;
    return inference_predict_exit1(features, latency_us);
  } 
  else {
    last_exit = EXIT_LINEAR;
    latency_us = 0;
    return oracle_cost;
  }
}
