#include "inference.h"
#include "config.h"
#include "model.h"
#include "model_exit1.h"

// TensorFlow Lite Micro (Harvard_TinyMLx library)
#include <TinyMLShield.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// TFLite Micro globals (full model)
namespace {
  tflite::ErrorReporter*    errorReporter = nullptr;
  const tflite::Model*      tflModel      = nullptr;
  tflite::MicroInterpreter* interpreter   = nullptr;
  TfLiteTensor*             tflInput      = nullptr;
  TfLiteTensor*             tflOutput     = nullptr;
}

// TFLite Micro globals (exit 1 model)
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

// Tensor arenas (static allocation)
alignas(16) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
alignas(16) static uint8_t tensor_arena_exit1[TENSOR_ARENA_EXIT1_SIZE];

// Track last exit level used
static ExitLevel last_exit = EXIT_FULL;

// Quantization parameters (declared extern in config.h)
float INPUT_SCALE  = 0.003922f;
int   INPUT_ZERO   = -128;
float OUTPUT_SCALE = 0.003906f;
int   OUTPUT_ZERO  = -128;

bool inference_init() {
  // 1. Setup error reporter
  static tflite::MicroErrorReporter microErrorReporter;
  errorReporter = &microErrorReporter;
  
  // 2. Load model
  tflModel = tflite::GetModel(model_data);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERROR: Model schema version mismatch. Expected ");
    Serial.print(TFLITE_SCHEMA_VERSION);
    Serial.print(", got ");
    Serial.println(tflModel->version());
    return false;
  }
  
  // 3. Setup interpreter
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter staticInterp(
      tflModel, resolver, tensor_arena, TENSOR_ARENA_SIZE, errorReporter);
  interpreter = &staticInterp;
  
  // 4. Allocate tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed! Increase TENSOR_ARENA_SIZE.");
    return false;
  }
  
  // 5. Get input/output tensor pointers
  tflInput  = interpreter->input(0);
  tflOutput = interpreter->output(0);
  
  // 6. Print tensor info
  Serial.println("TFLite model loaded successfully");
  Serial.print("  Arena used: ");
  Serial.print(interpreter->arena_used_bytes());
  Serial.print(" / ");
  Serial.print(TENSOR_ARENA_SIZE);
  Serial.println(" bytes");
  Serial.print("  Input dims: ");
  Serial.print(tflInput->dims->data[1]);
  Serial.print(", type=");
  Serial.println(tflInput->type);
  Serial.print("  Output dims: ");
  Serial.print(tflOutput->dims->data[1]);
  Serial.print(", type=");
  Serial.println(tflOutput->type);
  
  // 7. Read quantization params from model
  INPUT_SCALE  = tflInput->params.scale;
  INPUT_ZERO   = tflInput->params.zero_point;
  OUTPUT_SCALE = tflOutput->params.scale;
  OUTPUT_ZERO  = tflOutput->params.zero_point;
  
  Serial.print("  Input quant: scale=");
  Serial.print(INPUT_SCALE, 6);
  Serial.print(", zp=");
  Serial.println(INPUT_ZERO);
  Serial.print("  Output quant: scale=");
  Serial.print(OUTPUT_SCALE, 6);
  Serial.print(", zp=");
  Serial.println(OUTPUT_ZERO);
  
  return true;
}

float inference_predict(const float features[10], unsigned long &latency_us) {
  // Quantize input: int8_val = (float_val / scale) + zero_point
  int8_t* input_data = tflInput->data.int8;
  for (int i = 0; i < NUM_FEATURES; i++) {
    float quantized = features[i] / INPUT_SCALE + INPUT_ZERO;
    // Clamp to int8 range
    if (quantized < -128.0f) quantized = -128.0f;
    if (quantized > 127.0f) quantized = 127.0f;
    input_data[i] = (int8_t)quantized;
  }
  
  // Run inference with timing
  unsigned long t_start = micros();
  TfLiteStatus status = interpreter->Invoke();
  unsigned long t_end = micros();
  latency_us = t_end - t_start;
  
  if (status != kTfLiteOk) {
    Serial.println("ERROR: Inference failed!");
    return -1.0f;
  }
  
  // Dequantize output: float_val = (int8_val - zero_point) * scale
  // Cast to int to avoid int8 overflow when subtracting negative zero_point
  int output_quant = (int)tflOutput->data.int8[0];
  float result = (float)(output_quant - OUTPUT_ZERO) * OUTPUT_SCALE;
  
  // Clamp to valid range
  if (result < 0.0f) result = 0.0f;
  if (result > 1.0f) result = 1.0f;
  
  return result;
}

// =============================================================================
// EXIT 1 MODEL INITIALIZATION AND INFERENCE
// =============================================================================

bool inference_init_exit1() {
  // Reuse the same error reporter
  if (errorReporter == nullptr) {
    Serial.println("ERROR: Call inference_init() before inference_init_exit1()");
    return false;
  }
  
  // Load exit1 model
  exit1::tflModel = tflite::GetModel(model_exit1_data);
  if (exit1::tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERROR: Exit1 model schema version mismatch. Expected ");
    Serial.print(TFLITE_SCHEMA_VERSION);
    Serial.print(", got ");
    Serial.println(exit1::tflModel->version());
    return false;
  }
  
  // Setup interpreter for exit1
  static tflite::AllOpsResolver resolver_exit1;
  static tflite::MicroInterpreter staticInterp(
      exit1::tflModel, resolver_exit1, tensor_arena_exit1, 
      TENSOR_ARENA_EXIT1_SIZE, errorReporter);
  exit1::interp = &staticInterp;
  
  // Allocate tensors
  if (exit1::interp->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: Exit1 AllocateTensors() failed! Increase TENSOR_ARENA_EXIT1_SIZE.");
    return false;
  }
  
  // Get input/output tensor pointers
  exit1::input  = exit1::interp->input(0);
  exit1::output = exit1::interp->output(0);
  
  // Print tensor info
  Serial.println("Exit1 model loaded successfully");
  Serial.print("  Arena used: ");
  Serial.print(exit1::interp->arena_used_bytes());
  Serial.print(" / ");
  Serial.print(TENSOR_ARENA_EXIT1_SIZE);
  Serial.println(" bytes");
  Serial.print("  Input dims: ");
  Serial.print(exit1::input->dims->data[1]);
  Serial.print(", type=");
  Serial.println(exit1::input->type);
  Serial.print("  Output dims: ");
  Serial.print(exit1::output->dims->data[1]);
  Serial.print(", type=");
  Serial.println(exit1::output->type);
  
  // Read quantization params
  exit1::scale_in  = exit1::input->params.scale;
  exit1::zero_in   = exit1::input->params.zero_point;
  exit1::scale_out = exit1::output->params.scale;
  exit1::zero_out  = exit1::output->params.zero_point;
  
  Serial.print("  Input quant: scale=");
  Serial.print(exit1::scale_in, 6);
  Serial.print(", zp=");
  Serial.println(exit1::zero_in);
  Serial.print("  Output quant: scale=");
  Serial.print(exit1::scale_out, 6);
  Serial.print(", zp=");
  Serial.println(exit1::zero_out);
  
  return true;
}

float inference_predict_exit1(const float features[10], unsigned long &latency_us) {
  if (exit1::interp == nullptr) {
    Serial.println("ERROR: Exit1 model not initialized!");
    latency_us = 0;
    return -1.0f;
  }
  
  // Quantize input
  int8_t* input_data = exit1::input->data.int8;
  for (int i = 0; i < NUM_FEATURES; i++) {
    float quantized = features[i] / exit1::scale_in + exit1::zero_in;
    if (quantized < -128.0f) quantized = -128.0f;
    if (quantized > 127.0f) quantized = 127.0f;
    input_data[i] = (int8_t)quantized;
  }
  
  // Run inference with timing
  unsigned long t_start = micros();
  TfLiteStatus status = exit1::interp->Invoke();
  unsigned long t_end = micros();
  latency_us = t_end - t_start;
  
  if (status != kTfLiteOk) {
    Serial.println("ERROR: Exit1 inference failed!");
    return -1.0f;
  }
  
  // Dequantize output
  int output_quant = (int)exit1::output->data.int8[0];
  float result = (float)(output_quant - exit1::zero_out) * exit1::scale_out;
  
  // Clamp to valid range
  if (result < 0.0f) result = 0.0f;
  if (result > 1.0f) result = 1.0f;
  
  return result;
}

// =============================================================================
// ADAPTIVE INFERENCE
// =============================================================================

ExitLevel inference_last_exit_level() {
  return last_exit;
}

float inference_predict_adaptive(const float features[10], float budget,
                                 float oracle_cost, unsigned long &latency_us) {
  if (budget >= BUDGET_HIGH_THRESHOLD) {
    // High budget: use full model
    last_exit = EXIT_FULL;
    return inference_predict(features, latency_us);
  } 
  else if (budget >= BUDGET_LOW_THRESHOLD) {
    // Medium budget: use exit 1
    last_exit = EXIT_MIDDLE;
    return inference_predict_exit1(features, latency_us);
  } 
  else {
    // Low budget: use oracle formula (no TFLite, zero cost)
    last_exit = EXIT_LINEAR;
    latency_us = 0;
    return oracle_cost;
  }
}
