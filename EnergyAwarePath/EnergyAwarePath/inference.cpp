#include "inference.h"
#include "config.h"
#include "model.h"

// TensorFlow Lite Micro (Harvard_TinyMLx library)
#include <TinyMLShield.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// TFLite Micro globals
namespace {
  tflite::ErrorReporter*    errorReporter = nullptr;
  const tflite::Model*      tflModel      = nullptr;
  tflite::MicroInterpreter* interpreter   = nullptr;
  TfLiteTensor*             tflInput      = nullptr;
  TfLiteTensor*             tflOutput     = nullptr;
}

// Tensor arena (static allocation)
alignas(16) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

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
