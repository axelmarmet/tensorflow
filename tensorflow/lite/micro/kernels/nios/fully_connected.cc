/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/kernels/fully_connected.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

extern "C" {
  #include "system.h"
  #include "io.h"
  #include "altera_avalon_pio_regs.h"
  #include "altera_avalon_performance_counter.h"
  #include <sys/alt_cache.h>
  #include "QML_accelerator.h"
  #include "stdio.h"
}

namespace tflite {
namespace {

#define KERNEL_SECTION 1
#define LOOP_SECTION 2

inline void FullyConnectedSoft(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 2);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = output_shape.Dims(0);
  const int output_depth = output_shape.Dims(1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);

  if (accum_depth == 1 && output_depth == 16) {
    PERF_BEGIN(CPU_0_SUBSYSTEM_PERFORMANCE_COUNTER_INST_BASE, LOOP_SECTION);
  }
  // printf("int32_t interm[] = {\n\t");

  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      int32_t acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += (filter_val + filter_offset) * (input_val + input_offset);
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      // printf("0 : %ld, ", acc);
      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<int8_t>(acc);
    }
  }
  // printf("};\n");


  // printf("int32_t output_depth = %d\n", output_depth);
  // printf("int32_t accum_depth = %d\n", accum_depth);
  // printf("int32_t input_offset = %ld\n", input_offset);
  // printf("int32_t filter_offset = %ld\n", filter_offset);
  // printf("int32_t output_offset = %ld\n", output_offset);
  // printf("int32_t output_multiplier = %ld\n", output_multiplier);
  // printf("int32_t output_shift = %d\n", output_shift);
  // printf("int32_t output_activation_min = %ld\n", output_activation_min);
  // printf("int32_t output_activation_max = %ld\n", output_activation_max);

  // printf("constant input_memory: byte_memory(%d downto 0) := (\n", accum_depth - 1);
  // for (int d = 0; d < accum_depth; ++d) {
  //   printf("\t %d => std_logic_vector(to_signed(%d, 8)),\n", d, input_data[d]);
  // }
  // printf(");\n");

  // printf("constant weight_memory: byte_memory(%d downto 0) := (\n", output_depth * accum_depth - 1);
  // for (int out_c = 0; out_c < output_depth; ++out_c) {
  //   for (int d = 0; d < accum_depth; ++d) {
  //     printf("\t %d => std_logic_vector(to_signed(%d, 8)),\n", out_c * accum_depth + d, filter_data[out_c * accum_depth + d]);
  //   }
  // }
  // printf(");\n");

  // printf("constant formal_bias_memory: word_memory(%d downto 0) := (\n", output_depth - 1);
  // for (int out_c = 0; out_c < output_depth; ++out_c) {
  //   printf("\t %d => std_logic_vector(to_signed(%ld, 32)),\n", out_c, bias_data[out_c]);
  // }
  // printf(");\n");

  // printf("constant expected_output: byte_memory(%d downto 0) := (\n", output_depth - 1);
  // for (int out_c = 0; out_c < output_depth; ++out_c) {
  //   printf("\t %d => std_logic_vector(to_signed(%d, 8)),\n", out_c, output_data[out_c]);
  // }
  // printf(");\n");

  if (accum_depth == 1 && output_depth == 16) {
    PERF_END(CPU_0_SUBSYSTEM_PERFORMANCE_COUNTER_INST_BASE, LOOP_SECTION);
  }
}

inline void FullyConnectedHard(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 2);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = output_shape.Dims(0);
  const int output_depth = output_shape.Dims(1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);

  assert(batches == 1);
  assert(output_depth < 1024);
  assert(accum_depth < 1024);

  if (accum_depth == 1 && output_depth == 16) {
    PERF_BEGIN(CPU_0_SUBSYSTEM_PERFORMANCE_COUNTER_INST_BASE, LOOP_SECTION);
  }
  
  // printf("int32_t output_depth = %d\n", output_depth);
  // printf("int32_t accum_depth = %d\n", accum_depth);
  // printf("int32_t input_offset = %ld\n", input_offset);
  // printf("int32_t filter_offset = %ld\n", filter_offset);
  // printf("int32_t output_offset = %ld\n", output_offset);
  // printf("int32_t output_multiplier = %ld\n", output_multiplier);
  // printf("int32_t output_shift = %d\n", output_shift);
  // printf("int32_t output_activation_min = %ld\n", output_activation_min);
  // printf("int32_t output_activation_max = %ld\n", output_activation_max);

  alt_dcache_flush_all();

  write_config(CPU_0_SUBSYSTEM_QML_ACCELERATOR_0_BASE, accum_depth, output_depth, 0);
  write_weight_address(CPU_0_SUBSYSTEM_QML_ACCELERATOR_0_BASE, filter_data);
  write_bias_address(CPU_0_SUBSYSTEM_QML_ACCELERATOR_0_BASE, bias_data);
  write_input_address(CPU_0_SUBSYSTEM_QML_ACCELERATOR_0_BASE, input_data);
  write_result_address(CPU_0_SUBSYSTEM_QML_ACCELERATOR_0_BASE, output_data);
  write_zeros(CPU_0_SUBSYSTEM_QML_ACCELERATOR_0_BASE, output_offset, input_offset, filter_offset);
  write_mo(CPU_0_SUBSYSTEM_QML_ACCELERATOR_0_BASE, output_multiplier);
  write_m_exposant(CPU_0_SUBSYSTEM_QML_ACCELERATOR_0_BASE, output_shift);

  start_accelerator(CPU_0_SUBSYSTEM_QML_ACCELERATOR_0_BASE);

  while(!is_accelerator_idle(CPU_0_SUBSYSTEM_QML_ACCELERATOR_0_BASE)) {}

  // printf("constant input_memory: byte_memory(%d downto 0) := (\n", accum_depth - 1);
  // for (int d = 0; d < accum_depth; ++d) {
  //   printf("\t %d => std_logic_vector(to_signed(%d, 8)),\n", d, input_data[d]);
  // }
  // printf(");\n");

  // printf("constant weight_memory: byte_memory(%d downto 0) := (\n", output_depth * accum_depth - 1);
  // for (int out_c = 0; out_c < output_depth; ++out_c) {
  //   for (int d = 0; d < accum_depth; ++d) {
  //     printf("\t %d => std_logic_vector(to_signed(%d, 8)),\n", out_c * accum_depth + d, filter_data[out_c * accum_depth + d]);
  //   }
  // }
  // printf(");\n");

  // printf("constant formal_bias_memory: word_memory(%d downto 0) := (\n", output_depth - 1);
  // for (int out_c = 0; out_c < output_depth; ++out_c) {
  //   printf("\t %d => std_logic_vector(to_signed(%ld, 32)),\n", out_c, bias_data[out_c]);
  // }
  // printf(");\n");

  // printf("constant expected_output: byte_memory(%d downto 0) := (\n", output_depth - 1);
  // for (int out_c = 0; out_c < output_depth; ++out_c) {
  //   printf("\t %d => std_logic_vector(to_signed(%d, 8)),\n", out_c, output_data[out_c]);
  // }
  // printf(");\n");

  if (accum_depth == 1 && output_depth == 16) {
    PERF_END(CPU_0_SUBSYSTEM_PERFORMANCE_COUNTER_INST_BASE, LOOP_SECTION);
  }
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context,
                                           sizeof(OpDataFullyConnected));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* data = static_cast<OpDataFullyConnected*>(node->user_data);
  const auto params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  const TfLiteTensor* input =
      GetInput(context, node, kFullyConnectedInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  const TfLiteTensor* filter =
      GetInput(context, node, kFullyConnectedWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);
  const TfLiteTensor* bias =
      GetOptionalInputTensor(context, node, kFullyConnectedBiasTensor);
  TfLiteTensor* output = GetOutput(context, node, kFullyConnectedOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(context, input->type == filter->type,
                     "Hybrid models are not supported on TFLite Micro.");

  return CalculateOpDataFullyConnected(context, params->activation, input->type,
                                       input, filter, bias, output, data);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  int pio_value = IORD_ALTERA_AVALON_PIO_DATA(CPU_0_SUBSYSTEM_HEARTBEAT_BASE);
  ++pio_value;
  IOWR_ALTERA_AVALON_PIO_DATA(CPU_0_SUBSYSTEM_HEARTBEAT_BASE, pio_value);

  PERF_BEGIN(CPU_0_SUBSYSTEM_PERFORMANCE_COUNTER_INST_BASE, KERNEL_SECTION);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto* params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedBiasTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kFullyConnectedOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& data =
      *(static_cast<const OpDataFullyConnected*>(node->user_data));

  // Checks in Prepare ensure input, output and filter types are all the same.
  switch (input->type) {
    printf("doing float \n");
    case kTfLiteFloat32: {
      tflite::reference_ops::FullyConnected(
          FullyConnectedParamsFloat(params->activation),
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<float>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetTensorData<float>(bias),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output));
      break;
    }

    case kTfLiteInt8: {
      printf("doing quant \n");
      FullyConnectedHard(
          FullyConnectedParamsQuantized(data),
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<int8_t>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetTensorData<int32_t>(bias),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
      break;
    }

    default: {
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
    }
  }

  PERF_END(CPU_0_SUBSYSTEM_PERFORMANCE_COUNTER_INST_BASE, KERNEL_SECTION);

  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_FULLY_CONNECTED() {
  return {/*init=*/Init,
          /*free=*/nullptr,
          /*prepare=*/Prepare,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
