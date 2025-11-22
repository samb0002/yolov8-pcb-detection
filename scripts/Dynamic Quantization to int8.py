import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

original_model = "/content/best.onnx"          # FP32 model
quantized_model = "/content/best_dynamic_int8.onnx"

orig_size = os.path.getsize(original_model) / (1024 * 1024)  # MB
print(f"Original FP32 model size: {orig_size:.2f} MB")


quantize_dynamic(
    model_input=original_model,
    model_output=quantized_model,
    weight_type=QuantType.QInt8,      # INT8 weights
)


# Check quantized model size

quant_size = os.path.getsize(quantized_model) / (1024 * 1024)  # MB
print(f"Quantized INT8 model size: {quant_size:.2f} MB")

#  Compute size reduction

reduction = (orig_size - quant_size) / orig_size * 100
print(f"Size reduction: {reduction:.2f}%")
