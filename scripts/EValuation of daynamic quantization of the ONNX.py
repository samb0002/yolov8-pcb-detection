"""
This code is to fidn if the layers and weitghts are quantized or not 
NOTE: Dynamic quantization only quantize linear layers and weights, but does not quantize the Conv2d layers of the CNN architectuer of the YOLOv8 model
"""
import onnx

model_path = "/content/best_dynamic_int8.onnx"
model = onnx.load(model_path)

quantized_layers = 0
for tensor in model.graph.initializer:
    if "weight_quantized" in tensor.name or "weight_zero_point" in tensor.name:
        quantized_layers += 1

print(f"Total quantized layers found: {quantized_layers}")
