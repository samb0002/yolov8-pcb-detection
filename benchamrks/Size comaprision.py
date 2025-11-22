import os
fp32_size = os.path.getsize("/content/best.onnx") / (1024**2)
int8_size = os.path.getsize("/content/best_dynamic_int8.onnx") / (1024**2)
print(f"FP32: {fp32_size:.2f} MB, INT8: {int8_size:.2f} MB")
reduction=(fp32_size-int8_size)/fp32_size*100
print(f"The reduction in oercentage is : {reduction:.2f}%")
