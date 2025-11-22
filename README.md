
#  YOLOv8 PCB Defect Detection: INT8 Quantization Pipeline

##  Project Overview
This repository documents and implements a complete machine learning workflow for **Printed Circuit Board (PCB) Defect Detection**.  
The core objective is to train a high-performance **YOLOv8** model and then optimize it for **low-latency and hardware deployment** using **INT8 Quantization**.

The pipeline takes raw data, trains a custom model, and converts the resulting weights into highly optimized, reduces size files.

---

##  System Architecture and Justification of Technical Choices

### Overall Architecture
The system follows a standard ML pipeline optimized for **Edge/Embedded deployment**, divided into three phases:
1. Data Acquisition  
2. Training & Optimization  
3. Deployment  

### Justification of Choices

| Component              | Technical Choice                     | Justification                                                                 |
|------------------------|--------------------------------------|-------------------------------------------------------------------------------|
| **Base Model**         | YOLOv8s (Small)                      | A 72 layers based CNN model that provides best balance between accuracy (mAP) and inference speed (FPS). Small variant is ideal for quantization. |
| **Quantization**       | ONNX PTQ Dynamic Quantization    | Fast and reduces model size  |
| **Export Format**      | ONNX                       | Standard format for interoperability. Require post processing NMS(Non Maximum Supression) |

---

##  Quantitative Results of Optimizations

### A. Model Size Reduction

| Model     | Precision Format | File Size (MB) | Reduction (%) |
|-----------|------------------|----------------|---------------|
| best.pt /.onnx  | FP32 (32bit)    | 42MB | /|
| best_int8_dynamic.onnx) | INT8 (8bit)     | 11MB | 74% |

### B. Performance Gain (CPU Inference Speed)

| Model     | Precision | Avg Latency (ms) | Acceleration |
|-----------|-----------|------------------|--------------|
|  best.pt /.onnx   | FP32      | 2.9ms | /|
| best_int8_dynamic.onnx) | INT8      | /| /|

---

##  Detailed Instructions for Reproducing Results

### Step A: Environment Setup
- **Clone Repository**: Clone this repo locally or in the cloud.  
- **Install Dependencies**: Install libraries from `requirements.txt` (Ultralytics, Roboflow, ONNX Runtime) 
- **Data Acquisition**: Run the Roboflow API script to fetch the PCB Defects dataset and generate `data.yaml`.

### Step B: Training & Export
- **EDA**: Visualize class distribution and annotation quality.  
- **Training**: Train YOLOv8s for 50 epochs (adjustable). The best model is saved as `best.pt`.  
- **FP32 Export**: Convert `best.pt` → `model_fp32.onnx`.

### Step C: Quantization
- **INT8 Quantization**: Apply ONNX Runtime dynamic quantization → `best_int8_dynamic.onnx`.  
- **Benchmarking**: Run the benchmarking script to measure latency and FPS for FP32 vs INT8.

---

## Critical Analysis of Limitations and Trade-offs

### Adopted Trade-offs
- **Accuracy vs Speed**: INT8 quantization prioritizes speed, with a small decrease in precision.  
- **Quantization Method**: Dynamic quantization chosen for simplicity and due to time limitations, the ONNX quantized model required post processing phase as it does not include the NMS inside the architecture as YOLOv8 or the best.pt automaticallly does. Static quantization may be less problem causing and error but requires much more time, documentation, coding and calibration data.

### Current Limitations
- **No Post Quantization Accuracy Check**: Accuracy of INT8 model vs FP32 not formally validated since the ONNX quantized version requires post processign Non Mximum Suppression which is responsible for box and   
- **Non Valid Benchmarking**: Performance results depend on host CPU ; and hardware ressources.The Full Precision model with 42MB size is ver drectly deployable on hardware wit 4gb ram, but CPU usage may lower the inference time, GPU is highly recommended fgor Image detection and processing.
