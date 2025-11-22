# Data exploration
- We begin by exploring the data its size / number of total images, teh total number of images is calculated to be 13554 images.
- Then we chck if the data is balanced , each of the six classes, has an approximate similar amount of images to avoid overfitting, this can be shown in the following graph : 

-missing_hole: 1950 images <br>
-mouse_bite: 1914 images <br>
-open_circuit: 1820 images <br>
-short: 1792 images <br>
-spur: 1808 images <br>
-spurious_copper: 1882 images
<img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/fd3db505-aa3d-44b1-bd34-b11f9d572513" />

-The results of the Yolov8 model can be seen in the images in this file.<br>

## Evaluation Metrics

| Class            | Images | Instances | Precision (P) | Recall (R) | mAP@0.50 | mAP@0.50-0.95 |      F1 score |
|------------------|--------|-----------|---------------|------------|----------|---------------|---------------|
| **all**          | 1592   | 3266      | 0.978         | 0.990      | 0.990    | 0.646         |  0.98 overall for all classes|
| missing_hole     | 268    | 554       | 0.984         | 0.998      | 0.992    | 0.694         |               |
| mouse_bite       | 297    | 589       | 0.972         | 0.995      | 0.992    | 0.650         |               |
| open_circuit     | 255    | 532       | 0.983         | 0.991      | 0.992    | 0.629         |               |
| short            | 240    | 477       | 0.965         | 0.974      | 0.981    | 0.629         |               |
| spur             | 282    | 592       | 0.985         | 0.987      | 0.992    | 0.628         |               | 
| spurious_copper  | 250    | 522       | 0.982         | 0.992      | 0.991    | 0.644         |               |

- Preprocess: **1.2 ms** per image <br>
- Inference: **9.4 ms** per image  <br>
- Loss: **0.0 ms** per image  <br>
- Postprocess: **1.0 ms** per image

## Quantization:

We applied **static quantization** in order to avoid the `Conv2DInteger` error produced by ONNX Runtime when using dynamic PTQ. This required using a portion of our dataset as **calibration data**, allowing the quantized model to comprehend and take into consideration the ranges of confidence scores and thresholds during inference.

However, to properly measure inference time and evaluate predictions on real data, a **post-processing step** is necessary. In our case, we rely on **Non-Maximum Suppression (NMS)**, which is not a part of the onnx quantized vertsion architecture. NMS ensures that overlapping bounding boxes are filtered adn boxed, keeping only the most confident detections. This step is critical for the quantized ONNX model to produce accurate and interpretable results.

It is important to note that **measuring real inference time could not be fully achieved** in this study, as it requires deeper research, more time, and practical experience with deployment pipelines.


  
