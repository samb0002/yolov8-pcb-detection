
import onnxruntime as ort
import numpy as np
from pathlib import Path
from onnxruntime.quantization import (
    quantize_static, QuantFormat, QuantType, CalibrationDataReader
)

FP32_MODEL_PATH = "/content/best.onnx"
INT8_MODEL_PATH = "/content/best_int8_qdq.onnx"
TEST_IMAGE_DIR = "/content/PCB-Defects--1/test/images"

IMG_SIZE = 640
NUM_BENCHMARK_IMAGES = 50
NUM_WARMUP_RUNS = 10

# QDQ INT8 QUANTIZATION , as the onnxruntime has a limitation in inference time for direct  dynamic quantization
class PCBCalibrationDataReader(CalibrationDataReader):
    """
    Calibration reader using real images from a folder for static QDQ quantization.
    """
    def __init__(self, calibration_image_dir: str, img_size: int = 640, max_images: int = 100):
        self.img_size = img_size
        self.image_paths = sorted(list(Path(calibration_image_dir).glob("*.jpg")))[:max_images]
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {calibration_image_dir}")
        
        # Preprocess images once
        self.data = [self.preprocess_image(p) for p in self.image_paths]
        self.iterator = iter(self.data)
        print(f"✅ Loaded {len(self.data)} images for calibration.")

    def preprocess_image(self, image_path: str) -> dict:
        """
        Preprocess a single image for ONNX quantization calibration.
        Returns a dictionary with input name as key.
        """
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2,0,1).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return {"images": img}  # use input name 'images' as in your model

    def get_next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            return None

    def rewind(self):
        self.iterator = iter(self.data)

calibration_dir = "/content/PCB-Defects--1/valid/images"  # folder with 100 real images

from onnxruntime.quantization import quantize_static, QuantFormat, QuantType

reader = PCBCalibrationDataReader(calibration_dir, img_size=640, max_images=100)

quantize_static(
    model_input="/content/best.onnx",
    model_output="/content/best_int8_qdq.onnx",
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QUInt8
)

# Check quantized model size

quant_size = os.path.getsize('/content/best_int8_qdq.onnx') / (1024 * 1024)  # MB
print(f"Quantized INT8 model size: {quant_size:.2f} MB")

#  Compute size reduction

reduction = (orig_size - quant_size) / orig_size * 100
print(f"Size reduction: {reduction:.2f}%")
