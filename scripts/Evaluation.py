
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model using the best.pt file
model = YOLO("/content/runs/detect/train2/weights/best.pt")

# Validate and evalute the Yolo trained model on  on the test dataset  YAML
results = model.val(data="/content/PCB-Defects--1/data.yaml", plots=False) # True if plots are needed to be plotted.

