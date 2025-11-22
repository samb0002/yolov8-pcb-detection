import cv2
import matplotlib.pyplot as plt
import os
import yaml

# Load class names from data.yaml configuration file
yaml_path = "/content/PCB-Defects--1/data.yaml"
with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)
class_names = data["names"]   # list of class names

# Image and label paths
img_dir = "/content/PCB-Defects--1/train/images"
label_dir = "/content/PCB-Defects--1/train/labels"

# Pick one image to visualize (change the index to see another sample)
img_file = os.listdir(img_dir)[100]
label_file = img_file.rsplit('.', 1)[0] + ".txt"

# Load image
img = cv2.imread(os.path.join(img_dir, img_file))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape

# Load labels
with open(os.path.join(label_dir, label_file), "r") as f:
    anns = f.read().strip().splitlines()

# Draw bounding boxes and class names
for ann in anns:
    cls, x, y, bw, bh = map(float, ann.split())
    cls = int(cls)

    x1 = int((x - bw / 2) * w)
    y1 = int((y - bh / 2) * h)
    x2 = int((x + bw / 2) * w)
    y2 = int((y + bh / 2) * h)

    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (200, 255, 0), 2)

    label = class_names[cls]
    cv2.putText(
        img_rgb,
        label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 200, 0),
        2
    )

# Display the image with bounding boxes
plt.figure(figsize=(5, 8))
plt.imshow(img_rgb)
plt.axis("off")
plt.show()



"""
verify_dataset_balance.py

This script verifies that the dataset classes are balanced to avoid bias and overfitting.
It loads the class names from the YOLO data.yaml file, counts the number of annotations
per class in the training labels, and visualizes the distribution with a bar chart.
"""

import os
from collections import Counter
import yaml
import matplotlib.pyplot as plt

yaml_path = "/content/PCB-Defects--1/data.yaml"

with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

class_names = data["names"]
print("Class names:", class_names)

# Loop through all label files in the dataset from roboflow to check if classes are balanced or unbalanced as this will results in false results and OVERFITTING !!

import os
from collections import Counter
import yaml
import matplotlib.pyplot as plt

yaml_path = "/content/PCB-Defects--1/data.yaml"

with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

class_names = data["names"]
print("Class names:", class_names)

# Loop through all label files to count how many images contain each class

label_dir = "/content/PCB-Defects--1/train/labels"
image_class_counts = Counter()

print("\nCounting images containing each class in the training set...")
for file in os.listdir(label_dir):
    if file.endswith(".txt"):
        with open(os.path.join(label_dir, file), "r") as f:
            lines = f.read().strip().splitlines()

            
            unique_cls_ids_in_image = set()
            for line in lines:
                cls_id = int(line.split()[0])
                unique_cls_ids_in_image.add(cls_id)
            
            # Increment count for each class found in this image
            for cls_id in unique_cls_ids_in_image:
                class_name = class_names[cls_id]
                image_class_counts[class_name] += 1

print("\nNumber of images containing each class (training set):\n")
for name, count in sorted(image_class_counts.items()):
    print(f"{name}: {count} images")

# Plotting the bar graph to visualize the Number of images / Class
plt.figure(figsize=(10, 5))
plt.bar(image_class_counts.keys(), image_class_counts.values(), color="teal")
plt.xlabel("Class Name")
plt.ylabel("Number of Images")
plt.title("Class Distribution by Image in Training Dataset")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
