# Initialize YOLOv8 model
model = YOLO("yolov8s.pt")  # You can change to yolov8n.pt , yolov8.pt ...
model.info()

model.train(
    data=data_yaml,
    epochs=50,           # can adjust to 100 for better results but takes a loooooot of time
    batch=16,
    imgsz=640,
    lr0=0.001,           # baseline LR
    optimizer="AdamW",
    device="cuda",
    amp=True, # a 'evolve=True' paramerter can be added, this parameter will allow to the YOLO model to automtically tune the parameters and find the optimal ones, thus taking much more time than usual.
    workers=4,
    augment=True         # default YOLOv8 augmentation
)

