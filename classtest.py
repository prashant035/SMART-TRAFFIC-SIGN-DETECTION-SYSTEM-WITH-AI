from ultralytics import YOLO


model = YOLO('runs/detect/train4/weights/best.pt' )  


print(model.names)
