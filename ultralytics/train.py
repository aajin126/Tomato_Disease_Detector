from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model.train(data = 'D:\dataset_root\data.yaml', epochs=5, batch= 5)  # train the model
