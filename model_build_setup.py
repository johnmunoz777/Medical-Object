from ultralytics import YOLO
#model_two=YOLO('yolov8l.pt')
#source="htpps://ultraalytics.com/images/bus.jpg"
#model_two.predict(source,save=True,imgsz=640,conf=0.5)

import os
HOME = os.getcwd()
print(HOME)
DATA_YAML_PATH = f"{HOME}/data.yaml"
model = YOLO('yolov8l.pt')  # load a pretrained model (recommended for training)
results = model.train(data=DATA_YAML_PATH,batch=8, epochs=100)
