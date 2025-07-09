# Simple convenience script to load the 
# model supported by the yolo.py module.
#
# This is useful if the device hosting
# the processing server may not be connected
# to the internet at all times.

from ultralytics import YOLO
from yolo import YOLOModel

for model in [i.lower() for i in YOLOModel._member_names_]:
    print("Trying to download " + model)
    try:
        model = YOLO(model, task='detect')
        model.export(format="ncnn")
        model.export(format="onnx")
    except:
        print(f"Error for {model}")
