from ultralytics import YOLO

for model in ['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l']:
    print("Trying to download " + model)
    try:
        model = YOLO(model)
        model.export(format="ncnn")
        model.export(format="onnx")
    except:
        print(f"Error for {model}")
