from enum import StrEnum
from ultralytics import YOLO
from PIL.Image import Image
from ultralytics.engine.results import Results

class YOLOModelFormat(StrEnum):
    NCNN = 'ncnn'
    ONNX = 'onnx'
    PYTORCH = 'pytorch'

class YOLOModel(StrEnum):
    YOLO11N = 'yolo11n'
    YOLO11S = 'yolo11s'
    YOLO11M = 'yolo11m'
    YOLO11L = 'yolo11l'

class YOLOAnalysisManager:

    yolo_model: YOLO = YOLO(YOLOModel.YOLO11N)
    yolo_model_format: YOLOModelFormat = YOLOModelFormat.ONNX
    yolo_model_variant: YOLOModel = YOLOModel.YOLO11N

    @staticmethod
    def __load_yolo_variant__(yolo_model: YOLOModel, yolo_model_format: YOLOModelFormat):
        try:
            match yolo_model_format:
                case YOLOModelFormat.ONNX:
                    YOLOAnalysisManager.yolo_model = YOLO(f"{yolo_model}.onnx", task="detect")
                case YOLOModelFormat.NCNN:
                    YOLOAnalysisManager.yolo_model = YOLO(f"{yolo_model}_ncnn_model", task="detect")
                case _:
                    YOLOAnalysisManager.yolo_model = YOLO(yolo_model, task="detect")
            YOLOAnalysisManager.yolo_model_format = yolo_model_format
            YOLOAnalysisManager.yolo_model_variant = yolo_model
        except:
            model = YOLO(yolo_model)
            model.export(format=yolo_model_format)
            YOLOAnalysisManager.__load_yolo_variant__(yolo_model, yolo_model_format)
    
    @staticmethod
    def analyse_image(yolo_model: YOLOModel, format: YOLOModelFormat, image: Image) -> Results:
        if yolo_model!=YOLOAnalysisManager.yolo_model_variant or format!=YOLOAnalysisManager.yolo_model_format:
            YOLOAnalysisManager.__load_yolo_variant__(yolo_model, format)
        results = YOLOAnalysisManager.yolo_model.predict(source=image, task="detect", verbose=False, save=True)
        return results[0]
        
    
    


        