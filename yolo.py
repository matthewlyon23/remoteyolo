from enum import StrEnum
from ultralytics import YOLO
from PIL.Image import Image
from ultralytics.engine.results import Results
import os

class YOLOModelFormat(StrEnum):
    NCNN = 'ncnn'
    ONNX = 'onnx'
    PYTORCH = 'pytorch'

class YOLOModel(StrEnum):
    YOLO11N = 'yolo11n'
    YOLO11S = 'yolo11s'
    YOLO11M = 'yolo11m'
    YOLO11L = 'yolo11l'
    CUSTOM  = 'custom'
    
class AnalysisResult:    
    def __init__(self, success: bool, result: Results | None, reason: str | None):
        self.success = success
        self.result = result
        self.reason = reason

class YOLOAnalysisManager:

    yolo_model: YOLO = YOLO(YOLOModel.YOLO11N)
    yolo_model_format: YOLOModelFormat = YOLOModelFormat.ONNX
    yolo_model_variant: YOLOModel = YOLOModel.YOLO11N

    @staticmethod
    def __load_yolo_variant__(yolo_model: YOLOModel, yolo_model_format: YOLOModelFormat) -> bool:
        try:
            match yolo_model_format:
                case YOLOModelFormat.ONNX:
                    YOLOAnalysisManager.yolo_model = YOLO(f"{yolo_model}.onnx", task="detect")
                case YOLOModelFormat.NCNN:
                    YOLOAnalysisManager.yolo_model = YOLO(f"{yolo_model}_ncnn_model", task="detect")
                case _:
                    YOLOAnalysisManager.yolo_model = YOLO(f"{yolo_model}.pt", task="detect")
            YOLOAnalysisManager.yolo_model_format = yolo_model_format
            YOLOAnalysisManager.yolo_model_variant = yolo_model
            return True
        except:
            if yolo_model == YOLOModel.CUSTOM and not os.path.exists("custom.pt"):
                return False
            model = YOLO(f"{yolo_model}.pt")
            model.export(format=yolo_model_format)
            return YOLOAnalysisManager.__load_yolo_variant__(yolo_model, yolo_model_format)
    
    @staticmethod
    def analyse_image(yolo_model: YOLOModel, format: YOLOModelFormat, image: Image) -> AnalysisResult:
        if yolo_model!=YOLOAnalysisManager.yolo_model_variant or format!=YOLOAnalysisManager.yolo_model_format:
            success = YOLOAnalysisManager.__load_yolo_variant__(yolo_model, format)
            if not success:
                return AnalysisResult(success, None, "Could not load YOLO model")
        try:
            results = YOLOAnalysisManager.yolo_model.predict(source=image, task="detect", verbose=False, save=False)
        except:
            return AnalysisResult(False, None, "YOLO prediction failed")
        return AnalysisResult(True, results[0], None)
    
def test_model(model_file: str) -> bool:
    try:
        YOLO(model_file)
        return True
    except:
        return False
    
    


        