from dataclasses import dataclass
from blacksheep import post, status_code, FromForm, FormPart, pretty_json, Application, Request
from blacksheep.server.headers.cache import cache_control
import io
from PIL.Image import Image, open
from yolo import YOLOModelFormat, YOLOModel, YOLOAnalysisManager
from ultralytics.engine.results import Results

@dataclass
class YOLOAnalysisRequest:
    model: YOLOModel
    format: YOLOModelFormat 
    image: Image
    
    def __init__(self, model: YOLOModel, format: YOLOModelFormat, image: list[FormPart]):
        match model:
            case YOLOModel.YOLO11N:
                self.model = YOLOModel.YOLO11N
            case YOLOModel.YOLO11M:
                self.model = YOLOModel.YOLO11M
            case YOLOModel.YOLO11S:
                self.model = YOLOModel.YOLO11S
            case YOLOModel.YOLO11L:
                self.model = YOLOModel.YOLO11L
            case _:
                self.model = 'undefined'   
        match format:
            case YOLOModelFormat.ONNX:
                self.format = YOLOModelFormat.ONNX
            case YOLOModelFormat.PYTORCH:
                self.format = YOLOModelFormat.PYTORCH
            case YOLOModelFormat.NCNN:
                self.format = YOLOModelFormat.NCNN
            case _:
                self.format = 'undefined'
        self.image = open(io.BytesIO(image[0].data))

app = Application()

@post("/api/analyse")
@cache_control(no_cache=True, no_store=True)
async def analyse_image(request: Request):
    form: dict[str, any] = await request.form()
    
    format = form.get("format", YOLOModelFormat.NCNN)
    model = form.get("model", YOLOModel.YOLO11N)
    
    if "image" not in form:
        return status_code(400, {"success": False, "error": "You must provide an image"})    
    
    request: YOLOAnalysisRequest = YOLOAnalysisRequest(model, format, form["image"])
    
    if request.model == "undefined":
        return status_code(400, {"success": False, "error": "The provided YOLO model is not supported. Please use one of " + ", ".join([f"'{i.lower()}'" for i in list(YOLOModel.__members__.keys())])})
        
    if request.format == "undefined":
        return status_code(400, {"success": False, "error": "The provided format is not supported. Please use one of " + ", ".join([f"'{i.lower()}'" for i in list(YOLOModelFormat.__members__.keys())])})
    result: Results = YOLOAnalysisManager.do_yolo_analysis(request.model, request.format, request.image)
    response_data = {"success": True,  "metadata": {"speed": result.speed, "names": result.names}, "result":result.summary()}
    
    return response_data