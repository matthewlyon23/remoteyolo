from dataclasses import dataclass
from blacksheep import post, status_code, FormPart, Application, Request
from blacksheep.server.headers.cache import cache_control
import io
from PIL.Image import Image, open
from yolo import YOLOModelFormat, YOLOModel, YOLOAnalysisManager
from ultralytics.engine.results import Results
from datetime import datetime

@dataclass
class YOLOAnalysisRequest:
    model: YOLOModel
    format: YOLOModelFormat 
    image: Image
    
    def __init__(self, model: YOLOModel, format: YOLOModelFormat, image: list[FormPart]):
        self.model = model if str(model) in [i.lower() for i in YOLOModel._member_names_] else "undefined" # type: ignore
        self.format = format if str(format) in [i.lower() for i in YOLOModelFormat._member_names_] else "undefined" # type: ignore
        self.image = open(io.BytesIO(image[0].data))

app = Application()

@post("/api/analyse")
@cache_control(no_cache=True, no_store=True)
async def analyse_image(request: Request):
    start = datetime.now()
    form: dict = await request.form() # type: ignore
    
    format = form.get("format", YOLOModelFormat.NCNN)
    model = form.get("model", YOLOModel.YOLO11N)
    
    if "image" not in form:
        return status_code(400, {"success": False, "error": "You must provide an image"})    
    
    if type(form["image"]) != list:
        return status_code(400, {"success": False, "error": "Provided image is not in the correct format"})
    
    analysis_request: YOLOAnalysisRequest = YOLOAnalysisRequest(model, format, form["image"])
    
    if analysis_request.model == "undefined":
        return status_code(400, {"success": False, "error": "The provided YOLO model is not supported. Please use one of " + ", ".join([f"'{i.lower()}'" for i in list(YOLOModel.__members__.keys())])})
        
    if analysis_request.format == "undefined":
        return status_code(400, {"success": False, "error": "The provided format is not supported. Please use one of " + ", ".join([f"'{i.lower()}'" for i in list(YOLOModelFormat.__members__.keys())])})
    
    result: Results = YOLOAnalysisManager.do_yolo_analysis(analysis_request.model, analysis_request.format, analysis_request.image)
    end = datetime.now()

    response_data = {"success": True,  "metadata": {"speed": result.speed, "names": result.names, "request":{"time_ms": (end-start).microseconds/1000}}, "result":result.summary()}
    return response_data