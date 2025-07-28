from dataclasses import dataclass
from typing import Literal
from blacksheep import post, status_code, FormPart, Application, Request
from blacksheep.server.headers.cache import cache_control
import io
import PIL.Image
from PIL.Image import Image
from yolo import YOLOModelFormat, YOLOModel, YOLOAnalysisManager, AnalysisResult, test_model
from ultralytics.engine.results import Results
from datetime import datetime
import os
import shutil

@dataclass
class YOLOAnalysisRequest:
    model: YOLOModel | Literal["undefined"]
    format: YOLOModelFormat | Literal["undefined"]
    image: Image
    
    def __init__(self, model: YOLOModel, format: YOLOModelFormat, image: list[FormPart]):
        self.model = model if str(model) in [i.lower() for i in YOLOModel._member_names_] else "undefined"
        self.format = format if str(format) in [i.lower() for i in YOLOModelFormat._member_names_] else "undefined"
        self.image = PIL.Image.open(io.BytesIO(image[0].data))

app = Application()

@post("/api/analyse")
@cache_control(no_cache=True, no_store=True)
async def analyse_image(request: Request):
    start = datetime.now()

    content_type_header = request.headers.get_first(b"Content-Type")
    
    if content_type_header == None:
        return status_code(405, {"success": False, "error": "This endpoint accepts only multipart/form-data"})

    if content_type_header.decode().split(";")[0] != "multipart/form-data":
        return status_code(405, {"success": False, "error": "This endpoint accepts only multipart/form-data, not " + content_type_header.decode().split(";")[0]})
    
    untyped_form = await request.form()
    
    if untyped_form == None:
        return status_code(405, {"success": False, "error": "This endpoint accepts only multipart/form-data"})

    form: dict = untyped_form
    
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
    
    analysis_result: AnalysisResult = YOLOAnalysisManager.analyse_image(analysis_request.model, analysis_request.format, analysis_request.image)

    if not analysis_result.success or analysis_result.result == None:
        return status_code(500, {"success": False, "error": analysis_result.reason})
    
    result: Results = analysis_result.result

    end = datetime.now()

    summary = result.summary()
    for entry in summary:
        c = entry.pop("class")
        entry['class_id'] = c

    response_data = {"success": True,  "metadata": {"speed": result.speed, "names": result.names, "request":{"time_ms": (end-start).microseconds/1000}}, "result":summary}
    return response_data

@post("/api/custom-model")
async def upload_custom_model(request: Request):
    start = datetime.now()
    
    content_type_header = request.headers.get_first(b"Content-Type")        
    
    if content_type_header == None:
        return status_code(405, {"success": False, "error": "This endpoint accepts only multipart/form-data"})

    if content_type_header.decode().split(";")[0] != "multipart/form-data":
        return status_code(405, {"success": False, "error": "This endpoint accepts only multipart/form-data, not " + content_type_header.decode().split(";")[0]})
    
    untyped_form = await request.form()
    
    if untyped_form == None:
        return status_code(405, {"success": False, "error": "This endpoint accepts only multipart/form-data, not " + content_type_header.decode().split(";")[0]})
    
    form: dict = untyped_form
    
    if "model" not in form:
        return status_code(400, {"success": False, "error": "You must provide a custom model."})    

    model = form.get("model")
    
    if model == None or type(model) != list:
        return status_code(400, {"success": False, "error": "You must provide a valid custom model file in .pt format."})    

    model_file: FormPart = model[0]

    if model_file == None:
        return status_code(400, {"success": False, "error": "You must provide a valid custom model file in .pt format."})    

    if model_file.file_name == None:
        return status_code(400, {"success": False, "error": "You must provide a valid custom model file in .pt format."})    

    if not model_file.file_name.decode().endswith("pt"):
        return status_code(400, {"success": False, "error": "You must provide a valid custom model file in .pt format."})    
    
    file_bytes = model_file.data
    
    with open("custom_temp.pt", "xb") as f:
        f.write(file_bytes)
    
    
    valid = test_model("custom_temp.pt")
    
    if not valid:
        os.remove("custom_temp.pt")
        return status_code(422, {"success": False, "error": "Provided custom model is not a valid YOLO model, please provide a valid YOLO model."})

    if os.path.exists("custom.pt") and os.path.isfile("custom.pt"):
        os.remove("custom.pt")
    if os.path.exists("custom_ncnn_model"):
        shutil.rmtree("custom_ncnn_model")
    if os.path.exists("custom.onnx") :
        os.remove("custom.onnx")
    os.rename("custom_temp.pt", "custom.pt")

    end = datetime.now()

    return status_code(201, {"success": True, "result": "Accepted custom model " + model_file.file_name.decode(), "metadata":{"request":{"time_ms": (end-start).microseconds/1000}}})