from flask import Flask, request, Response
from yolo import YOLOAnalysisManager, YOLOModel, YOLOModelFormat
from PIL.Image import Image, open
import io

app = Flask(__name__)

@app.route("/")
def hello():
    return "<p>Hello, World!<p>"

class YOLOAnalysisRequest:    
    model: YOLOModel
    format: YOLOModelFormat
    image: Image
    
    def __init__(self, model: YOLOModel, format: YOLOModelFormat, image: bytes) -> None:
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
        
        self.image = open(io.BytesIO(image))
        

@app.route("/api/analyse", methods=["POST"])
def analyse_image():
    model = YOLOModel(request.form.get("model", "yolo11n"))
    format = YOLOModelFormat(request.form.get("format", "ncnn"))
    if "image" in request.files:
        image_bytes = io.BytesIO()
        request.files["image"].save(image_bytes)
    else:
        return Response({"success": False, "result": "An image must be provided"}, status=400)
    yolo_request = YOLOAnalysisRequest(YOLOModel(request.form["model"]), YOLOModelFormat(request.form["format"]), image_bytes.getvalue())
    if yolo_request.format=='undefined':
        return Response({"success": False, "result": f"{request.form.get("format")} is not a valid format. Valid formats are {",".join(YOLOModelFormat._member_names_)}."}, status=400)
    if yolo_request.model=='undefined':
        return Response({"success": False, "result": f"{request.form.get("model")} is not a valid model. Valid models are {",".join(YOLOModel._member_names_)}."}, status=400)
    
    
    results = YOLOAnalysisManager.do_yolo_analysis(yolo_request.model, yolo_request.format, yolo_request.image)
    return {"success": True, "metadata": {"speed": results.speed, "names": results.names}, "result": results.summary()}



if (__name__=="__main__"):
    app.run(debug=True, host="0.0.0.0", port=44777)