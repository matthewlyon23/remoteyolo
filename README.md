# remoteyolo
A YOLO processing server written in Python with the [Blacksheep](https://github.com/Neoteroi/BlackSheep) library.

The server is designed with real-time applications in mind, therefore the implementation is as minimal as possible to reduce processing latency.

This project was created as part of the [YOLOQuestUnity](https://github.com/matthewlyon23/yoloquestunity) project.

**Contents**

- [`Endpoints`](#endpoints)
- [`Models`](#models)
- [`Formats`](#formats)
- [`Installation & Usage`](#installation-and-usage)

## Endpoints

### /api/

- [`POST - /api/analyse`](#post---apianalyse)
- [`POST - /api/custom-model`](#post---apicustom-model)

### POST - /api/analyse

**Accepts: multipart/form-data**

Parameters:
| parameter | content-type | required | default |
|---|---|---|---|
| **image** | `image/*` | ✅ | `none` |
| **model** | `text/plain` | ❌ | `yolo11n` |
| **format** | `text/plain` | ❌ | `ncnn` |

Example Request:

**Note that the @ preceding the image path is required in curl**

```bash
curl -L \
  -X POST \
  -H "Accept: application/json" \
  -F image=@path/to/image \
  -F model=yolo11n \
  -F format=ncnn \
  http://host:port/api/analyse
```

Example Responses:

**200**

```json
{
  "success": true,
  "metadata": {
    "speed": {
      "preprocess": 1.7089169996324927,
      "inference": 13.822042004903778,
      "postprocess": 2.263624999613967
    },
    "names": {
      "0": "person",
      "1": "bicycle",
      "2": "car",
      "3": "motorcycle",
      "4": "airplane",
      "5": "bus",
      "6": "train",
      "7": "truck",
      "8": "boat"
    },
    "request": {
      "time_ms": 70.46
    }
  },
  "result": [
    {
      "name": "person",
      "class_id": 0,
      "confidence": 0.57129,
      "box": {
        "x1": 1090.13672,
        "y1": 292.96875,
        "x2": 1410.64453,
        "y2": 876.5625
      }
    }
  ]
}

```

**400**

```json
{
  "success": false,
  "error": "The provided format is not supported. Please use one of 'ncnn', 'onnx', 'pytorch'"
}
```

### POST - /api/custom-model

**Accepts: multipart/form-data**

Parameters:
| parameter | content-type | required | default |
|---|---|---|---|
| **model** | `application/octet-stream` | ✅ | `none` |

The `model` parameter is a YOLO model stored as a pytorch `.pt` model file. The filename must end with the `.pt` extension.

Example Request:

**Note that the @ preceding the model path is required in curl**

```bash
curl -L \
  -X POST \
  -H "Accept: application/json" \
  -F model=@path/to/model \
  http://host:port/api/custom-model
```

Example Responses:

**201**

```json
{
  "success": true,
  "result": "Accepted custom model custom.pt",
  "metadata": {
    "request": {
      "time_ms": 18.955
    }
  }
}
```

**400**

```json
{
  "success": false,
  "error": "You must provide a valid custom model file in .pt format."
}
```

**422**

```json
{
  "success": false,
  "error": "Provided custom model is not a valid YOLO model, please provide a valid YOLO model."
}
```

## Models

The current implementation supports all YOLO11 model variants.

| model | supported |
| ----- | --------- |
| yolo11n | ✅ |
| yolo11s | ✅ |
| yolo11m | ✅ |
| yolo11l | ✅ |
| custom  | ✅ |

To use the 'custom' model type, a corresponding custom.pt, custom.onnx or custom_ncnn_model must exist in the root directory of the application. This can either be added manually or can be uploaded in .pt format using the [custom-model](#post---apicustom-model) endpoint.

## Formats

Supported Formats:

| format | supported |
| ------ | --------- |
| ncnn   | ✅        |
| onnx   | ✅        |
| pytorch| ✅        |

## Installation and Usage

### Requirements

- \>= Python 3.12
- pip

### Installation

Linux/Unix:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_models.py
```

Windows:

```sh
python -m venv .venv
.venv/bin/activate.ps1
pip install -r requirements.txt
python download_models.py
```

### Usage

The server can be run directly in development mode simply by running:

```sh
python dev.py
```

The repository also provides a Dockerfile which includes all requirements and deploys the server in production mode. The easiest way to use this is with the provided Docker Compose file.

```sh
docker compose up -d
```

#### Model Download Script

A script, [`download_models.py`](/download_models.py), has been provided which downloads all default models supported by the server. This is essential when the device hosting the server may not be connected to the internet, so all models must be downloaded prior to attempting to use them. 

If the device is connected to the internet, the models will be downloaded automatically, thought this will result in a longer response time.

## Copyright

Matthew Lyon © 2025

