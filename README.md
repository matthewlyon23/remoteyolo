# remoteyolo
A YOLO11 processing server written in Python with the [Blacksheep](https://github.com/Neoteroi/BlackSheep) library.

## Endpoints

### /api/analyse

Accepts: multipart/form-data

Parameters:
| parameter | content-type | required | default |
|---|---|---|---|
| **image** | `image/*` | ✅ | `none` |
| **model** | `text/plain` | ❌ | `yolo11n` |
| **format** | `text/plain` | ❌ | `ncnn` |

Example Request:

```bash
curl -L \
  -X POST \
  -H "Accept: application/json" \
  -F image=@path/to/image \
  -F model=yolo11n \
  -F format=ncnn \
  http://host:port/api/analyse
```

Example Response:

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
      "class": 0,
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

## Models

The current implementation supports all YOLO11 model variants.

| model | supported |
| ----- | --------- |
| yolo11n | ✅ |
| yolo11s | ✅ |
| yolo11m | ✅ |
| yolo11l | ✅ |

## Formats

Supported Formats:

| format | supported |
| ------ | --------- |
| ncnn   | ✅        |
| onnx   | ✅        |
| pytorch  | ✅        |

## Copyright

Matthew Lyon © 2025

