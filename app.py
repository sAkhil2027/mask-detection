from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

model = YOLO("best.pt")

# Class mapping
CLASS_NAMES = {
    0: "masked",
    1: "unmasked"
}

CONF_THRESHOLD = 0.5

@app.get("/")
def root():
    return {"message": "Mask Detection API running"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):

    contents = await image.read()
    img = Image.open(io.BytesIO(contents))

    results = model(img)[0]

    detections = []

    for box in results.boxes:

        conf = float(box.conf)

        # filter weak predictions
        if conf < CONF_THRESHOLD:
            continue

        cls_id = int(box.cls)
        label = CLASS_NAMES.get(cls_id, "unknown")

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "label": label,
            "confidence": round(conf, 3),
            "bbox": [
                round(x1, 1),
                round(y1, 1),
                round(x2, 1),
                round(y2, 1)
            ]
        })

    return {
        "count": len(detections),
        "detections": detections
    }
