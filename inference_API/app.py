from fastapi import FastAPI, UploadFile, File
from PIL import Image
from ultralytics import YOLO
import io

# API to expose the Yolo model

app = FastAPI()

# Loading model
model = YOLO("best.pt")
print("Model classes :", model.names)

# Health check
@app.get("/")
def home():
    return {"status": "ok", "classes": model.names}

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")

    results = model.predict(image)[0]

    detections = []
    # For each bounding box, 3 pieces of information are extracted
    for box in results.boxes:
        detections.append({
            # converte 0 and 1 into fire or smoke
            "classe": model.names[int(box.cls)],
            # confidence score
            "confidence": float(box.conf),
            # bbox coordinates
            "bbox": box.xyxy[0].tolist(),
        })

    return {"detections": detections}
