from fastapi import FastAPI, UploadFile, File
import torch
import torchvision.models as models
from .model import PatchCore
from pathlib import Path
from torchvision.io import decode_image

BASE_PATH = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = BASE_PATH/'model'/'v2'/'patchcore.pt'

app = FastAPI()
patchcore = PatchCore(checkpoint_path=CHECKPOINT_PATH)


@app.get("/")
def root():
    return {'message': 'PatchCore API is running'}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    content = file.file.read()
    image = decode_image(torch.frombuffer(content, dtype=torch.uint8))
    return patchcore.predict(image)