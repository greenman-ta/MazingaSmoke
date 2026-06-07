from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Literal
import tempfile
import os

from .preprocess import pipeline_preprocess
from .inference import inference

app = FastAPI(
    title="Mazinga Smoke Classifier API",
)

class PredictionRequest(BaseModel):
    file: UploadFile = Field(..., description="File to be uploaded")
    version: Literal["v1", "v2"] = Field(..., examples =["v1 o v2"])
    #threshold: float = 0.5


ALLOWED_EXTS = {".mp4", ".npy"}

ALLOWED_CONTENT_TYPES = {

    ".mp4": {"video/mp4", "application/mp4"},

    ".npy": {"application/octet-stream", "application/x-npy", "application/vnd.numpy"},

}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict", response_model = dict)
async def predict(
    file: UploadFile = File(..., description=".mp4 o .npy da caricare"),
    version: Literal["v1", "v2"] = Form(..., description="Versione del modello: v1 o v2"),
):

    tmp_path = None

    suffix = os.path.splitext(file.filename)[1].lower() if file.filename else None

    #controlli sui file in ingresso
    if suffix not in ALLOWED_EXTS:
        return {"error": "File non supportato. Caricare un file .mp4 o .npy."}

    allowed_types = ALLOWED_CONTENT_TYPES.get(suffix, set())
    if file.content_type not in allowed_types:
        return {"error": f"Content-Type non valido per {suffix}. Ricevuto: {file.content_type}"}

    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)
        
        preprocessed_data = pipeline_preprocess(tmp_path)
        result = inference(preprocessed_data, version)
        return result
    except Exception as e:
        return {"error": str(e)}

    finally:

        await file.close()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
