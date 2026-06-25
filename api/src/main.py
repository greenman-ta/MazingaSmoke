from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from enum import Enum
import tempfile
import os
from exceptions import GeneralInputError, MediaError
from fastapi.responses import JSONResponse

from .preprocess import pipeline_preprocess
from .inference import inference


app = FastAPI(
    title="Mazinga Smoke Classifier API",
)

class ModelVersion(str, Enum):
    v1 = "v1"
    v2 = "v2"

ALLOWED_EXTS = {".mp4", ".npy"}

ALLOWED_CONTENT_TYPES = {

    ".mp4": {"video/mp4", "application/mp4"},

    ".npy": {"application/octet-stream", "application/x-npy", "application/vnd.numpy"},

}

@app.exception_handler(GeneralInputError)
async def exception_general_handler(request: Request, exc: GeneralInputError):
    return JSONResponse(
        status_code= 400,
        content={"detail": exc.detail},
    )

@app.exception_handler(MediaError)
async def exception_media_handler(request: Request, exc: MediaError):
    return JSONResponse(
        status_code= 415,
        content={"detail": exc.detail},
    )



@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict", response_model = dict)
async def predict(
    file: UploadFile = File(..., description=".mp4 o .npy da caricare"),
    version: ModelVersion = Form(..., description="Versione del modello: v1 o v2"),
):

    tmp_path = None

    suffix = os.path.splitext(file.filename)[1].lower() if file.filename else None

    #controlli sui file in ingresso
    if suffix not in ALLOWED_EXTS:
        raise HTTPException(status_code=415, detail="File non supportato. Caricare un file .mp4 o .npy.")

    allowed_types = ALLOWED_CONTENT_TYPES.get(suffix, set())
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail="File non supportato. Caricare un file .mp4 o .npy.")

    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)
        
        preprocessed_data = pipeline_preprocess(tmp_path)
        result = inference(preprocessed_data, version)
        return result
    
    finally:
        await file.close()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
