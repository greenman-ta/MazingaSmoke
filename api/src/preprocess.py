import cv2 as cv
import numpy as np
import tempfile, os
from exceptions import MediaError, GeneralInputError

TARGET_FRAMES = 36
TARGET_SIZE = 224

def mp4_to_npy(mp4_path: str) -> np.ndarray:

    cap = cv.VideoCapture(mp4_path) 
    ok, frame = cap.read()

    frames = []
    while ok:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frames.append(frame)
        ok, frame = cap.read()

    cap.release()
    return np.array(frames)

def preprocess_npy(frames: np.ndarray) -> np.ndarray:

    n = frames.shape[0]

    if n < TARGET_FRAMES:
        raise GeneralInputError(detail=f"Not enough frames: {n} < {TARGET_FRAMES}")

    idx = np.linspace(0, n-1, TARGET_FRAMES).round().astype(int)
    sampled = frames[idx]

    return sampled

def resize_video_to_temp(input_path: str, size=(TARGET_SIZE, TARGET_SIZE)) -> str:

    cap = cv.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise GeneralInputError(detail=f"Could not open video: {input_path}")

    
    fps = cap.get(cv.CAP_PROP_FPS) or 12.0
    width, height = size

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(tmp.name, fourcc, fps, (width, height))

    if not out.isOpened():
        cap.release()
        os.remove(tmp.name)
        raise ValueError(f"Could not create video writer for: {tmp.name}")
    
    
    ok, frame = cap.read()
    while ok:
        frame = cv.resize(frame, size, interpolation=cv.INTER_AREA)
        out.write(frame)
        ok, frame = cap.read()

    cap.release()
    out.release()

    return tmp.name

def detect_file_type(data: bytes) -> str:
    if data[:6] == b"\x93NUMPY":          # firma del formato .npy
        return "npy"
    if len(data) >= 8 and data[4:8] == b"ftyp":   # box "ftyp" degli mp4
        return "mp4"
    raise MediaError(detail="Il file deve essere .npy o .mp4")


def pipeline_preprocess(input_path: str) -> np.ndarray:
    
    with open(input_path, "rb") as f:
        head = f.read(12)

    file_type = detect_file_type(head)

    match  file_type:
        case "mp4":
            temp_file = resize_video_to_temp(input_path)
            try:
                npy_raw = mp4_to_npy(temp_file)
            finally:
                os.remove(temp_file)

        case "npy":
            npy_raw = np.load(input_path, allow_pickle=False)
            if npy_raw.ndim != 4 or npy_raw.shape[3] != 3:
                raise GeneralInputError(detail=f"Formato .npy non valido: shape {npy_raw.shape} non compatibile con {(TARGET_FRAMES, TARGET_SIZE, TARGET_SIZE, 3)}")

        case _:
            raise MediaError(detail="Il file deve essere .npy o .mp4")

    return preprocess_npy(npy_raw)