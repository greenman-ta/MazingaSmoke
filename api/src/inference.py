import torch
import os
import sys
import numpy as np

CURRENT_FILE = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mazinga_smoke.mazinga_smoke import MazingaSmokeClassifier as v1
from mazinga_smoke.mazinga_smoke_v2 import MazingaSmokeClassifier as v2
from src.base_learner_3_f import BaseLearner

DEFAULT_THRESHOLD = 0.5

MODEL_WEIGHTS = {
    "v1": (v1, os.path.join(PROJECT_ROOT, "finetune_ilva_v1/ft_split3/best_f1.pt")),
    "v2": (v2, os.path.join(PROJECT_ROOT, "finetune_ilva_v2/ft_split5/best_model.pt")),
}

class TransformHelper(BaseLearner):
    def fit(self): pass
    def test(self): pass


helper = TransformHelper(use_cuda=False)

def _load_model(version: str):

    model_class, weight_path = MODEL_WEIGHTS[version]
    model = model_class(36)
    state = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(state)
    model.eval()
    return model

_MODELS = {version: _load_model(version) for version in MODEL_WEIGHTS.keys()}

def inference(file: np.ndarray, version: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    
    if version not in _MODELS:
        raise ValueError(f"Versione sconosciuta: {version!r}")
    
    model = _MODELS[version]

    #file = np.expand_dims(file, axis=0)
    val_transform = helper.get_transform(mode="rgb", phase="val", image_size=224)
    file = val_transform(file)

    file = file.unsqueeze(0).transpose(1, 2) 

    with torch.no_grad():
        logit = model(file)
        prob = torch.sigmoid(logit).item()
    smoke = prob >= threshold

    return {"smoke": bool(smoke), "confidence": prob, "threshold": threshold}