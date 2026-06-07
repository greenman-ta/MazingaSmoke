import pytest
import torch
import os
import sys
import numpy as np

CURRENT_FILE = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, ".."))
TEST_FILE_DIR = os.path.join(CURRENT_DIR, "test_files")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.src.inference import inference
from api.src.preprocess import preprocess_npy

from mazinga_smoke.mazinga_smoke import MazingaSmokeClassifier as v1
from mazinga_smoke.mazinga_smoke_v2 import MazingaSmokeClassifier as v2
from src.base_learner_3_f import BaseLearner

_MODELS = {
    "v1": v1(36).eval(),
    "v2": v2(36).eval()
}


#TEST MODELLI
def test_models_loaded():
    assert "v1" in _MODELS, "Model v1 should be loaded"
    assert "v2" in _MODELS, "Model v2 should be loaded"

# Test per verificare che i modelli caricati producano output con la forma corretta
def test_forward_output_shape():
    for version, model in _MODELS.items():
        dummy_input = torch.randn(1, 36, 3, 224, 224).float()  # Simula un batch di 36 immagini RGB
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape[0] == 1, f"Output batch size should be 1 for version {version}"

#test per inferenza con stesso input
def test_models_inference_output():

    class TransformHelper(BaseLearner):
        def fit(self): pass
        def test(self): pass

    helper = TransformHelper(use_cuda=False)

    for version, model in _MODELS.items():
        print(f"Current directory: {CURRENT_DIR}")
        load_npy = np.load(os.path.join(TEST_FILE_DIR, "0.npy"))
        
        val_transform = helper.get_transform(mode="rgb", phase="val", image_size=224)
        file_torch = val_transform(load_npy)
        file_torch = file_torch.unsqueeze(0).transpose(1, 2)
        with torch.no_grad():
            logit1 = model(file_torch)
            prob1 = torch.sigmoid(logit1).item()
            smoke1 = prob1 >= 0.5
            logit2 = model(file_torch)
            prob2 = torch.sigmoid(logit2).item()
            smoke2 = prob2 >= 0.5
        output1 = {"smoke": bool(smoke1), "confidence": prob1, "threshold": 0.5}        
        output2 = {"smoke": bool(smoke2), "confidence": prob2, "threshold": 0.5}
        assert output1 == output2, f"Inference outputs should be the same for the same input and version {version}"

#test per verificare se produce numeri finiti
def test_models_finite():
    for version, model in _MODELS.items():
        dummy_input = torch.randn(1, 36, 3, 224, 224).float()
        with torch.no_grad():
            output = model(dummy_input)
        assert torch.isfinite(output).all().item(), f"Inference outputs should be finite for version {version}"

#test per verificare output shape su batch >1
def test_batch_gt_1():
    for version, model in _MODELS.items():
        dummy_input = torch.randn(2, 36, 3, 224, 224).float()
        with torch.no_grad():
            output = model(dummy_input)
            assert output.shape[0] == 2, f"Output batch size should be 2 for version {version}"

#test per verificare che gettrasform dia un tensore e non un array
def test_get_transform_output_type():
    class TransformHelper(BaseLearner):
        def fit(self): pass
        def test(self): pass

    helper = TransformHelper(use_cuda=False)

    val_transform = helper.get_transform(mode="rgb", phase="val", image_size=224)
    dummy_input = np.random.rand(36, 224, 224, 3).astype(np.float32)
    output = val_transform(dummy_input)
    output = output.unsqueeze(0).transpose(1, 2)
    assert isinstance(output, torch.Tensor), "get_transform should return a torch.Tensor"
    assert output.shape == (1, 36, 3, 224, 224), "get_transform output shape should be (1, 36, 3, 224, 224)"


#TEST FUNZIONI
def test_n_frames_not_enough():
    frames =np.random.rand(10,224,224,3).astype(np.float32)
    with pytest.raises(ValueError, match="Not enough frames"):
        preprocess_npy(frames)

def test_n_frames_downsampled():
    frames = np.random.rand(72,224,224,3).astype(np.float32)
    sampled = preprocess_npy(frames)
    assert sampled.shape[0] == 36, "preprocess_npy should return 36 frames"

def test_inference():
    file = np.load(os.path.join(TEST_FILE_DIR,"0.npy"))
    for version in _MODELS.keys():
        output = inference(file, version=version, threshold=0.5)
        assert "smoke" in output, "Inference output should contain 'smoke' key"
        assert "confidence" in output, "Inference output should contain 'confidence' key"
        assert "threshold" in output, "Inference output should contain 'threshold' key"
        assert isinstance(output["smoke"], bool), "'smoke' value should be a boolean"
        assert isinstance(output["confidence"], float), "'confidence' value should be a float"
        assert isinstance(output["threshold"], float), "'threshold' value should be a float"