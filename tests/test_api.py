import pytest
import unittest.mock as mock
import os
import sys
from fastapi.testclient import TestClient
import unittest.mock as mock
import numpy as np

CURRENT_FILE = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.src.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def make_dummy_npy(tmp_path):
    dummy_data = np.random.randint(0, 255, (50, 224, 224, 3), dtype=np.uint8)
    npy_path = tmp_path / "dummy.npy"
    np.save(npy_path, dummy_data)
    return npy_path

@pytest.fixture
def make_wrong_input(tmp_path):
    path = tmp_path /"wrong.mp3"
    path.write_bytes(b"contenuto qualsiasi")
    return path

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@mock.patch("api.src.main.inference")
def test_predict(mock_inference, client, make_dummy_npy):
    mock_inference.return_value = {"smoke": True, "confidence": 0.9, "threshold": 0.5}

    with open(make_dummy_npy, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("dummy.npy", f, "application/octet-stream")},
            data={"version": "v1"}
        )

    assert response.status_code == 200
    mock_inference.assert_called_once()
    assert response.json() == {"smoke": True, "confidence": 0.9, "threshold": 0.5}

def test_mp3(make_wrong_input, client):
    with open(make_wrong_input, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("wrong.mp3", f, "application/octet-stream")},
            data={"version": "v1"}
        )
    assert response.status_code == 415
    
    