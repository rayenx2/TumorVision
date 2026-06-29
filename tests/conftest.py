import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_model_service
from api.main import app


class MockModelService:
    def __init__(self):
        self.is_loaded = False

    def load(self):
        self.is_loaded = True

    def get_model(self):
        return "mock_model"

    def get_class_names(self):
        return ["Cyst", "Normal", "Stone", "Tumor"]


@pytest.fixture
def mock_model_svc():
    svc = MockModelService()
    svc.load()
    return svc


@pytest.fixture
def app_client(mock_model_svc):
    app.dependency_overrides[get_model_service] = lambda: mock_model_svc
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()
