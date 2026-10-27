import sys
from pathlib import Path

import pytest

DASH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DASH))


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from server.app import app
    with TestClient(app) as c:
        yield c
