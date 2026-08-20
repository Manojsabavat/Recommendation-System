from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "recommendation-system"


def test_warm_user_recommendations():
    response = client.get("/recommend/172?k=10")

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == 172
    assert data["strategy"] == "hybrid"
    assert len(data["recommendations"]) == 10

    assert all(
        isinstance(item, int)
        for item in data["recommendations"]
    )


def test_cold_user_recommendations():
    response = client.get(
        "/recommend/999999999?k=10"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == 999999999
    assert (
        data["strategy"]
        == "cold_user_popularity"
    )
    assert len(data["recommendations"]) == 10

    assert all(
        isinstance(item, int)
        for item in data["recommendations"]
    )


def test_cold_item_recommendations():
    response = client.get(
        "/recommend/3/cold-items?k=10"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == 3
    assert (
        data["strategy"]
        == "cold_item_category"
    )
    assert len(data["recommendations"]) == 10

    assert all(
        isinstance(item, int)
        for item in data["recommendations"]
    )


def test_cold_item_expected_output():
    response = client.get(
        "/recommend/3/cold-items?k=10"
    )

    assert response.status_code == 200

    data = response.json()

    expected = [
        3,
        8809,
        272037,
        403570,
        167183,
        168618,
        302384,
        303241,
        173927,
        177582
    ]

    assert data["recommendations"] == expected


def test_different_k_values():

    for k in [5, 10, 15, 20]:

        response = client.get(
            f"/recommend/172?k={k}"
        )

        assert response.status_code == 200

        data = response.json()

        assert len(data["recommendations"]) == k


def test_invalid_k():

    response = client.get(
        "/recommend/172?k=0"
    )

    assert response.status_code == 422


def test_warm_user_strategy():

    response = client.get(
        "/users/172/strategy"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == 172
    assert data["strategy"] == "hybrid"
    assert data["is_warm_user"] is True


def test_cold_user_strategy():

    response = client.get(
        "/users/999999999/strategy"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == 999999999

    assert (
        data["strategy"]
        == "cold_user_popularity"
    )

    assert data["is_warm_user"] is False