from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional

from src.inference.recommender import Recommender


# Application

app = FastAPI(
    title="Recommendation System API",
    description=(
        "Hybrid recommendation system using "
        "Content-Based Filtering, BPR, Popularity, "
        "and Cold-Item Category recommendations."
    ),
    version="1.0.0"
)


# Load recommender once when API starts

recommender = Recommender(
    models_dir="models",
    data_dir="data/processed"
)


# Response schema

class RecommendationResponse(BaseModel):

    user_id: int

    strategy: str

    recommendations: List[int]


# Health check

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "recommendation-system"
    }


# Standard recommendations

@app.get(
    "/recommend/{user_id}",
    response_model=RecommendationResponse
)
def recommend(
    user_id: int,
    k: int = Query(
        default=10,
        ge=1,
        le=100
    )
):

    try:

        result = recommender.recommend(
            user_id=user_id,
            k=k
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# Standard recommendations with exclusions

@app.get(
    "/recommend/{user_id}/exclude",
    response_model=RecommendationResponse
)
def recommend_with_exclusions(
    user_id: int,
    k: int = Query(
        default=10,
        ge=1,
        le=100
    ),
    exclude: Optional[str] = None
):

    try:

        exclude_items = set()

        if exclude:

            exclude_items = {
                int(item.strip())
                for item in exclude.split(",")
                if item.strip()
            }

        result = recommender.recommend(
            user_id=user_id,
            k=k,
            exclude_items=exclude_items
        )

        return result

    except ValueError:

        raise HTTPException(
            status_code=400,
            detail=(
                "exclude must contain "
                "comma-separated integer item IDs."
            )
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# Cold-item recommendations

@app.get(
    "/recommend/{user_id}/cold-items",
    response_model=RecommendationResponse
)
def recommend_cold_items(
    user_id: int,
    k: int = Query(
        default=10,
        ge=1,
        le=100
    )
):

    try:

        result = (
            recommender
            .recommend_cold_items(
                user_id=user_id,
                k=k
            )
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# User strategy

@app.get("/users/{user_id}/strategy")
def user_strategy(
    user_id: int
):

    return {
        "user_id": user_id,
        "strategy": recommender.get_strategy(
            user_id
        ),
        "is_warm_user": recommender.is_warm_user(
            user_id
        )
    }