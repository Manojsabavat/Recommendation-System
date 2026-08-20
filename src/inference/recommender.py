import pandas as pd

from pathlib import Path

from src.models.popularity import PopularityRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.bpr import BPRRecommender
from src.models.hybrid import HybridRecommender
from src.models.cold_item import ColdItemCategoryRecommender


class Recommender:
    """
    Unified recommendation engine.

    Routing:

    warm user
        -> Hybrid

    cold user
        -> Popularity

    cold-item mode
        -> Category-based cold-item recommender
    """

    def __init__(
        self,
        models_dir="models",
        data_dir="data/processed"
    ):

        self.models_dir = Path(
            models_dir
        )

        self.data_dir = Path(
            data_dir
        )

        # User interaction history
        self.train = pd.read_parquet(
            self.data_dir / "train.parquet",
            columns=["user_id", "item_id"]
        )

        self.user_seen_items = (
            self.train
            .groupby("user_id")["item_id"]
            .apply(set)
            .to_dict()
        )

        # Content-Based Model

        self.content_model = (
            ContentBasedRecommender(
                str(
                    self.models_dir
                    / "product_tfidf.npz"
                ),
                str(
                    self.models_dir
                    / "product_ids.npy"
                ),
                str(
                    self.models_dir
                    / "user_profiles.npz"
                ),
                str(
                    self.models_dir
                    / "warm_user_ids.npy"
                )
            )
        )

        # BPR Model

        self.bpr_model = BPRRecommender(
            str(
                self.models_dir
                / "bpr_model.pt"
            )
        )

        # Popularity Model

        self.popularity_model = (
            PopularityRecommender()
        )

        self.popularity_model.load(
            str(
                self.models_dir
                / "popularity_scores.csv"
            )
        )

        # Hybrid Model

        self.hybrid_model = HybridRecommender(
            content_model=self.content_model,
            bpr_model=self.bpr_model,
            popularity_model=self.popularity_model,
            content_weight=0.4,
            bpr_weight=0.4,
            popularity_weight=0.2
        )

        # Cold-Item Category Model

        self.cold_item_model = (
            ColdItemCategoryRecommender(
                item_categories_path=str(
                    self.data_dir
                    / "item_categories.parquet"
                ),
                cold_item_ids_path=str(
                    self.models_dir
                    / "cold_start"
                    / "cold_item_ids.npy"
                ),
                train_path=str(
                    self.data_dir
                    / "train.parquet"
                )
            )
        )

        # User Sets

        self.content_users = {
            int(user_id)
            for user_id
            in self.content_model.user_ids
        }

        self.bpr_users = {
            int(user_id)
            for user_id
            in self.bpr_model.user_ids
        }

        self.warm_users = (
            self.content_users
            &
            self.bpr_users
        )

        print(
            f"Content users: "
            f"{len(self.content_users):,}"
        )

        print(
            f"BPR users: "
            f"{len(self.bpr_users):,}"
        )

        print(
            f"Warm users: "
            f"{len(self.warm_users):,}"
        )

    # Standard Recommendation

    def recommend(
        self,
        user_id,
        k=10,
        exclude_items=None
    ):
        """
        Standard recommendation route.

        Warm user:
            Hybrid 40/40/20

        Cold user:
            Popularity
        """

        user_id = int(user_id)

        if k <= 0:
            return {
                "user_id": user_id,
                "strategy": "none",
                "recommendations": []
            }

        # Explicit exclusions
        if exclude_items is None:
            exclude_items = set()
        else:
            exclude_items = {
                int(item)
                for item in exclude_items
            }

# Automatically exclude items already seen by the user
        seen_items = self.user_seen_items.get(
            user_id,
            set()
        )

        exclude_items = (
            set(exclude_items)
            | {int(item) for item in seen_items}
        )

        # Warm user

        if user_id in self.warm_users:

            recommendations = (
                self.hybrid_model.recommend(
                    user_id=user_id,
                    k=k,
                    exclude_items=exclude_items
                )
            )

            return {
                "user_id": user_id,
                "strategy": "hybrid",
                "recommendations": recommendations
            }

        # Cold user

        recommendations = (
            self.popularity_model.recommend(
                k=k,
                exclude_items=exclude_items
            )
        )

        return {
            "user_id": user_id,
            "strategy": "cold_user_popularity",
            "recommendations": recommendations
        }

    # Cold Item Recommendation

    def recommend_cold_items(
        self,
        user_id,
        k=10
    ):
        """
        Recommend previously unseen cold items
        using the user's category preferences.
        """

        user_id = int(user_id)

        recommendations = (
            self.cold_item_model.recommend(
                user_id=user_id,
                k=k
            )
        )

        return {
            "user_id": user_id,
            "strategy": "cold_item_category",
            "recommendations": recommendations
        }

    # Convenience Method

    def recommend_items(
        self,
        user_id,
        k=10,
        exclude_items=None
    ):
        """
        Return only recommendation IDs.
        """

        result = self.recommend(
            user_id=user_id,
            k=k,
            exclude_items=exclude_items
        )

        return result["recommendations"]

    # User Information

    def is_warm_user(
        self,
        user_id
    ):
        return (
            int(user_id)
            in self.warm_users
        )

    def get_strategy(
        self,
        user_id
    ):
        if self.is_warm_user(
            user_id
        ):
            return "hybrid"

        return "cold_user_popularity"