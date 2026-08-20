import numpy as np
import pandas as pd


class ColdItemCategoryRecommender:
    """
    Category-based recommender for cold items.

    This reproduces the cold-item recommendation logic
    from 07_cold_start(2).ipynb.
    """

    def __init__(
        self,
        item_categories_path,
        cold_item_ids_path,
        train_path
    ):

        # ------------------------------------------------
        # Load item-category mapping
        # ------------------------------------------------

        self.item_categories = pd.read_parquet(
            item_categories_path
        )

        required_columns = {
            "itemid",
            "categoryid"
        }

        missing_columns = (
            required_columns
            - set(self.item_categories.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing columns in item_categories: "
                f"{missing_columns}"
            )

        # ------------------------------------------------
        # Load exact cold-item IDs
        # ------------------------------------------------

        self.cold_item_ids = np.asarray(
            np.load(cold_item_ids_path),
            dtype=np.int64
        )

        # Preserve the original notebook's
        # candidate ordering
        self.cold_item_ids = np.asarray(
            self.cold_item_ids
        )

        # ------------------------------------------------
        # Cold item -> category mapping
        # ------------------------------------------------

        self.cold_item_to_category = (
            self.item_categories[
                self.item_categories["itemid"].isin(
                    self.cold_item_ids
                )
            ]
            .groupby("itemid")["categoryid"]
            .apply(set)
            .to_dict()
        )

        # ------------------------------------------------
        # Load training interactions
        # ------------------------------------------------

        self.train = pd.read_parquet(
            train_path,
            columns=[
                "user_id",
                "item_id",
                "interaction_strength"
            ]
        )

        # ------------------------------------------------
        # User category profile
        #
        # IMPORTANT:
        # The original notebook uses
        # interaction_strength.sum(), NOT count().
        # ------------------------------------------------

        train_with_categories = self.train.merge(
            self.item_categories,
            left_on="item_id",
            right_on="itemid",
            how="inner"
        )

        user_category_counts = (
            train_with_categories
            .groupby(
                ["user_id", "categoryid"]
            )["interaction_strength"]
            .sum()
        )

        self.user_category_profile = (
            user_category_counts
            .groupby(level=0)
            .apply(
                lambda x:
                x.sort_values(
                    ascending=False
                )
                .index
                .get_level_values(
                    "categoryid"
                )
                .tolist()
            )
            .to_dict()
        )

        # ------------------------------------------------
        # Precompute seen items
        # ------------------------------------------------

        self.user_seen_items = (
            self.train
            .groupby("user_id")["item_id"]
            .apply(set)
            .to_dict()
        )

        # ------------------------------------------------
        # Diagnostics
        # ------------------------------------------------

        print(
            "Cold items:",
            len(self.cold_item_ids)
        )

        print(
            "Cold items with category mapping:",
            len(
                self.cold_item_to_category
            )
        )

        print(
            "Users with category profiles:",
            len(
                self.user_category_profile
            )
        )

    def recommend(
        self,
        user_id,
        k=10
    ):
        """
        Generate cold-item recommendations.

        Exact notebook logic:

        1. Get user's preferred categories.
        2. Remove items already seen by user.
        3. Check category overlap.
        4. Keep candidates with overlap > 0.
        5. Sort by overlap descending.
        6. Return top-K.
        """

        user_id = int(user_id)

        preferred_categories = set(
            self.user_category_profile.get(
                user_id,
                []
            )
        )

        if not preferred_categories:
            return []

        # Items already seen by the user
        seen_items = self.user_seen_items.get(
            user_id,
            set()
        )

        candidates = []

        # IMPORTANT:
        # Iterate in the original cold_item_ids
        # order, exactly like the notebook.
        for item_id in self.cold_item_ids:

            item_id = int(item_id)

            # Never recommend training items
            if item_id in seen_items:
                continue

            categories = (
                self.cold_item_to_category.get(
                    item_id,
                    set()
                )
            )

            if not categories:
                continue

            overlap = len(
                preferred_categories
                & categories
            )

            if overlap > 0:
                candidates.append(
                    (
                        item_id,
                        overlap
                    )
                )

        # Exact notebook sorting:
        # overlap only, descending.
        candidates.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return [
            item_id
            for item_id, score
            in candidates[:k]
        ]

    def is_cold_item(
        self,
        item_id
    ):
        """
        Check whether an item belongs to
        the cold-item candidate set.
        """

        return int(item_id) in set(
            self.cold_item_ids
        )

    def get_category_overlap(
        self,
        user_id,
        item_id
    ):
        """
        Return the category overlap between
        a user and a cold item.
        """

        user_id = int(user_id)
        item_id = int(item_id)

        preferred_categories = set(
            self.user_category_profile.get(
                user_id,
                []
            )
        )

        item_categories = (
            self.cold_item_to_category.get(
                item_id,
                set()
            )
        )

        return len(
            preferred_categories
            & item_categories
        )