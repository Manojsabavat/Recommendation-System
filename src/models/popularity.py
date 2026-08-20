import pandas as pd


class PopularityRecommender:
    """
    Popularity-based recommendation model.

    Can either:
    1. Fit from historical interactions, or
    2. Load a precomputed popularity artifact.
    """

    def __init__(self):
        self.item_scores = None
        self.fitted = False

    def fit(
        self,
        interactions,
        item_col="item_id",
        strength_col="interaction_strength"
    ):
        if item_col not in interactions.columns:
            raise ValueError(
                f"Missing column: {item_col}"
            )

        if strength_col not in interactions.columns:
            raise ValueError(
                f"Missing column: {strength_col}"
            )

        self.item_scores = (
            interactions
            .groupby(item_col)[strength_col]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        self.fitted = True

        return self

    def load(self, path):
        """
        Load precomputed popularity scores.
        """

        popularity = pd.read_csv(path)

        required_columns = {
            "item_id",
            "popularity_score"
        }

        missing = (
            required_columns
            - set(popularity.columns)
        )

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        self.item_scores = (
            popularity
            .set_index("item_id")[
                "popularity_score"
            ]
            .sort_values(
                ascending=False
            )
        )

        self.fitted = True

        return self

    def recommend(
        self,
        k=10,
        exclude_items=None
    ):

        if not self.fitted:
            raise RuntimeError(
                "Model must be fitted or loaded "
                "before recommendation."
            )

        if exclude_items is None:
            exclude_items = set()
        else:
            exclude_items = set(
                exclude_items
            )

        recommendations = []

        for item_id in self.item_scores.index:

            if int(item_id) in exclude_items:
                continue

            recommendations.append(
                int(item_id)
            )

            if len(recommendations) >= k:
                break

        return recommendations

    def score(self, item_id):

        if not self.fitted:
            raise RuntimeError(
                "Model must be fitted or loaded "
                "before scoring."
            )

        return float(
            self.item_scores.get(
                item_id,
                0.0
            )
        )