import numpy as np


class HybridRecommender:
    """
    Hybrid recommender combining:

    - Content-Based recommendations
    - BPR collaborative filtering
    - Popularity baseline

    Default weights:
        Content     = 0.4
        BPR         = 0.4
        Popularity  = 0.2
    """

    def __init__(
        self,
        content_model,
        bpr_model,
        popularity_model,
        content_weight=0.4,
        bpr_weight=0.4,
        popularity_weight=0.2
    ):

        total = (
            content_weight
            + bpr_weight
            + popularity_weight
        )

        if not np.isclose(total, 1.0):
            raise ValueError(
                "Hybrid weights must sum to 1.0"
            )

        self.content_model = content_model
        self.bpr_model = bpr_model
        self.popularity_model = popularity_model

        self.content_weight = content_weight
        self.bpr_weight = bpr_weight
        self.popularity_weight = popularity_weight

    @staticmethod
    def min_max_normalize(scores):
        """
        Min-max normalize scores to [0, 1].
        """

        scores = np.asarray(
            scores,
            dtype=float
        )

        if len(scores) == 0:
            return scores

        finite_mask = np.isfinite(scores)

        if not finite_mask.any():
            return np.zeros_like(scores)

        finite_scores = scores[
            finite_mask
        ]

        minimum = finite_scores.min()
        maximum = finite_scores.max()

        if np.isclose(
            minimum,
            maximum
        ):
            normalized = np.zeros_like(
                scores
            )

            normalized[finite_mask] = 1.0

            return normalized

        normalized = np.zeros_like(
            scores
        )

        normalized[finite_mask] = (
            (
                finite_scores - minimum
            )
            /
            (
                maximum - minimum
            )
        )

        return normalized

    def recommend(
        self,
        user_id,
        k=10,
        exclude_items=None
    ):
        """
        Generate hybrid recommendations.

        The models are combined using normalized
        scores rather than simply merging top-K lists.
        """

        if exclude_items is None:
            exclude_items = set()
        else:
            exclude_items = set(
                int(item)
                for item in exclude_items
            )

        # Determine candidate universe

        candidate_items = set(
            int(item)
            for item in self.content_model.item_ids
        )

        candidate_items.update(
            int(item)
            for item in self.bpr_model.item_ids
        )

        candidate_items.update(
            int(item)
            for item in self.popularity_model.item_scores.index
        )

        candidate_items -= exclude_items

        if not candidate_items:
            return []

        candidate_items = np.array(
            list(candidate_items),
            dtype=np.int64
        )

        # Content scores

        content_scores_dict = (
            self.content_model.score_items(
                user_id,
                candidate_items
            )
        )

        content_scores = np.array(
            [
                content_scores_dict.get(
                    int(item),
                    0.0
                )
                for item in candidate_items
            ],
            dtype=float
        )

        # BPR scores

        bpr_scores_dict = (
            self.bpr_model.score_items(
                user_id,
                candidate_items
            )
        )

        bpr_scores = np.array(
            [
                bpr_scores_dict.get(
                    int(item),
                    0.0
                )
                for item in candidate_items
            ],
            dtype=float
        )

        # Popularity scores

        popularity_scores = np.array(
            [
                self.popularity_model.score(
                    int(item)
                )
                for item in candidate_items
            ],
            dtype=float
        )

        # Normalize each component

        content_normalized = (
            self.min_max_normalize(
                content_scores
            )
        )

        bpr_normalized = (
            self.min_max_normalize(
                bpr_scores
            )
        )

        popularity_normalized = (
            self.min_max_normalize(
                popularity_scores
            )
        )

        # Weighted hybrid score

        hybrid_scores = (
            self.content_weight
            * content_normalized
            +
            self.bpr_weight
            * bpr_normalized
            +
            self.popularity_weight
            * popularity_normalized
        )

        # Rank

        top_indices = np.argsort(
            hybrid_scores
        )[::-1][:k]

        return [
            int(candidate_items[index])
            for index in top_indices
        ]

    def score_items(
        self,
        user_id,
        item_ids
    ):
        """
        Return detailed hybrid scores for
        selected items.
        """

        item_ids = [
            int(item)
            for item in item_ids
        ]

        content_dict = (
            self.content_model.score_items(
                user_id,
                item_ids
            )
        )

        bpr_dict = (
            self.bpr_model.score_items(
                user_id,
                item_ids
            )
        )

        popularity_dict = {
            item: self.popularity_model.score(
                item
            )
            for item in item_ids
        }

        content_raw = np.array(
            [
                content_dict.get(
                    item,
                    0.0
                )
                for item in item_ids
            ]
        )

        bpr_raw = np.array(
            [
                bpr_dict.get(
                    item,
                    0.0
                )
                for item in item_ids
            ]
        )

        popularity_raw = np.array(
            [
                popularity_dict[item]
                for item in item_ids
            ]
        )

        content_norm = (
            self.min_max_normalize(
                content_raw
            )
        )

        bpr_norm = (
            self.min_max_normalize(
                bpr_raw
            )
        )

        popularity_norm = (
            self.min_max_normalize(
                popularity_raw
            )
        )

        hybrid = (
            self.content_weight
            * content_norm
            +
            self.bpr_weight
            * bpr_norm
            +
            self.popularity_weight
            * popularity_norm
        )

        return [
            {
                "item_id": item_ids[i],
                "content_score": float(
                    content_norm[i]
                ),
                "bpr_score": float(
                    bpr_norm[i]
                ),
                "popularity_score": float(
                    popularity_norm[i]
                ),
                "hybrid_score": float(
                    hybrid[i]
                )
            }
            for i in range(
                len(item_ids)
            )
        ]