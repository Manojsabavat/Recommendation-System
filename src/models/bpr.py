import numpy as np
import torch
import torch.nn as nn


class BPRRecommender:

    def __init__(
        self,
        model_path,
        device="cpu"
    ):
        self.device = torch.device(device)

        # Load the saved checkpoint
        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False
        )

        # Load the exact mappings saved inside
        # the original BPR checkpoint
        self.user_to_idx = checkpoint[
            "cf_user_to_idx"
        ]

        self.item_to_idx = checkpoint[
            "cf_item_to_idx"
        ]

        # Reverse mappings
        self.idx_to_user = {
            index: user_id
            for user_id, index
            in self.user_to_idx.items()
        }

        self.idx_to_item = {
            index: item_id
            for item_id, index
            in self.item_to_idx.items()
        }

        # Load embedding matrices directly
        self.user_embedding = (
            checkpoint["user_embedding"]
            .to(self.device)
        )

        self.item_embedding = (
            checkpoint["item_embedding"]
            .to(self.device)
        )

        # Keep arrays available for convenience
        self.user_ids = np.array(
            [
                self.idx_to_user[i]
                for i in range(
                    len(self.idx_to_user)
                )
            ]
        )

        self.item_ids = np.array(
            [
                self.idx_to_item[i]
                for i in range(
                    len(self.idx_to_item)
                )
            ]
        )

    def recommend(
        self,
        user_id,
        k=10,
        exclude_items=None
    ):
        """
        Generate top-K BPR recommendations
        for a known BPR user.
        """

        if user_id not in self.user_to_idx:
            return []

        user_idx = self.user_to_idx[
            user_id
        ]

        user_vector = (
            self.user_embedding[user_idx]
        )

        # Score against every BPR item
        scores = torch.matmul(
            self.item_embedding,
            user_vector
        )

        scores = (
            scores
            .detach()
            .cpu()
            .numpy()
        )

        # Remove already-seen items
        if exclude_items is not None:

            for item_id in exclude_items:

                item_idx = (
                    self.item_to_idx.get(
                        item_id
                    )
                )

                if item_idx is not None:
                    scores[item_idx] = -np.inf

        top_indices = np.argsort(
            scores
        )[::-1][:k]

        recommendations = [
            int(
                self.idx_to_item[index]
            )
            for index in top_indices
        ]

        return recommendations

    def score_items(
        self,
        user_id,
        item_ids
    ):
        """
        Return BPR scores for specific items.
        """

        if user_id not in self.user_to_idx:
            return {}

        user_idx = self.user_to_idx[
            user_id
        ]

        user_vector = (
            self.user_embedding[user_idx]
        )

        result = {}

        for item_id in item_ids:

            item_idx = (
                self.item_to_idx.get(
                    item_id
                )
            )

            if item_idx is None:
                continue

            item_vector = (
                self.item_embedding[item_idx]
            )

            score = torch.dot(
                user_vector,
                item_vector
            ).item()

            result[int(item_id)] = float(
                score
            )

        return result