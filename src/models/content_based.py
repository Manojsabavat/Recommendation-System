import numpy as np
from scipy.sparse import load_npz


class ContentBasedRecommender:
    """
    Content-based recommender using a precomputed
    item-feature matrix and user profiles.
    """

    def __init__(
        self,
        item_matrix_path,
        item_ids_path,
        user_profiles_path=None,
        user_ids_path=None
    ):
        self.item_matrix = load_npz(
            item_matrix_path
        )

        self.item_ids = np.load(
            item_ids_path
        )

        self.item_id_to_index = {
            int(item_id): idx
            for idx, item_id
            in enumerate(self.item_ids)
        }

        self.user_profiles = None
        self.user_ids = None
        self.user_id_to_index = {}

        if (
            user_profiles_path is not None
            and user_ids_path is not None
        ):
            self.user_profiles = load_npz(
                user_profiles_path
            )

            self.user_ids = np.load(
                user_ids_path
            )

            self.user_id_to_index = {
                int(user_id): idx
                for idx, user_id
                in enumerate(self.user_ids)
            }

    def get_user_profile(self, user_id):
        """
        Return the stored content profile
        for a known user.
        """

        if self.user_profiles is None:
            raise RuntimeError(
                "User profiles are not loaded."
            )

        if user_id not in self.user_id_to_index:
            return None

        index = self.user_id_to_index[
            user_id
        ]

        return self.user_profiles[index]

    def recommend(
        self,
        user_id,
        k=10,
        exclude_items=None
    ):
        """
        Generate top-K content-based recommendations.
        """

        profile = self.get_user_profile(
            user_id
        )

        if profile is None:
            return []

        scores = (
            profile @ self.item_matrix.T
        ).toarray().ravel()

        if exclude_items is not None:
            for item_id in exclude_items:

                index = self.item_id_to_index.get(
                    int(item_id)
                )

                if index is not None:
                    scores[index] = -np.inf

        top_indices = np.argsort(
            scores
        )[::-1][:k]

        return [
            int(self.item_ids[index])
            for index in top_indices
        ]

    def score_items(
        self,
        user_id,
        item_ids
    ):
        """
        Return content scores for selected items.
        """

        profile = self.get_user_profile(
            user_id
        )

        if profile is None:
            return {}

        scores = (
            profile @ self.item_matrix.T
        ).toarray().ravel()

        result = {}

        for item_id in item_ids:

            index = self.item_id_to_index.get(
                int(item_id)
            )

            if index is not None:
                result[int(item_id)] = float(
                    scores[index]
                )

        return result