import numpy as np
import pandas as pd


TRAIN_PATH = "data/processed/train.parquet"


def build_bpr_mappings():

    train = pd.read_parquet(
        TRAIN_PATH
    )

    print(
        "Original training interactions:",
        len(train)
    )

    # Same active-interaction filtering
    user_counts = (
        train["user_id"]
        .value_counts()
    )

    item_counts = (
        train["item_id"]
        .value_counts()
    )

    # Keep the active users/items used by BPR
    active_users = (
        user_counts[
            user_counts >= 5
        ].index
    )

    active_items = (
        item_counts[
            item_counts >= 5
        ].index
    )

    active_train = train[
        train["user_id"].isin(
            active_users
        )
        &
        train["item_id"].isin(
            active_items
        )
    ].copy()

    print(
        "Active interactions:",
        len(active_train)
    )

    print(
        "Active users:",
        active_train["user_id"].nunique()
    )

    print(
        "Active items:",
        active_train["item_id"].nunique()
    )

    # IMPORTANT:
    # Use sorted IDs so mapping is deterministic.
    bpr_user_ids = np.sort(
        active_train["user_id"]
        .unique()
    )

    bpr_item_ids = np.sort(
        active_train["item_id"]
        .unique()
    )

    np.save(
        "models/bpr_user_ids.npy",
        bpr_user_ids
    )

    np.save(
        "models/bpr_item_ids.npy",
        bpr_item_ids
    )

    print(
        "Saved BPR user IDs:",
        len(bpr_user_ids)
    )

    print(
        "Saved BPR item IDs:",
        len(bpr_item_ids)
    )


if __name__ == "__main__":
    build_bpr_mappings()