import pandas as pd
from pathlib import Path


TRAIN_PATH = Path(
    "data/processed/train.parquet"
)

OUTPUT_PATH = Path(
    "models/popularity_scores.csv"
)


def build_popularity_artifact():

    print("Loading training interactions...")

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "item_id",
            "interaction_strength"
        ]
    )

    print(
        f"Interactions loaded: {len(train):,}"
    )

    popularity = (
        train
        .groupby("item_id")[
            "interaction_strength"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
        .rename("popularity_score")
        .reset_index()
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    popularity.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"Unique items: {len(popularity):,}"
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print("\nTop 10 popular items:")

    print(
        popularity.head(10)
    )


if __name__ == "__main__":
    build_popularity_artifact()