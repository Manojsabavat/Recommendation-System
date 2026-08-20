import numpy as np


def precision_at_k(recommended, relevant, k):
    """
    Precision@K:
    Fraction of the top-K recommendations
    that are relevant.
    """
    if k <= 0:
        return 0.0

    recommended = recommended[:k]

    if len(recommended) == 0:
        return 0.0

    hits = sum(
        item in relevant
        for item in recommended
    )

    return hits / len(recommended)


def recall_at_k(recommended, relevant, k):
    """
    Recall@K:
    Fraction of relevant items recovered
    within the top-K recommendations.
    """
    if not relevant:
        return 0.0

    recommended = recommended[:k]

    hits = sum(
        item in relevant
        for item in recommended
    )

    return hits / len(relevant)


def hit_rate_at_k(recommended, relevant, k):
    """
    HitRate@K:
    Whether at least one relevant item
    appears in the top-K recommendations.
    """
    recommended = recommended[:k]

    return float(
        any(
            item in relevant
            for item in recommended
        )
    )


def ndcg_at_k(recommended, relevant, k):
    """
    NDCG@K using binary relevance.
    """

    if not relevant:
        return 0.0

    recommended = recommended[:k]

    dcg = 0.0

    for rank, item in enumerate(
        recommended,
        start=1
    ):
        if item in relevant:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(
        len(relevant),
        k
    )

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    return dcg / idcg