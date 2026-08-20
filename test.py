from src.inference.recommender import Recommender


recommender = Recommender(
    models_dir="models",
    data_dir="data/processed"
)


# =====================================================
# 1. Warm User -> Hybrid
# =====================================================

warm_result = recommender.recommend(
    user_id=172,
    k=10
)

print("\n========== WARM USER ==========")
print(warm_result)


# =====================================================
# 2. Unknown User -> Popularity
# =====================================================

cold_user_result = recommender.recommend(
    user_id=999999999,
    k=10
)

print("\n========== COLD USER ==========")
print(cold_user_result)


# =====================================================
# 3. Known User -> Cold Items
# =====================================================

cold_item_result = (
    recommender.recommend_cold_items(
        user_id=3,
        k=10
    )
)

print("\n========== COLD ITEMS ==========")
print(cold_item_result)