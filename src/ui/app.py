import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


# Page configuration

st.set_page_config(
    page_title="Intelligent Recommendation System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #f7f9fc;
    }

    /* Main content width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Header */
    .hero {
        background: linear-gradient(
            135deg,
            #111827,
            #1e3a8a
        );
        padding: 30px 35px;
        border-radius: 18px;
        margin-bottom: 25px;
        color: white;
        box-shadow: 0 8px 25px rgba(0,0,0,0.10);
    }

    .hero-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 16px;
        opacity: 0.88;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 3px 12px rgba(0,0,0,0.05);
        height: 100%;
    }

    .metric-label {
        color: #6b7280;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-value {
        color: #111827;
        font-size: 25px;
        font-weight: 750;
        margin-top: 7px;
    }

    /* Recommendation cards */
    .recommendation-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.04);
        transition: 0.2s ease;
    }

    .recommendation-card:hover {
        border-color: #93c5fd;
        box-shadow: 0 5px 16px rgba(37,99,235,0.10);
    }

    .rank {
        color: #2563eb;
        font-size: 13px;
        font-weight: 700;
    }

    .item-id {
        color: #111827;
        font-size: 19px;
        font-weight: 700;
        margin-top: 4px;
    }

    .item-description {
        color: #6b7280;
        font-size: 12px;
        margin-top: 3px;
    }

    /* Strategy banner */
    .strategy-banner {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 14px;
        padding: 18px 20px;
        margin: 20px 0;
    }

    .strategy-title {
        color: #1d4ed8;
        font-size: 17px;
        font-weight: 750;
    }

    .strategy-text {
        color: #4b5563;
        font-size: 13px;
        margin-top: 5px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 12px;
        padding: 25px 0 10px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# Header

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            🎯 Intelligent Recommendation System
        </div>
        <div class="hero-subtitle">
            Personalized recommendations powered by
            Content-Based Filtering, BPR Collaborative Filtering,
            Popularity and Cold-Start Modeling.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# Sidebar

with st.sidebar:

    st.header("Recommendation Settings")

    st.markdown(
        "Configure the recommendation request below."
    )

    st.divider()

    user_id = st.number_input(
        "User ID",
        min_value=0,
        value=172,
        step=1
    )

    k = st.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10
    )

    mode = st.selectbox(
        "Recommendation Mode",
        [
            "Personalized",
            "Cold Items"
        ]
    )

    st.divider()

    # API status
    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=3
        )

        api_online = (
            response.status_code == 200
        )

    except requests.RequestException:

        api_online = False

    if api_online:

        st.success("● API Online")

    else:

        st.error("● API Offline")

    st.caption(
        "FastAPI backend: "
        "127.0.0.1:8000"
    )


# Recommendation request

if st.button(
    "🚀 Get Recommendations",
    type="primary",
    use_container_width=True
):

    if not api_online:

        st.error(
            "FastAPI is not running. Start it using:"
        )

        st.code(
            "python -m uvicorn src.api.app:app --reload"
        )

        st.stop()

    try:

        # -------------------------------------------------
        # API request
        # -------------------------------------------------

        if mode == "Personalized":

            response = requests.get(
                f"{API_URL}/recommend/{user_id}",
                params={
                    "k": k
                },
                timeout=30
            )

        else:

            response = requests.get(
                f"{API_URL}/recommend/"
                f"{user_id}/cold-items",
                params={
                    "k": k
                },
                timeout=30
            )

        # -------------------------------------------------
        # Error handling
        # -------------------------------------------------

        if response.status_code != 200:

            st.error(
                f"API returned "
                f"HTTP {response.status_code}"
            )

            st.code(
                response.text
            )

            st.stop()

        result = response.json()

        recommendations = result[
            "recommendations"
        ]

        strategy = result[
            "strategy"
        ]

        # =================================================
        # Results header
        # =================================================

        st.subheader(
            "Recommendation Results"
        )

        # =================================================
        # Metrics
        # =================================================

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        User ID
                    </div>
                    <div class="metric-value">
                        {result["user_id"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            strategy_display = (
                strategy
                .replace("_", " ")
                .title()
            )

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        Strategy
                    </div>
                    <div class="metric-value">
                        {strategy_display}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        Recommendations
                    </div>
                    <div class="metric-value">
                        {len(recommendations)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # =================================================
        # Strategy explanation
        # =================================================

        if strategy == "hybrid":

            title = (
                "Hybrid Recommendation Engine"
            )

            description = (
                "40% Content-Based + "
                "40% BPR Collaborative Filtering + "
                "20% Popularity"
            )

        elif strategy == "cold_user_popularity":

            title = (
                "Cold-User Fallback"
            )

            description = (
                "The user is not present in the "
                "warm-user model intersection, so "
                "the system uses popularity-based "
                "recommendations."
            )

        elif strategy == "cold_item_category":

            title = (
                "Cold-Item Category Recommendation"
            )

            description = (
                "Cold items are ranked according to "
                "the user's learned category preferences "
                "and category overlap."
            )

        else:

            title = strategy_display

            description = (
                "Recommendation generated by "
                "the selected strategy."
            )

        st.markdown(
            f"""
            <div class="strategy-banner">
                <div class="strategy-title">
                    {title}
                </div>
                <div class="strategy-text">
                    {description}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # =================================================
        # Recommendation list
        # =================================================

        st.subheader(
            "Recommended Items"
        )

        if not recommendations:

            st.warning(
                "No recommendations were generated "
                "for this user."
            )

        else:

            for rank, item_id in enumerate(
                recommendations,
                start=1
            ):

                st.markdown(
                    f"""
                    <div class="recommendation-card">
                        <div class="rank">
                            RANK #{rank}
                        </div>
                        <div class="item-id">
                            Item {item_id}
                        </div>
                        <div class="item-description">
                            Recommendation score position:
                            {rank}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    except requests.RequestException as e:

        st.error(
            f"Could not connect to FastAPI: {e}"
        )

    except Exception as e:

        st.error(
            f"Unexpected error: {e}"
        )


# Project information

st.divider()

with st.expander(
    "ℹ️ About this Recommendation System"
):

    st.markdown(
        """
        ### Model Architecture

        The system combines multiple recommendation
        strategies:

        - **Content-Based Filtering** — TF-IDF item
          representations and user profiles.
        - **BPR Collaborative Filtering** — personalized
          implicit-feedback ranking.
        - **Popularity Baseline** — global item popularity.
        - **Hybrid Model** — optimized combination of
          Content, BPR and Popularity.
        - **Cold-Item Category Model** — category-overlap
          recommendations for unseen items.

        ### Hybrid Weights

        | Component | Weight |
        |---|---:|
        | Content-Based | 40% |
        | BPR | 40% |
        | Popularity | 20% |

        ### Serving Architecture

        **Streamlit → FastAPI → Unified Recommender → ML Models**
        """
    )


# Footer

st.markdown(
    """
    <div class="footer">
        Intelligent Recommendation System ·
        Content + Collaborative Filtering +
        Popularity + Cold Start
    </div>
    """,
    unsafe_allow_html=True
)