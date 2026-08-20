# Recommendation System

An end-to-end recommendation platform combining **Content-Based Filtering, BPR Collaborative Filtering, Popularity-Based Ranking, Hybrid Recommendation, and Cold-Start Handling**.

The system is implemented as a complete ML application with a **FastAPI inference backend**, **Streamlit interface**, offline ranking evaluation, persisted model artifacts, and automated API testing.

---

## Project Highlights

- Content-Based recommendation using TF-IDF
- BPR Collaborative Filtering
- Popularity-based recommendation baseline
- Hybrid ranking using Content + BPR + Popularity
- Warm-user recommendation routing
- Cold-user fallback strategy
- Cold-item category-based recommendation
- Precision, Recall, NDCG and HitRate evaluation
- FastAPI inference service
- Streamlit interactive UI
- Automated API test suite
- Saved model artifacts for inference

---

## System Architecture

```text
                         USER
                           |
                           v
                    Streamlit UI
                      Port 8501
                           |
                           v
                     FastAPI API
                      Port 8000
                           |
                           v
                 Unified Recommender
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
    Content-Based         BPR          Popularity
      Filtering           CF             Ranking
          |                |                |
          +----------------+----------------+
                           |
                           v
                    Hybrid Ranking
                     40% + 40% + 20%
                           |
                +----------+----------+
                |                     |
                v                     v
            Warm User             Cold User
             Hybrid              Popularity
                |
                v
            Cold Items
                |
                v
       Category-Based Ranking
```

---

## Recommendation Pipeline

```text
Raw Interaction Data
        |
        v
Data Preprocessing
        |
        v
Feature Engineering
        |
        +------------------+
        |                  |
        v                  v
 Content-Based          BPR CF
        |                  |
        +--------+---------+
                 |
                 v
          Hybrid Ranking
                 |
                 v
        Cold-Start Routing
                 |
                 v
          Unified Inference
                 |
        +--------+---------+
        |                  |
        v                  v
      FastAPI          Streamlit
```

---

## Models

### 1. Content-Based Filtering

TF-IDF representations are used to model item characteristics and construct user preference profiles.

The system uses sparse representations for efficient similarity-based recommendation.

### 2. BPR Collaborative Filtering

Bayesian Personalized Ranking learns latent user-item preference relationships from implicit interactions.

The trained checkpoint contains:

- user embeddings
- item embeddings
- user mappings
- item mappings

### 3. Popularity Baseline

Item popularity is calculated from training interactions and used as:

- a baseline model
- a fallback for cold users
- a robust candidate source

### 4. Hybrid Recommender

The warm-user system combines:

```text
40% Content-Based
40% BPR Collaborative Filtering
20% Popularity
```

This combines personalized item similarity, collaborative behavior, and global popularity.

---

## Cold-Start Handling

### Cold Users

Users without sufficient model history are automatically routed to:

```text
cold_user_popularity
```

This guarantees recommendations even when personalized models cannot produce reliable results.

### Cold Items

Items absent from the training interaction data are handled using category information.

The cold-item pipeline:

```text
User interaction history
        |
        v
User category profile
        |
        v
Cold-item category mapping
        |
        v
Category overlap scoring
        |
        v
Top-K cold-item recommendations
```

Strategy:

```text
cold_item_category
```

---

## Dataset

The project uses a large interaction/recommendation dataset containing:

- users
- items
- events
- timestamps
- item properties
- category information

The raw dataset is intentionally excluded from the GitHub repository because of its size.

Expected raw files:

```text
data/
├── raw/
│   ├── events.csv
│   ├── item_properties_part1.csv
│   ├── item_properties_part2.csv
│   └── category_tree.csv
│
└── processed/
    ├── interactions.parquet
    ├── train.parquet
    ├── validation.parquet
    ├── test.parquet
    └── item_categories.parquet
```

---

## Evaluation

The project evaluates recommendation quality using:

- Precision@K
- Recall@K
- NDCG@K
- HitRate@K

Evaluation artifacts are stored under:

```text
models/evaluation/
```

The final evaluation compares:

- Popularity
- BPR
- Content-Based
- Hybrid

The hybrid model provides the strongest overall ranking performance among the evaluated approaches, particularly on NDCG and HitRate.

---

## API

The backend is implemented using FastAPI.

### Health Check

```http
GET /health
```

Example:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

### Personalized Recommendations

```http
GET /recommend/{user_id}?k=10
```

Example:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/recommend/172?k=10"
```

A warm user is routed to the hybrid strategy.

### Cold User

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/recommend/999999999?k=10"
```

Example strategy:

```text
cold_user_popularity
```

### Cold Items

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/recommend/3/cold-items?k=10"
```

Example strategy:

```text
cold_item_category
```

---

## Streamlit UI

Start the API:

```powershell
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Open another terminal and start Streamlit:

```powershell
python -m streamlit run src/ui/app.py
```

Open:

```text
http://localhost:8501
```

The Streamlit application communicates with the FastAPI backend to generate recommendations.

---

## Testing

The project includes automated API tests using pytest.

Run:

```powershell
python -m pytest tests -v
```

Current validated result:

```text
9 passed
```

The test suite covers:

- health endpoint
- warm-user recommendations
- cold-user recommendations
- cold-item recommendations
- expected cold-item output
- different K values
- invalid K handling
- warm-user strategy
- cold-user strategy

---

## Project Structure

```text
Recommendation-System/
│
├── data/
│   ├── raw/                    # Local dataset, excluded from Git
│   └── processed/
│
├── models/
│   ├── bpr_model.pt
│   ├── product_tfidf.npz
│   ├── user_profiles.npz
│   ├── content_vectorizer.joblib
│   ├── product_ids.npy
│   ├── popularity_scores.csv
│   ├── warm_user_ids.npy
│   ├── cold_start/
│   └── evaluation/
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_popularity_baseline.ipynb
│   ├── 03_content_based.ipynb
│   ├── 04_collaborative_filtering.ipynb
│   ├── 05_matrix_factorization.ipynb
│   ├── 06_hybrid_recommender.ipynb
│   └── 07_cold_start.ipynb
│
├── reports/
│   └── model_results.csv
│
├── src/
│   ├── api/
│   │   └── app.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── inference/
│   │   └── recommender.py
│   ├── models/
│   │   ├── bpr.py
│   │   ├── content_based.py
│   │   ├── cold_item.py
│   │   ├── hybrid.py
│   │   ├── popularity.py
│   │   └── save_popularity.py
│   └── ui/
│       └── app.py
│
├── tests/
│   └── test_api.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Tech Stack

### Programming

- Python

### Machine Learning

- NumPy
- Pandas
- SciPy
- PyTorch
- TF-IDF
- Bayesian Personalized Ranking

### Backend

- FastAPI
- Uvicorn

### Frontend

- Streamlit

### Data

- CSV
- Parquet
- SciPy sparse matrices

### Testing

- Pytest
- HTTPX

### Development

- Jupyter Notebook
- VS Code
- Git
- GitHub

---

## Installation

Clone the repository:

```powershell
git clone https://github.com/Manojsabavat/Recommendation-System.git
cd Recommendation-System
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## Running the Project

### 1. Start the FastAPI Backend

```powershell
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

### 2. Start the Streamlit Frontend

Open a second terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run src/ui/app.py
```

Then open:

```text
http://localhost:8501
```

### 3. Run Automated Tests

```powershell
python -m pytest tests -v
```

---

## Development Workflow

```text
Data Audit
    ↓
Data Preprocessing
    ↓
Popularity Baseline
    ↓
Content-Based Filtering
    ↓
Collaborative Filtering / BPR
    ↓
Hybrid Recommendation
    ↓
Cold-Start Handling
    ↓
Offline Evaluation
    ↓
Unified Inference
    ↓
FastAPI
    ↓
Streamlit UI
    ↓
Automated Testing
```

---

## Key Features

- Personalized recommendations
- Content-based filtering
- BPR collaborative filtering
- Popularity baseline
- Hybrid recommendation architecture
- Warm-user routing
- Cold-user handling
- Cold-item handling
- Category-based cold-item recommendations
- Offline ranking evaluation
- FastAPI inference API
- Streamlit interactive UI
- Automated API testing
- Persisted model artifacts
- GitHub-ready project structure

---

## Project Status

**End-to-end recommendation system implemented and tested.**

The project includes model development, hybrid ranking, cold-start handling, offline evaluation, inference serving, an interactive UI, automated testing, and a cleaned GitHub repository structure.

---

## Author

**Sabavat Manoj**

IIT Kharagpur
