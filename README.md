# 🌾 Agri Inventory & Price Forecasting System

A comprehensive agricultural inventory management and commodity price forecasting platform powered by real-time mandi data (Agmarknet / data.gov.in), machine learning forecasting models, and an AI-driven warehouse decision engine.

---

## 🚀 Key Features

- **Real-Time Mandi Price Integration**: Syncs live market prices directly from `data.gov.in` (Agmarknet).
- **Price Forecasting Engine**: Multi-horizon commodity price predictions using Machine Learning (Random Forest, XGBoost, LSTM).
- **Smart Reorder & Procurement Engine**: Suggests optimal reorder schedules and procurement decisions factoring in forecasted price trends and storage costs.
- **AI Decision Assistant**: Natural language decision support powered by Google Gemini.
- **Dual-Portal UI**: Streamlit interface supporting Warehouse Admin and Vendor Purchase workflows.
- **RESTful API**: Fast, asynchronous FastAPI backend with PostgreSQL and SQLite support.

---

## 📁 Repository Structure

```text
├── data/
│   ├── raw/                  # Raw mandi price datasets (.gitkeep)
│   └── processed/            # Engineered features & datasets (.gitkeep)
├── models/
│   └── artifacts/            # Model weights, scalers & metadata (.gitkeep)
├── src/
│   ├── api/                  # FastAPI routes, schemas & server entry point
│   │   ├── main.py
│   │   ├── routes_forecast.py
│   │   ├── routes_inventory.py
│   │   └── schemas.py
│   ├── database/             # SQLAlchemy ORM models, connection & seed data
│   │   ├── connection.py
│   │   ├── models.py
│   │   └── seed.py
│   ├── ml/                   # ML pipelines: data fetch, features, training & inference
│   │   ├── check_historical.py
│   │   ├── evaluate.py
│   │   ├── features.py
│   │   ├── inference.py
│   │   ├── live_api_client.py
│   │   └── train.py
│   ├── services/             # Core business logic: reorder engine, sync, LLM agent
│   │   ├── live_sync.py
│   │   ├── llm_agent.py
│   │   └── reorder_engine.py
│   └── ui/                   # Streamlit dashboard
│       └── app.py
├── tests/                    # Unit and integration tests
├── .env.example              # Example environment variables template
├── .gitignore                # Excludes secrets, datasets, and compiled binaries
└── requirements.txt          # Python dependencies
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Sukhpreet2572/Agri_Inventory_System.git
cd Agri_Inventory_System
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```
Open `.env` and fill in your keys:
- `DATA_GOV_API_KEY`: API key from [data.gov.in](https://data.gov.in).
- `DATABASE_URL`: Connection string for PostgreSQL or SQLite (`sqlite:///./data/smart_inventory.db`).
- `GEMINI_API_KEY`: API key from Google AI Studio (optional for AI agent).

---

## 🚦 Running the Application

### 1. Initialize and Seed Database
```bash
python -m src.database.seed
```

### 2. Start FastAPI Backend
```bash
uvicorn src.api.main:app --reload --port 8000
```
API Documentation will be available at: `http://127.0.0.1:8000/docs`

### 3. Launch Streamlit UI
In a separate terminal:
```bash
streamlit run src/ui/app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔒 Security Note
No API keys, passwords, credentials, or sensitive datasets are tracked or committed in this repository. All credentials are read from `.env` at runtime.
