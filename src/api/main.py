import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes_forecast import router as forecast_router
from src.api.routes_inventory import router as inventory_router


from fastapi import Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.services.llm_agent import InventoryAIAssistant


app = FastAPI(
    title="Smart Agri-Inventory & Price Forecasting API",
    description="Real-time commodity price forecasting (data.gov.in + Random Forest) and price-driven warehouse procurement engine.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inventory_router)
app.include_router(forecast_router)


@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy", "service": "smart-inventory-api"}


@app.get("/api/v1/model-benchmarks", tags=["System"])
def get_model_benchmarks():
    metrics_path = Path("models/artifacts/benchmark_metrics.json")
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

assistant_service = InventoryAIAssistant()


class ChatRequest(BaseModel):
    question: str


@app.post("/api/v1/assistant/chat", tags=["AI Assistant"])
def ask_inventory_assistant(payload: ChatRequest, db: Session = Depends(get_db)):
    reply = assistant_service.answer_question(payload.question, db)
    return {"question": payload.question, "answer": reply}
