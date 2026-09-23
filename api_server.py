"""
SPnPOP - Forecast API Server
===========================================================

Exposes the training/prediction pipeline over HTTP so the dashboard can
pull live forecasts without running any Python scripts by hand.

Endpoints:
    POST /train/{client_id}?grain=item&n_iters=20
        Kicks off training in the background (this can take several
        minutes - tuning runs up to 20 x 7 = 140 individual model fits).
        Returns immediately with a status token; poll the status endpoint.

    GET  /train/{client_id}/status
        Check on a training run: "idle" | "running" | "done" | "error",
        plus a summary of test metrics per horizon once done.

    GET  /forecast/{client_id}?grain=item
        Pulls FRESH live data from the ERP right now, loads the
        already-trained models for this client, and returns the Day+1..7
        forecast as JSON - this is what the dashboard should call.

    GET  /health
        Basic liveness check.

Requires: pip install fastapi uvicorn

Run:
    uvicorn api_server:app --host 0.0.0.0 --port 8000

IMPORTANT - credentials: this reads the ERP username/password from
environment variables (SPNPOP_USER / SPNPOP_PASSWORD / SPNPOP_CCD) instead
of hardcoding them, because this file is now a server other machines will
call, not a one-off local script. Set them before running, e.g. (PowerShell):
    $env:SPNPOP_USER = "Abhiraami.BL"
    $env:SPNPOP_PASSWORD = "..."
    $env:SPNPOP_CCD = "dc"
"""

import os
import threading
import time
import traceback
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from erp_client import fetch_live_payload
from model import run_training_job, predict_next_horizons, get_customer_forecast

app = FastAPI(title="SPnPOP Forecast API")

# Allow the dashboard (likely a different origin) to call this API directly
# from the browser. Tighten allow_origins to the dashboard's actual domain
# once you know it, instead of "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory training status per client. Fine for a single-process server;
# swap for a real job queue (Celery, RQ) if you need multiple workers.
TRAINING_STATUS: dict = {}


# ---------------------------------------------------------------------------
# Background training worker
# ---------------------------------------------------------------------------

def _run_training_background(client_id: str, grain: str, n_iters: int):
    TRAINING_STATUS[client_id] = {
        "status": "running",
        "grain": grain,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        payload = fetch_live_payload()
        metadata = run_training_job(payload, client_id, grain=grain, n_iters=n_iters)

        # Trim to a dashboard-friendly summary instead of the full metadata
        summary = {
            h: {
                "test_WAPE": info["test_metrics"]["WAPE"],
                "test_Bias_pct": info["test_metrics"]["Bias_pct"],
                "test_MAE": info["test_metrics"]["MAE"],
            }
            for h, info in metadata["horizons"].items()
        }

        TRAINING_STATUS[client_id] = {
            "status": "done",
            "grain": grain,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "n_entities_trained": metadata.get("n_entities_trained"),
            "n_entities_dropped": len(metadata.get("dropped_entities", [])),
            "test_metrics_by_horizon": summary,
        }
    except Exception as exc:
        TRAINING_STATUS[client_id] = {
            "status": "error",
            "grain": grain,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train/{client_id}")
def start_training(client_id: str, grain: str = "item", n_iters: int = 20):
    if grain not in ("item", "category"):
        raise HTTPException(400, "grain must be 'item' or 'category'")

    current = TRAINING_STATUS.get(client_id)
    if current and current.get("status") == "running":
        raise HTTPException(409, f"Training already running for {client_id}")

    thread = threading.Thread(
        target=_run_training_background,
        args=(client_id, grain, n_iters),
        daemon=True,
    )
    thread.start()
    return {"message": f"Training started for {client_id}", "grain": grain, "n_iters": n_iters}


@app.get("/train/{client_id}/status")
def training_status(client_id: str):
    status = TRAINING_STATUS.get(client_id)
    if status is None:
        raise HTTPException(404, f"No training run found for {client_id}. Call POST /train/{client_id} first.")
    return status


@app.get("/forecast/{client_id}")
def get_forecast(client_id: str, grain: str = "item", recency_days: int = 90):
    if grain not in ("item", "category"):
        raise HTTPException(400, "grain must be 'item' or 'category'")

    output_dir = os.path.join("SPnPOP_outputs", client_id)
    if not os.path.exists(os.path.join(output_dir, "model_day1.json")):
        raise HTTPException(
            404,
            f"No trained model found for {client_id} (grain={grain}). "
            f"Call POST /train/{client_id}?grain={grain} first.",
        )

    start = time.time()
    try:
        payload = fetch_live_payload()
        forecast_df = predict_next_horizons(payload, client_id, grain=grain, recency_days=recency_days)
    except Exception as exc:
        print(f"[/forecast/{client_id}] ERROR:\n{traceback.format_exc()}")
        raise HTTPException(500, f"Forecast failed: {exc}")

    # JSON-friendly: convert dates to ISO strings
    forecast_df = forecast_df.copy()
    for col in ("AsOfDate", "ForecastDate"):
        forecast_df[col] = forecast_df[col].astype(str)

    return {
        "recordCount": len(forecast_df),
        "request": f"/forecast/{client_id}",
        "elapsed": round(time.time() - start, 3),
        "dataSet": forecast_df.to_dict(orient="records"),
        "keyOrder": [],
    }


@app.get("/forecast/{client_id}/by-customer")
def get_forecast_by_customer(client_id: str, grain: str = "item", recency_days: int = 90,
                              discount_weight: float = 0.0):
    """Same Day+1..7 forecast as /forecast, but split across customers using
    each customer's historical share of that item's/category's demand
    (optionally nudged by AvgDisPer via discount_weight)."""
    if grain not in ("item", "category"):
        raise HTTPException(400, "grain must be 'item' or 'category'")

    output_dir = os.path.join("SPnPOP_outputs", client_id)
    if not os.path.exists(os.path.join(output_dir, "model_day1.json")):
        raise HTTPException(
            404,
            f"No trained model found for {client_id} (grain={grain}). "
            f"Call POST /train/{client_id}?grain={grain} first.",
        )

    start = time.time()
    try:
        payload = fetch_live_payload()
        forecast_df = get_customer_forecast(
            payload, client_id, grain=grain,
            recency_days=recency_days, discount_weight=discount_weight,
        )
    except Exception as exc:
        print(f"[/forecast/{client_id}/by-customer] ERROR:\n{traceback.format_exc()}")
        raise HTTPException(500, f"Forecast failed: {exc}")

    forecast_df = forecast_df.copy()
    for col in ("AsOfDate", "ForecastDate"):
        forecast_df[col] = forecast_df[col].astype(str)

    return {
        "recordCount": len(forecast_df),
        "request": f"/forecast/{client_id}/by-customer",
        "elapsed": round(time.time() - start, 3),
        "dataSet": forecast_df.to_dict(orient="records"),
        "keyOrder": [],
    }