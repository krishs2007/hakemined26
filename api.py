#!/usr/bin/env python3
"""
SmartContainer Risk Engine - FastAPI REST API
Optional REST endpoint for programmatic access.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="SmartContainer Risk Engine API",
    description="REST API for container risk scoring",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "service": "SmartContainer Risk Engine",
        "version": "1.0.0",
        "endpoints": [
            "/run-pipeline",
            "/predictions",
            "/predictions/{container_id}",
            "/dashboard",
            "/report",
            "/health",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run-pipeline")
def run_pipeline_endpoint():
    """Run the full risk scoring pipeline."""
    try:
        from pipeline import run_pipeline
        results = run_pipeline()
        total = len(results)
        critical = int((results['Risk_Level'] == 'Critical').sum())
        medium = int((results['Risk_Level'] == 'Medium Risk').sum())
        low = int((results['Risk_Level'] == 'Low Risk').sum())
        return {
            "status": "success",
            "total_containers": total,
            "critical": critical,
            "medium_risk": medium,
            "low_risk": low,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions")
def get_predictions(risk_level: str = None, limit: int = 100):
    """Get predictions, optionally filtered by risk level."""
    pred_path = 'outputs/predictions.csv'
    if not os.path.exists(pred_path):
        raise HTTPException(status_code=404, detail="No predictions found. Run pipeline first via POST /run-pipeline")

    df = pd.read_csv(pred_path)
    if risk_level:
        df = df[df['Risk_Level'] == risk_level]
    df = df.head(limit)
    return df.to_dict(orient='records')


@app.get("/predictions/{container_id}")
def get_prediction_by_id(container_id: str):
    """Get prediction for a specific container."""
    pred_path = 'outputs/predictions.csv'
    if not os.path.exists(pred_path):
        raise HTTPException(status_code=404, detail="No predictions found. Run pipeline first.")

    df = pd.read_csv(pred_path)
    match = df[df['Container_ID'] == container_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="Container not found: " + container_id)
    return match.iloc[0].to_dict()


@app.get("/dashboard")
def get_dashboard():
    """Get the dashboard image."""
    path = 'outputs/summary_report.png'
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No dashboard found. Run pipeline first.")
    return FileResponse(path, media_type='image/png')


@app.get("/report")
def get_report():
    """Get the HTML summary report."""
    path = 'outputs/summary_report.html'
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No HTML report found. Run pipeline first.")
    return FileResponse(path, media_type='text/html')
