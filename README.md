# SmartContainer Risk Engine

AI-powered container risk scoring engine for customs inspection. Combines supervised machine learning (Random Forest), unsupervised anomaly detection (Isolation Forest), and rule-based flags to produce a 0-100 risk score for each container.

## Project Structure

```
├── src/
│   ├── preprocessor.py        # Data loading + leakage-safe train/test split
│   ├── feature_engineer.py    # 26 features (direct + historical)
│   ├── anomaly_detector.py    # Isolation Forest unsupervised layer
│   ├── model_trainer.py       # Random Forest classifier
│   ├── risk_scorer.py         # Composite risk score: model + anomaly + rules
│   ├── explainer.py           # 1-2 line plain English explanations
│   └── dashboard.py           # 6-panel matplotlib dashboard
├── models/                    # Saved trained models (auto-created)
├── outputs/                   # Predictions CSV + dashboard PNG (auto-created)
├── pipeline.py                # End-to-end orchestrator
├── app.py                     # Streamlit interactive dashboard
├── api.py                     # FastAPI REST API
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Setup

```bash
git clone https://github.com/krishs2007/hakemined26.git
cd hakemined26
git checkout smartcontainer-risk-engine
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Files

The engine auto-detects CSV files in two locations:
- Root level: Historical Data.csv and Real-Time Data.csv
- data/ folder: data/Historical_Data.csv and data/Real-Time_Data.csv

## Run the Pipeline

```bash
python pipeline.py
```

Steps: Load data, engineer 26 features, train Isolation Forest, train Random Forest, compute risk scores (0-100), generate explanations, save predictions CSV, generate dashboard PNG.

## Run the Streamlit Dashboard

```bash
streamlit run app.py
```

## Run the FastAPI Server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints: POST /run-pipeline, GET /predictions, GET /predictions/{container_id}, GET /dashboard, GET /health

## Risk Levels

| Score Range | Level | Action |
|------------|-------|--------|
| 0-30 | Low Risk | Routine processing |
| 31-60 | Medium Risk | Targeted review |
| 61-100 | Critical | Mandatory inspection |

## Leakage Prevention

Training set = Historical data MINUS all real-time Container IDs. Test set = Real-time data only. Historical features are computed from training data only. Anomaly normalization parameters are fitted on training scores and reused for test/inference.

## Technical Details

- 26 engineered features covering weight discrepancy, value, dwell time, temporal, regime, and historical behavioral signals
- Random Forest with class_weight={0:1, 1:2, 2:50} to handle extreme class imbalance
- Isolation Forest anomaly layer catches multivariate outliers
- Composite scoring: 70% model + 20% anomaly + 10% rule-based flags
- Global random seed (42) for full reproducibility