# Rajasthan HeatWatch AI

FastAPI + scikit-learn backend, React (Vite) frontend. Forecasts come from Open-Meteo; ML predictions come from a model trained on `Rajasthan_Heatwave_2006_2025.csv`.

## Run locally
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py                  # writes model/models.joblib and model/metrics.json (already included)
uvicorn app:app --port 8000      # needs internet for Open-Meteo
# second terminal
cd frontend && npm install && npm run dev     # http://localhost:5173 (proxies /api to :8000)
```
Environment variables: see `backend/.env.example`. API keys stay in the backend; the browser never sees them.

## Deploy (single service)
`cd frontend && npm install && npm run build`, then run `uvicorn app:app --host 0.0.0.0 --port $PORT` from `backend/`. The backend serves `frontend/dist` automatically. Set `ALLOWED_ORIGINS` if the frontend is hosted separately.

## API
`GET /api/forecast/tomorrow?district=` · `/api/forecast/week?district=` · `/api/predict?district=&date=` · `/api/risk?tmax=` · `/api/map?day=` · `/api/alerts` · `POST /api/simulate` · `/api/analytics` · `/api/model` · `/api/districts` · `/api/health`

## What the data allows (read this before your demo)
- Dataset: 21,960 daily rows, 9 districts (Barmer, Bikaner, Churu, Jaipur, Jaisalmer, Jodhpur, Kota, Nagaur, Sri Ganganagar), **1 Mar-30 Jun only**. Udaipur and Ajmer are not in the data, so they are not supported.
- **Out of season, the ML model refuses to predict** (e.g. in October); only the external forecast is shown. Test the ML panels between March and June.
- `HEATWAVE` = TMAX >= 43.18 C (the dataset-wide 95th percentile), not an IMD definition. Project risk levels: <40 Low, 40-43.18 Moderate, 43.18-45 High, >=45 Very high. These are not IMD warnings.
- `TMIN` is not a true overnight minimum (TMAX-TMIN averages 0.45 K), so no minimum-temperature model exists; minimum temperature comes only from Open-Meteo.
- Model inputs from the live API are mapped approximately onto reanalysis-style training variables; the reported test errors do not measure that mismatch.
- Alerts are displayed and logged to `alerts_history.json`; email/SMS/push is not implemented.
- Map shows grid-point markers; district boundary polygons are not included.

## Test-set results (2023-2025, trained 2006-2019, validated 2020-2022)
| Days ahead | Model MAE (C) | Persistence MAE | Climatology MAE |
|---|---|---|---|
| 1 | 1.19 | 1.33 | 2.78 |
| 3 | 2.12 | 2.50 | 2.75 |
| 7 | 2.66 | 3.43 | 2.76 |
