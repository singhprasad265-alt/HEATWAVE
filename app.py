"""RAJASTHAN HEATWATCH AI - FastAPI backend. Run: uvicorn app:app --port 8000"""
import os, json, time, threading, datetime as dt, joblib, httpx, numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from train import load, build, FEATS, K

OM_URL = os.getenv("FORECAST_API_URL", "https://api.open-meteo.com/v1/forecast")
OM_KEY = os.getenv("OPEN_METEO_API_KEY")  # optional (commercial plan); never sent to the browser
TTL = int(os.getenv("CACHE_TTL_SECONDS", "1800")); HIST = os.path.join(os.path.dirname(__file__), "alerts_history.json")
app = FastAPI(title="Rajasthan HeatWatch AI")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(","), allow_methods=["*"], allow_headers=["*"])
D = load(); MODELS = joblib.load("model/models.joblib"); MET = json.load(open("model/metrics.json"))
LOC = {r.DISTRICT: dict(lat=float(r.LAT), lon=float(r.LON), elev=float(r.GEOP)/9.80665) for r in D.drop_duplicates("DISTRICT").itertuples()}
DATASET_HW_C = float(D.loc[D.HEATWAVE == 1, "TMAX"].min()-K)  # 43.18: label is TMAX above the dataset's 95th percentile
CACHE, LOCK = {}, threading.Lock()
LEVELS = [("Low", "#3d8b4f"), ("Moderate", "#e0b020"), ("High", "#e8801c"), ("Very high", "#c62828")]

def risk_level(tmax):
    """Project thresholds (NOT IMD warnings): <40 Low; 40-<43.18 Moderate (IMD plains heat threshold 40C);
    43.18-<45 High (dataset HEATWAVE label cut); >=45 Very high (IMD absolute severe-heatwave temperature)."""
    return 0 if tmax < 40 else 1 if tmax < DATASET_HW_C else 2 if tmax < 45 else 3
def risk_obj(i, basis): return {"level": LEVELS[i][0], "color": LEVELS[i][1], "index": i, "basis": basis}
def cached(key, fn):
    with LOCK:
        if key in CACHE and time.time()-CACHE[key][0] < TTL: return CACHE[key][1]
    v = fn()
    with LOCK: CACHE[key] = (time.time(), v)
    return v
def need(name):
    if name not in LOC: raise HTTPException(404, f"Unsupported district '{name}'. Supported: {sorted(LOC)}")
    return LOC[name]

def fetch_external(name):
    p = LOC[name]
    def go():
        q = dict(latitude=p["lat"], longitude=p["lon"], timezone="Asia/Kolkata", past_days=3, forecast_days=8, wind_speed_unit="ms",
                 hourly="temperature_2m,relative_humidity_2m,dew_point_2m,wind_speed_10m,pressure_msl,cloud_cover,weather_code",
                 daily="temperature_2m_max,temperature_2m_min,weather_code")
        if OM_KEY: q["apikey"] = OM_KEY
        try: r = httpx.get(OM_URL, params=q, timeout=15); r.raise_for_status(); j = r.json()
        except Exception as e: raise HTTPException(503, f"External forecast unavailable ({type(e).__name__}). No forecast is shown rather than an invented one.")
        j["_retrieved"] = dt.datetime.utcnow().isoformat()+"Z"; return j
    return cached("om:"+name, go)

def hourly_df(j):
    h = pd.DataFrame(j["hourly"]); h["time"] = pd.to_datetime(h.time); h["date"] = h.time.dt.strftime("%Y-%m-%d"); return h
def day_features(h, date, name):
    """Map external hourly data for `date` and 2 prior days to the training features (approximate: ERA5 vs forecast model)."""
    def dm(day): x = h[h.date == day]; return x if len(x) >= 20 else None
    ds = [(pd.Timestamp(date)-pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(3)]
    X = [dm(x) for x in ds]
    if any(x is None for x in X): return None
    t = [x.temperature_2m.max() for x in X]; p = LOC[name]
    return dict(tmax=t[0], tmax_l1=t[1], tmax_l2=t[2], tmax_m3=float(np.mean(t)), t2m=X[0].temperature_2m.mean(), dew=X[0].dew_point_2m.mean(),
        wspd=X[0].wind_speed_10m.mean(), mslp=X[0].pressure_msl.mean(), cloud=X[0].cloud_cover.mean()/100, doy=pd.Timestamp(date).dayofyear, lat=p["lat"], lon=p["lon"], elev=p["elev"])

def ml_predict(feat, h, target_date):
    td = pd.Timestamp(target_date)
    if not (60 <= td.dayofyear <= 182):  # Mar 1 - Jun 30: the only period in the training data
        return {"available": False, "reason": "Out of season: the model was trained only on 1 March - 30 June data, so it makes no prediction for this date."}
    m = MODELS[h]; x = pd.DataFrame([feat])[FEATS]; mu = float(m["reg"].predict(x)[0]); me = MET["horizons"][str(h)]
    out = {"available": True, "tmax_c": round(mu, 1), "interval_90_c": [round(mu+m["q05"], 1), round(mu+m["q95"], 1)], "horizon_days": h,
           "model_version": MET["version"], "test_mae_c": round(me["mae_model"], 2)}
    if me["clf_calibrated_useful"]: out["heatwave_probability"] = round(float(m["iso"].predict(m["clf"].predict_proba(x)[:, 1])[0]), 3); out["probability_note"] = f"Dataset-defined heatwave (TMAX >= {DATASET_HW_C:.2f} C); isotonic-calibrated, test Brier {me['clf_brier']:.3f} vs {me['clf_brier_climatology']:.3f} for climatology."
    else: out["heatwave_probability"] = None; out["probability_note"] = "Probability withheld: at this horizon it is not meaningfully better than climatology."
    return out

def clim(name, date):
    d = pd.Timestamp(date).dayofyear; v = [MET["clim_by_district_doy"].get(f"{name}|{k}") for k in range(d-3, d+4)]; v = [x for x in v if x is not None]
    return round(float(np.mean(v)), 1) if v else None

def day_report(name, date, j, h_ahead):
    h = hourly_df(j); dly = j["daily"]; i = dly["time"].index(date) if date in dly["time"] else None
    if i is None: raise HTTPException(404, f"No external forecast for {date}")
    hh = h[h.date == date]; ext = {"tmax_c": dly["temperature_2m_max"][i], "tmin_c": dly["temperature_2m_min"][i], "weather_code": dly["weather_code"][i],
        "hottest_hour": hh.loc[hh.temperature_2m.idxmax(), "time"].strftime("%H:%M") if len(hh) else None,
        "humidity_pct_mean": round(float(hh.relative_humidity_2m.mean())) if len(hh) else None, "wind_ms_mean": round(float(hh.wind_speed_10m.mean()), 1) if len(hh) else None,
        "hourly": [{"t": r.time.strftime("%H:%M"), "temp_c": r.temperature_2m} for r in hh.itertuples()], "source": "Open-Meteo forecast API", "retrieved_utc": j["_retrieved"]}
    ml = {"available": False, "reason": "Not computed"}
    base = (pd.Timestamp(date)-pd.Timedelta(days=h_ahead)).strftime("%Y-%m-%d"); f = day_features(h, base, name)
    if f is None: ml = {"available": False, "reason": "Insufficient recent hourly data to build model inputs."}
    else: ml = ml_predict(f, h_ahead, date)
    ei = risk_level(ext["tmax_c"]); mi = risk_level(ml["tmax_c"]) if ml["available"] else -1; fi = max(ei, mi)
    final = risk_obj(fi, f"Highest of external forecast tmax ({ext['tmax_c']} C -> {LEVELS[ei][0]})" + (f" and ML prediction ({ml['tmax_c']} C -> {LEVELS[mi][0]})." if mi >= 0 else "; ML prediction unavailable."))
    return {"district": name, "date": date, "external_forecast": ext, "ml_prediction": ml, "risk_assessment": final, "climatology_tmax_c": clim(name, date),
            "climatology_note": "Dataset TMAX mean for nearby days of year, 2006-2019 (reanalysis-style grid values; not directly comparable to station maxima).",
            "official_warning": "Not an IMD warning. Check mausam.imd.gov.in for official alerts.", "explanation": explain(ext, ml, final)}
def explain(ext, ml, final):
    s = f"The external forecast expects a high of {ext['tmax_c']} C" + (f" around {ext['hottest_hour']}" if ext["hottest_hour"] else "") + "."
    if ml["available"]: s += f" The ML model, using recent temperature, dew point, wind, pressure and cloud, predicts {ml['tmax_c']} C (90% interval {ml['interval_90_c'][0]}-{ml['interval_90_c'][1]} C)."
    else: s += " " + ml["reason"]
    return s + f" Risk is '{final['level']}' under the project thresholds."
PRECAUTION = ["Normal precautions; stay hydrated.", "Limit outdoor work 12:00-16:00; drink water regularly.", "Avoid midday exposure; check on elderly people and children.", "Stay indoors at midday; seek medical help for heat-stroke symptoms."]
def today(): return dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d")

@app.get("/api/health")
def health(): return {"ok": True, "model_version": MET["version"], "cache_ttl_s": TTL}
@app.get("/api/districts")
def districts(): return {"districts": [{"name": k, **v} for k, v in sorted(LOC.items())], "note": "Only the 9 districts present in the training dataset are supported."}
@app.get("/api/forecast/tomorrow")
def tomorrow(district: str):
    need(district); j = fetch_external(district); t = (pd.Timestamp(today())+pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    r = day_report(district, t, j, 1); r["precaution"] = PRECAUTION[r["risk_assessment"]["index"]]; r["generated_utc"] = dt.datetime.utcnow().isoformat()+"Z"; return r
@app.get("/api/forecast/week")
def week(district: str):
    need(district); j = fetch_external(district); t0 = pd.Timestamp(today()); days = []
    for k in range(1, 8):
        r = day_report(district, (t0+pd.Timedelta(days=k)).strftime("%Y-%m-%d"), j, k); r["precaution"] = PRECAUTION[r["risk_assessment"]["index"]]; r["weekday"] = (t0+pd.Timedelta(days=k)).strftime("%A"); days.append(r)
    return {"district": district, "days": days, "retrieved_utc": j["_retrieved"]}
@app.get("/api/predict")
def predict(district: str, date: str):
    need(district); j = fetch_external(district)
    try: n = (pd.Timestamp(date)-pd.Timestamp(today())).days
    except Exception: raise HTTPException(400, "date must be YYYY-MM-DD")
    if not 1 <= n <= 7: raise HTTPException(400, "date must be 1-7 days after today")
    return day_report(district, date, j, n)
@app.get("/api/risk")
def risk(tmax: float):
    if not -10 <= tmax <= 60: raise HTTPException(400, "tmax out of plausible range")
    i = risk_level(tmax); return {**risk_obj(i, "Project thresholds: <40, 40-43.18, 43.18-45, >=45 C"), "precaution": PRECAUTION[i], "disclaimer": "Project estimate, not an IMD warning."}
@app.get("/api/map")
def rmap(day: int = 1):
    if not 1 <= day <= 7: raise HTTPException(400, "day must be 1-7")
    out = []
    for n in LOC:
        try: r = week(n)["days"][day-1]; out.append({"district": n, **LOC[n], "date": r["date"], "tmax_c": r["external_forecast"]["tmax_c"], "ml_tmax_c": r["ml_prediction"].get("tmax_c"), "risk": r["risk_assessment"]})
        except HTTPException as e: out.append({"district": n, **LOC[n], "error": e.detail})
    return {"markers": out, "note": "City-level (grid point) forecasts, not district-wide averages."}
@app.get("/api/alerts")
def alerts():
    new = []
    for n in LOC:
        try: w = week(n)
        except HTTPException: continue
        for r in w["days"]:
            i = r["risk_assessment"]["index"]
            if i >= 2: new.append({"district": n, "date": r["date"], "tmax_c": max(r["external_forecast"]["tmax_c"], r["ml_prediction"].get("tmax_c") or -99), "risk": r["risk_assessment"]["level"], "reason": r["risk_assessment"]["basis"], "precaution": r["precaution"], "source": "Open-Meteo + HeatWatch ML", "updated_utc": w["retrieved_utc"], "delivered": False})
    hist = json.load(open(HIST)) if os.path.exists(HIST) else []; keys = {(a["district"], a["date"], a["risk"]) for a in hist}
    hist += [a for a in new if (a["district"], a["date"], a["risk"]) not in keys]; json.dump(hist, open(HIST, "w"))
    return {"active": new, "history": hist[-200:], "notifications": "Not implemented: nothing is emailed, texted or pushed.", "disclaimer": "Project warnings are not a substitute for official IMD alerts."}
class Sim(BaseModel):
    district: str; month: int = 5; day: int = 15; temp_delta_c: float = 0; dew_delta_c: float = 0
@app.post("/api/simulate")
def simulate(s: Sim):
    need(s.district)
    if not (-5 <= s.temp_delta_c <= 5 and -10 <= s.dew_delta_c <= 10 and 3 <= s.month <= 6): raise HTTPException(400, "Inputs outside range supported by the model (months 3-6, temp +/-5 C, dew +/-10 C).")
    F = build(D); doy = pd.Timestamp(2023, s.month, s.day).dayofyear; m = (D.DISTRICT == s.district) & (F.doy.between(doy-3, doy+3)) & F[FEATS].notna().all(axis=1)
    base = F.loc[m, FEATS].median(); mod = base.copy()
    for k in ("tmax", "tmax_l1", "tmax_l2", "tmax_m3", "t2m"): mod[k] += s.temp_delta_c
    mod["dew"] += s.dew_delta_c; mo = MODELS[1]; f = lambda r: ml_predict(r.to_dict(), 1, f"2023-{s.month:02d}-{s.day:02d}")
    a, b = f(base), f(mod); lv = lambda r: LEVELS[risk_level(r["tmax_c"])][0]
    return {"label": "HYPOTHETICAL SIMULATION - not a weather forecast", "baseline": {**a, "risk": lv(a), "inputs_tmax_c": round(base.tmax, 1)}, "scenario": {**b, "risk": lv(b), "inputs_tmax_c": round(mod.tmax, 1)},
            "basis": "Median historical conditions for this district within +/-3 days of the date; the model's next-day prediction is recomputed after shifting inputs."}
@app.get("/api/model")
def model(): return {k: v for k, v in MET.items() if k != "clim_by_district_doy"}
@app.get("/api/analytics")
def analytics():
    d = D.assign(t=D.TMAX-K, y=D.YEAR)
    return {"rows": len(D), "period": [int(D.YEAR.min()), int(D.YEAR.max())], "heatwave_label_threshold_c": round(DATASET_HW_C, 2),
        "yearly": d.groupby("y").agg(avg_tmax=("t", "mean"), heatwave_days=("HEATWAVE", "sum")).round(2).reset_index().to_dict("records"),
        "by_district": d.groupby("DISTRICT").agg(avg_tmax=("t", "mean"), max_tmax=("t", "max"), heatwave_days=("HEATWAVE", "sum")).round(2).reset_index().to_dict("records"),
        "by_month": d.groupby("MONTH").agg(avg_tmax=("t", "mean"), heatwave_days=("HEATWAVE", "sum")).round(2).reset_index().to_dict("records"),
        "missing_values": {c: int(D[c].isna().sum()) for c in D.columns if c != "date"},
        "data_quality_notes": ["Covers 1 Mar-30 Jun only.", "TMAX-TMIN averages 0.45 K: TMIN is not an overnight minimum.", "HEATWAVE equals TMAX above one fixed dataset-wide 95th percentile; it is not an IMD criterion."]}
dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(dist): app.mount("/", StaticFiles(directory=dist, html=True), name="web")
