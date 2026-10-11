"""Train and evaluate next-N-day TMAX models + calibrated heatwave classifiers.
Chronological split: train 2006-2019, validation 2020-2022, test 2023-2025.
Run: python train.py"""
import json, datetime as dt, joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGR, HistGradientBoostingClassifier as HGC
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error, roc_auc_score, brier_score_loss
K = 273.15; HORIZONS = range(1, 8); VERSION = "hgb-1.0"
FEATS = ["tmax","tmax_l1","tmax_l2","tmax_m3","t2m","dew","wspd","mslp","cloud","doy","lat","lon","elev"]
def load(path="data/Rajasthan_Heatwave_2006_2025.csv"):
    d = pd.read_csv(path)
    need = {"YEAR","MONTH","DAY","TMAX","TEMP2M","DEW2M","WIND_U10","WIND_V10","MSLP","CLOUD","HEATWAVE","LAT","LON","GEOP","DISTRICT"}
    if need - set(d.columns): raise ValueError(f"Missing columns: {need - set(d.columns)}")
    d = d.dropna(subset=list(need)).copy()
    d["date"] = pd.to_datetime(dict(year=d.YEAR, month=d.MONTH, day=d.DAY))
    return d.sort_values(["DISTRICT","date"]).reset_index(drop=True)
def build(d):
    g = d.groupby("DISTRICT"); f = pd.DataFrame(index=d.index)
    f["tmax"] = d.TMAX-K; f["tmax_l1"] = g.TMAX.shift(1)-K; f["tmax_l2"] = g.TMAX.shift(2)-K
    f["tmax_m3"] = (g.TMAX.rolling(3).mean().reset_index(level=0, drop=True))-K
    f["t2m"] = d.TEMP2M-K; f["dew"] = d.DEW2M-K; f["wspd"] = np.hypot(d.WIND_U10, d.WIND_V10)
    f["mslp"] = d.MSLP/100; f["cloud"] = d.CLOUD; f["doy"] = d.date.dt.dayofyear
    f["lat"] = d.LAT; f["lon"] = d.LON; f["elev"] = d.GEOP/9.80665  # geopotential -> metres
    # lags must not cross a season gap (data covers Mar-Jun only)
    gap = g.date.diff().dt.days.fillna(99) != 1
    f.loc[gap | g.date.diff(2).dt.days.ne(2), ["tmax_l1","tmax_l2","tmax_m3"]] = np.nan
    return f
def ece(y, p, bins=10):
    b = np.minimum((p*bins).astype(int), bins-1); e = 0
    for i in range(bins):
        m = b == i
        if m.any(): e += m.mean()*abs(y[m].mean()-p[m].mean())
    return float(e)
if __name__ == "__main__":
    d = load(); F = build(d); g = d.groupby("DISTRICT"); doy = F.doy
    split = lambda y: np.where(y <= 2019, "train", np.where(y <= 2022, "val", "test"))
    out = {"version": VERSION, "trained_at": dt.datetime.utcnow().isoformat()+"Z", "features": FEATS,
           "split": {"train": "2006-2019", "validation": "2020-2022", "test": "2023-2025"}, "horizons": {}}
    models = {}; sample = None
    clim_tr = (d.assign(t=d.TMAX-K, doy=doy)[d.YEAR <= 2019].groupby(["DISTRICT","doy"]).t.mean())
    for h in HORIZONS:
        y = g.TMAX.shift(-h)-K; yh = g.HEATWAVE.shift(-h)
        ok = F[FEATS].notna().all(axis=1) & y.notna() & (g.date.shift(-h)-d.date).dt.days.eq(h)
        sp = pd.Series(split(d.YEAR), index=d.index)
        tr, va, te = [ok & (sp == s) for s in ("train","val","test")]
        reg = HGR(max_iter=300, learning_rate=.05, max_depth=5, early_stopping=False, random_state=0).fit(F.loc[tr, FEATS], y[tr])
        res = (y[va]-reg.predict(F.loc[va, FEATS]))
        pt = reg.predict(F.loc[te, FEATS]); yt = y[te]
        tdoy = (doy[te]+h).clip(upper=181); clim = np.array([clim_tr.get((a,b), np.nan) for a,b in zip(d.DISTRICT[te], tdoy)])
        pers = F.tmax[te]; rm = lambda a,b: float(np.sqrt(mean_squared_error(a,b)))
        clf = HGC(max_iter=200, learning_rate=.05, max_depth=4, random_state=0).fit(F.loc[tr, FEATS], yh[tr])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(clf.predict_proba(F.loc[va, FEATS])[:,1], yh[va])
        pr = iso.predict(clf.predict_proba(F.loc[te, FEATS])[:,1]); base = yh[tr].mean()
        brier, brier_c = brier_score_loss(yh[te], pr), brier_score_loss(yh[te], np.full(len(pr), base))
        out["horizons"][h] = {"n_test": int(te.sum()), "mae_model": float(mae(yt,pt)), "rmse_model": rm(yt,pt),
            "mae_persistence": float(mae(yt,pers)), "rmse_persistence": rm(yt,pers),
            "mae_climatology": float(np.nanmean(abs(yt-clim))), "rmse_climatology": float(np.sqrt(np.nanmean((yt-clim)**2))),
            "resid_std_val": float(res.std()), "resid_q05": float(res.quantile(.05)), "resid_q95": float(res.quantile(.95)),
            "clf_auc": float(roc_auc_score(yh[te], pr)), "clf_brier": float(brier), "clf_brier_climatology": float(brier_c), "clf_ece": ece(yh[te].values, pr),
            "clf_calibrated_useful": bool(brier < brier_c*0.9)}
        models[h] = {"reg": reg, "clf": clf, "iso": iso, "q05": out["horizons"][h]["resid_q05"], "q95": out["horizons"][h]["resid_q95"]}
        if h == 1:
            s = pd.DataFrame({"date": d.date[te].dt.strftime("%Y-%m-%d"), "district": d.DISTRICT[te], "actual": yt.round(2), "predicted": pt.round(2)})
            out["actual_vs_predicted_h1"] = s[s.district.isin(["Jaipur","Jaisalmer","Bikaner"])].to_dict("records")
        print(h, {k: round(v,3) for k,v in out["horizons"][h].items() if isinstance(v,float)})
    out["clim_by_district_doy"] = {f"{a}|{b}": round(v,2) for (a,b), v in clim_tr.items()}
    joblib.dump(models, "model/models.joblib"); json.dump(out, open("model/metrics.json","w"))
