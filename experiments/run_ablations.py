"""
Ablation : mesure l'apport de chaque bloc de features sur la prévision du prix
day-ahead français, en séparant deux usages :

  - "reconstruction" : on autorise les prix DA des pays voisins. Ils sont fixés
    en même temps que le prix FR (couplage EUPHEMIA), donc inconnus la veille ->
    borne haute, pas une vraie prévision.
  - "forecast_*"     : uniquement ce qui est connu avant la clôture du day-ahead
    (J-1, 12h) : météo, prévisions ENTSO-E, gaz J-2, capacités, calendrier, lags.

Des benchmarks naïfs (prix de la veille / de la semaine dernière au même créneau)
sont évalués sur les mêmes plis : un modèle ne vaut que par l'écart à ces baselines.

Lancer (depuis n'importe où) :
    python -u experiments/run_ablations.py

Sorties (experiments/results/) :
    ablation_<stamp>.csv           un résumé par (modèle, config)
    folds_<stamp>.csv              les métriques par pli
    last_fold_<model>_<cfg>.parquet prédictions du dernier pli (pour les graphes)
    importance_<model>_<cfg>.csv   importance LightGBM (gain)
Les modèles entraînés sont mis en cache dans experiments/_runs/ (non versionné).
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")  # logs lisibles en direct

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from src.data.loader import load_data
import src.models.pricing_from_meteo as pm
from src.models.pricing_from_meteo import model_learn


# ----------------------------------------------------------------------
# Paramètres
# ----------------------------------------------------------------------
START = "2023-01-01"
END = "2026-05-01"
FREQ = "1h"
N_SPLITS = 5
MODEL_TYPES = ["LightGBM", "Simple"]
METEO_MODE = "department"

RUNS_DIR = ROOT / "experiments" / "_runs"
OUT_DIR = ROOT / "experiments" / "results"

CONFIGS = [
    dict(name="reconstruction", neighbours=True,  gas=True,  capacity=True),
    dict(name="forecast_full",  neighbours=False, gas=True,  capacity=True),
    dict(name="forecast_nocap", neighbours=False, gas=True,  capacity=False),
    dict(name="forecast_base",  neighbours=False, gas=False, capacity=False),
]


def _apply_config(data: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    drop = []
    if not cfg["gas"]:
        drop.append("gas_price")
    if not cfg["capacity"]:
        drop += ["solar_capacity", "wind_capacity"]
    return data.drop(columns=drop, errors="ignore")


def _naive_benchmarks(data: pd.DataFrame) -> list[dict]:
    # Prix au même créneau J-1 / J-7, évalué sur exactement les mêmes plis que les
    # modèles (même préparation, même TimeSeriesSplit).
    df = pm._prepare_features(data, drop_missing_price=True, meteo_mode="national")
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    rows = []
    for lag_col, name in (("price_lag_1d", "naive_D-1"), ("price_lag_7d", "naive_D-7")):
        folds = []
        for _, test_idx in tscv.split(df):
            te = df.iloc[test_idx].dropna(subset=[lag_col])
            y, p = te["price"].values, te[lag_col].values
            folds.append(dict(
                r2=r2_score(y, p), mae=mean_absolute_error(y, p),
                rmse=float(np.sqrt(mean_squared_error(y, p))),
                test_start=te["time"].iloc[0].isoformat(),
                test_end=te["time"].iloc[-1].isoformat(),
            ))
        f = pd.DataFrame(folds)
        rows.append(dict(
            model=name, config="benchmark",
            test_mae=round(f["mae"].mean(), 3), test_mae_std=round(f["mae"].std(ddof=0), 3),
            test_rmse=round(f["rmse"].mean(), 3), test_r2=round(f["r2"].mean(), 4),
            folds=folds,
        ))
    return rows


def _amplitude_ratio(last_fold: pd.DataFrame | None):
    # std(prédiction) / std(réel) sur le dernier pli (hors échantillon).
    # < 1 => le modèle rabote les pics.
    if last_fold is None or last_fold["y_true"].std() == 0:
        return None
    return round(float(last_fold["y_pred"].std() / last_fold["y_true"].std()), 3)


def _save_importance(bundle: dict, tag: str) -> None:
    if bundle.get("model_type") != "LightGBM":
        return
    gain = bundle["model"].booster_.feature_importance("gain")
    imp = pd.DataFrame({"feature": bundle["feature_cols"], "gain": gain})
    # Les ~500 colonnes météo départementales sont regroupées par variable.
    imp["group"] = imp["feature"].map(lambda c: c.split("__")[0] + " (by dept)" if "__" in c else c)
    imp = imp.groupby("group", as_index=False)["gain"].sum()
    imp["share"] = imp["gain"] / imp["gain"].sum()
    imp.sort_values("share", ascending=False).to_csv(OUT_DIR / f"importance_{tag}.csv", index=False)


def run():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = OUT_DIR / f"ablation_{stamp}.csv"
    folds_csv = OUT_DIR / f"folds_{stamp}.csv"

    print(f"[ablation] Chargement des données {START} -> {END} (freq={FREQ})")
    t_load = time.time()
    data = load_data(START, END, freq=FREQ)
    print(f"[ablation] {len(data)} lignes chargées en {time.time() - t_load:.0f}s\n")

    rows, fold_rows = [], []
    for b in _naive_benchmarks(data):
        for i, f in enumerate(b.pop("folds"), start=1):
            fold_rows.append(dict(model=b["model"], config="benchmark", fold=i, **f))
        rows.append(b)
        print(f"[ablation] {b['model']} -> test MAE={b['test_mae']} R²={b['test_r2']}")

    jobs = [(mt, cfg) for mt in MODEL_TYPES for cfg in CONFIGS]
    for i, (model_type, cfg) in enumerate(jobs, start=1):
        tag = f"{model_type}_{cfg['name']}"
        print("#" * 72 + f"\n[ablation] ({i}/{len(jobs)}) {tag}\n" + "#" * 72)
        t0 = time.time()
        try:
            pm.MODEL_DIR = RUNS_DIR / tag
            bundle = model_learn(
                _apply_config(data, cfg), model_type, n_splits=N_SPLITS,
                meteo_mode=METEO_MODE, use_neighbour_prices=cfg["neighbours"],
            )
            scores = bundle.get("scores") or {}
            test, spike = scores.get("test", {}), scores.get("spike", {})
            last_fold = bundle.get("last_fold_pred")
            if last_fold is not None:
                last_fold.to_parquet(OUT_DIR / f"last_fold_{tag}.parquet", index=False)
            _save_importance(bundle, tag)
            for k, f in enumerate(scores.get("folds", []), start=1):
                fold_rows.append(dict(model=model_type, config=cfg["name"], fold=k, **f))
            row = dict(
                model=model_type, config=cfg["name"],
                neighbours=cfg["neighbours"], gas=cfg["gas"], capacity=cfg["capacity"],
                n_features=len(bundle.get("feature_cols") or []),
                test_mae=round(test.get("mae", np.nan), 3),
                test_mae_std=round(test.get("mae_std", np.nan), 3),
                test_rmse=round(test.get("rmse", np.nan), 3),
                test_r2=round(test.get("r2", np.nan), 4),
                spike_precision=round(spike["precision"], 3) if spike else None,
                spike_recall=round(spike["recall"], 3) if spike else None,
                spike_f1=round(spike["f1"], 3) if spike else None,
                ampl_ratio=_amplitude_ratio(last_fold),
                secs=round(time.time() - t0, 1),
            )
        except Exception as exc:
            traceback.print_exc()
            row = dict(model=model_type, config=cfg["name"], error=repr(exc),
                       secs=round(time.time() - t0, 1))
        finally:
            pm.MODEL_DIR = Path("results/models")

        rows.append(row)
        # Réécriture incrémentale : rien n'est perdu si interruption.
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        pd.DataFrame(fold_rows).to_csv(folds_csv, index=False)
        print(f"\n[ablation] {tag} -> test MAE={row.get('test_mae')} "
              f"R²={row.get('test_r2')} ({row.get('secs')}s)\n")

    res = pd.DataFrame(rows).sort_values("test_mae", na_position="last")
    cols = [c for c in ["model", "config", "n_features", "test_mae", "test_rmse", "test_r2",
                        "spike_recall", "spike_precision", "ampl_ratio", "secs"] if c in res]
    print("=" * 72 + "\n[ablation] RÉCAPITULATIF — trié par test MAE croissant\n" + "=" * 72)
    print(res[cols].to_string(index=False))
    print(f"\n[ablation] CSV : {out_csv}")


if __name__ == "__main__":
    run()
