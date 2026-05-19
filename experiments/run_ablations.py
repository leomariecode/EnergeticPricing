"""
Ablation overnight : mesure l'impact du prix du gaz, des capacités installées,
de l'écrêtage du vent et de l'objectif LightGBM sur la qualité de prédiction.

Lancer (depuis n'importe où) :
    python -u experiments/run_ablations.py
En gardant un log consultable en direct (tail -f) :
    python -u experiments/run_ablations.py > experiments/ablation.log 2>&1 &

Les données météo / ENTSO-E / gaz sont chargées UNE seule fois puis réutilisées
pour toutes les configs. Les résultats sont écrits dans
experiments/ablation_results_<timestamp>.csv et réécrits après CHAQUE config,
donc rien n'est perdu en cas d'interruption.

Pour forcer un ré-entraînement propre : supprimer experiments/_runs/.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# On se place à la racine du repo (les chemins du loader sont relatifs).
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(line_buffering=True)  # logs lisibles en direct

import pandas as pd

from src.data.loader import load_data
import src.models.pricing_from_meteo as pm
from src.models.pricing_from_meteo import model_learn, predict_price_from_data


# ----------------------------------------------------------------------
# Paramètres — à ajuster AVANT de lancer
# ----------------------------------------------------------------------
START = "2023-01-01"
END = "2026-05-01"
FREQ = "1h"                  # "1h" ou "15min" (15min ≈ 4x plus lent)
N_SPLITS = 5
MODEL_TYPES = ["LightGBM","Simple"]   # ajouter "Simple", "RandomForest"... pour comparer
METEO_MODE = "department"    # "department" ou "national"

RUNS_DIR = ROOT / "experiments" / "_runs"


# ----------------------------------------------------------------------
# Grille des configurations testées
# ----------------------------------------------------------------------
# Champs d'une config (les optionnels prennent les valeurs de DEFAULTS) :
#   use_gas          : inclure la feature prix du gaz
#   use_capacity     : inclure les features de capacité (potentiel + totaux)
#   capacity_totals  : si True, garde solar/wind_capacity_total ; sinon potentiel seul
#   wind_clip        : vitesse de vent (m/s) d'écrêtage du cube éolien
#                      (12 = défaut ; 100 = quasi pas d'écrêtage)
#   objective        : objectif LightGBM — "regression" (L2, défaut),
#                      "regression_l1" (MAE) ou "huber" (les 2 lissent moins)
CONFIGS = [
    dict(name="baseline",               use_gas=False, use_capacity=False),
    dict(name="gas",                    use_gas=True,  use_capacity=False),
    dict(name="gas+cap_full",           use_gas=True,  use_capacity=True, capacity_totals=True),
    dict(name="gas+cap_potonly",        use_gas=True,  use_capacity=True, capacity_totals=False),
    dict(name="gas+cap_declip",         use_gas=True,  use_capacity=True, capacity_totals=True,  wind_clip=100.0),
    dict(name="gas+cap_potonly_declip", use_gas=True,  use_capacity=True, capacity_totals=False, wind_clip=100.0),
    dict(name="gas+cap_full_MAE",       use_gas=True,  use_capacity=True, capacity_totals=True,  objective="regression_l1"),
    dict(name="gas+cap_full_huber",     use_gas=True,  use_capacity=True, capacity_totals=True,  objective="huber"),
]

DEFAULTS = dict(capacity_totals=True, wind_clip=12.0, objective="regression")


# ----------------------------------------------------------------------
# Mécanique : application d'une config via patch des globals du module
# ----------------------------------------------------------------------
_ORIG_BUILD_MODEL = pm._build_model
_ORIG_MODEL_DIR = pm.MODEL_DIR
_ORIG_WIND_CLIP = pm._WIND_RATED_SPEED
_FULL_CAPACITY_FEATURES = list(pm.CAPACITY_FEATURES)
_POTENTIAL_ONLY = [c for c in _FULL_CAPACITY_FEATURES if c.endswith("_potential")]


def _patched_build_model(objective):
    # Applique l'objectif voulu au LightGBM, laisse les autres modèles inchangés.
    def _bm(model_type):
        model = _ORIG_BUILD_MODEL(model_type)
        if model_type == "LightGBM" and objective != "regression":
            model.set_params(objective=objective)
        return model
    return _bm


def _apply_config(data, cfg):
    # Renvoie (data_cfg, restore) : les données filtrées selon la config et une
    # fonction qui remet les globals du module dans leur état d'origine.
    data_cfg = data.copy()
    if not cfg["use_gas"]:
        data_cfg = data_cfg.drop(columns=["gas_price"], errors="ignore")
    if not cfg["use_capacity"]:
        data_cfg = data_cfg.drop(columns=["solar_capacity", "wind_capacity"], errors="ignore")

    pm.CAPACITY_FEATURES = _FULL_CAPACITY_FEATURES if cfg["capacity_totals"] else _POTENTIAL_ONLY
    pm._WIND_RATED_SPEED = cfg["wind_clip"]
    pm._build_model = _patched_build_model(cfg["objective"])

    def restore():
        pm.CAPACITY_FEATURES = _FULL_CAPACITY_FEATURES
        pm._WIND_RATED_SPEED = _ORIG_WIND_CLIP
        pm._build_model = _ORIG_BUILD_MODEL
        pm.MODEL_DIR = _ORIG_MODEL_DIR

    return data_cfg, restore


def _amplitude_ratio(bundle, data_cfg):
    # std(prédiction) / std(prix réel) sur tout le jeu (in-sample, indicatif).
    # < 1 => le modèle rabote les pics (lissage) ; proche de 1 => amplitude gardée.
    try:
        pred = predict_price_from_data(bundle, data_cfg)
        actual = (
            data_cfg[["time", "price"]].dropna()
            .drop_duplicates("time").set_index("time")["price"]
        )
        j = pred.to_frame("p").join(actual.rename("a"), how="inner").dropna()
        if len(j) < 10 or j["a"].std() == 0:
            return None
        return round(float(j["p"].std() / j["a"].std()), 3)
    except Exception:
        traceback.print_exc()
        return None


def run():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = ROOT / "experiments" / f"ablation_results_{stamp}.csv"

    print(f"[ablation] Chargement des données {START} -> {END} (freq={FREQ})")
    t_load = time.time()
    data = load_data(START, END, freq=FREQ)
    print(f"[ablation] {len(data)} lignes chargées en {time.time() - t_load:.0f}s\n")

    jobs = [(mt, cfg) for mt in MODEL_TYPES for cfg in CONFIGS]
    rows = []
    for i, (model_type, raw_cfg) in enumerate(jobs, start=1):
        cfg = {**DEFAULTS, **raw_cfg}
        tag = f"{model_type}/{cfg['name']}"
        print("#" * 72)
        print(f"[ablation] ({i}/{len(jobs)}) {tag}")
        print("#" * 72)

        data_cfg, restore = _apply_config(data, cfg)
        t0 = time.time()
        try:
            # MODEL_DIR dédié -> pas de collision de cache entre configs.
            pm.MODEL_DIR = RUNS_DIR / f"{model_type}__{cfg['name']}"
            bundle = model_learn(
                data_cfg, model_type, n_splits=N_SPLITS, meteo_mode=METEO_MODE,
            )
            scores = bundle.get("scores") or {}
            test = scores.get("test", {})
            spike = scores.get("spike", {})
            row = {
                "model": model_type,
                "config": cfg["name"],
                "use_gas": cfg["use_gas"],
                "use_capacity": cfg["use_capacity"],
                "capacity_totals": cfg["capacity_totals"],
                "wind_clip": cfg["wind_clip"],
                "objective": cfg["objective"],
                "n_features": len(bundle.get("feature_cols") or []),
                "test_r2": round(test.get("r2", float("nan")), 4),
                "test_mae": round(test.get("mae", float("nan")), 3),
                "test_rmse": round(test.get("rmse", float("nan")), 3),
                "spike_precision": round(spike["precision"], 3) if spike else None,
                "spike_recall": round(spike["recall"], 3) if spike else None,
                "spike_f1": round(spike["f1"], 3) if spike else None,
                "ampl_ratio": _amplitude_ratio(bundle, data_cfg),
                "secs": round(time.time() - t0, 1),
            }
        except Exception as exc:
            traceback.print_exc()
            row = {
                "model": model_type, "config": cfg["name"],
                "error": repr(exc), "secs": round(time.time() - t0, 1),
            }
        finally:
            restore()

        rows.append(row)
        # Réécriture incrémentale du CSV : rien n'est perdu si interruption.
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\n[ablation] {tag} -> test MAE={row.get('test_mae')} "
              f"R²={row.get('test_r2')} ampl={row.get('ampl_ratio')} "
              f"({row.get('secs')}s)\n")

    # ------------------------------------------------------------------
    # Récapitulatif
    # ------------------------------------------------------------------
    res = pd.DataFrame(rows)
    if "test_mae" in res.columns:
        res = res.sort_values("test_mae", na_position="last")
    cols = ["model", "config", "n_features", "test_mae", "test_r2", "test_rmse",
            "spike_recall", "spike_precision", "ampl_ratio", "secs"]
    cols = [c for c in cols if c in res.columns]
    print("=" * 72)
    print("[ablation] RÉCAPITULATIF — trié par test MAE croissant")
    print("  ampl_ratio proche de 1 = peu de lissage ; bas = pics rabotés")
    print("=" * 72)
    print(res[cols].to_string(index=False))
    print(f"\n[ablation] CSV complet : {out_csv}")


if __name__ == "__main__":
    run()
