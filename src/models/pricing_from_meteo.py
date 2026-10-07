from pathlib import Path

# IMPORTANT (macOS x86_64) : charger l'OpenMP de LightGBM *puis* celui de PyTorch
# fait segfaulter le process — deux runtimes OpenMP incompatibles cohabitent. Si
# PyTorch est installé (chemin TFT), on l'importe ICI, avant LightGBM, pour que
# son runtime OpenMP soit initialisé en premier. No-op si torch est absent.
try:  # pragma: no cover - dépend de l'environnement
    import torch  # noqa: F401
except ImportError:
    pass

import holidays
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_data


_FR_HOLIDAYS = holidays.country_holidays("FR")


def _infer_freq_tag(time_series) -> str:
    # Détecte si le pas de temps des données est horaire ou 15-min, à partir
    # du pas médian observé. Sert à la fois pour le cache modèle et pour les logs.
    t = pd.to_datetime(time_series).drop_duplicates().sort_values()
    if len(t) < 2:
        return "1h"
    median_step = t.diff().dropna().median()
    return "15min" if median_step <= pd.Timedelta("16min") else "1h"


METEO_FEATURES = [
    "temperature_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "cloud_cover",
    "precipitation",
]

# Prévisions ENTSO-E : conso et renouvelable prévus. Ce sont les drivers principaux
# du prix DA (équilibre offre/demande anticipé). Optionnels : on tombe en retrait
# avec une moyenne mobile si la prévision manque (heure récente, ou modèle entraîné
# sur un historique antérieur à la mise en place du chargement).
ENTSOE_FORECAST_FEATURES = [
    "load_forecast",
    "wind_forecast",
    "solar_forecast",
    "net_load_forecast",  # load - (wind + solar), proxy de la demande résiduelle
]

TEMPORAL_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

# Prix day-ahead des pays voisins (couplage de marché). ATTENTION : le DA de tous
# les pays couplés est fixé simultanément par l'algorithme EUPHEMIA -> en usage
# "prévision pure" du lendemain, le prix voisin n'est pas connu à l'avance non plus.
# Très utile pour le backtest / la reconstruction, à manier avec prudence en live.
NEIGHBOUR_PRICE_FEATURES = [
    "price_DE",
    "price_BE",
    "price_ES",
    "price_IT",
    "price_CH",
    "price_GB",
]

# Prix du gaz TTF (référence européenne, EUR/MWh). Dès qu'une centrale CCGT est
# en marge, le gaz fixe le prix marginal de l'électricité : c'est un driver direct
# du prix day-ahead. Connu la veille (le DA gaz se règle avant le DA élec) -> pas
# de fuite pour la prévision J+1. Optionnel (cf. _select_available).
GAS_FEATURES = [
    "gas_price",
]

# Parc nucléaire (cf. loader.load_nuclear_features) : production moyenne de J-2
# et sa tendance hebdo (aucune fuite), MW en arrêt planifié à l'heure h (fuite
# faible : seules les prolongations d'arrêts déjà annoncés sont connues ex post).
NUCLEAR_FEATURES = [
    "nuclear_gen_d2",
    "nuclear_gen_d2_trend",
    "nuclear_planned_unavail",
]

# Dérivées du nucléaire : ce que le prix de la veille ne peut pas contenir.
#   nuclear_unavail_delta_1d : variation des arrêts planifiés J vs J-1 (même heure)
#   thermal_residual         : demande résiduelle - nucléaire disponible estimé
#                              (= production J-2 corrigée des arrêts planifiés
#                              intervenus depuis J-2) -> ce qui reste au gaz/charbon.
NUCLEAR_DERIVED_FEATURES = [
    "nuclear_unavail_delta_1d",
    "thermal_residual",
]
USE_NUCLEAR_DERIVED = True

# Features ciblées sur les prix négatifs (10h-16h, week-ends et ponts, avril-juin) :
#   renewable_share  : (éolien + solaire prévus) / conso prévue -> excès de fatal
#   is_bridge        : jour de pont (entre un férié et un week-end)
#   solar_x_offday   : solaire prévu un jour non travaillé (week-end, férié, pont)
#   neg_hours_d1/_d7 : nb d'heures à prix <= 0 à J-1 / J-7 (épisodes en vagues)
#   price_min_d1     : prix minimum de J-1 (profondeur du dernier épisode)
# Les prix de J-1 sont publiés à J-2 midi : connus avant l'enchère de J.
NEG_FEATURES = [
    "renewable_share",
    "is_bridge",
    "solar_x_offday",
    "neg_hours_d1",
    "neg_hours_d7",
    "price_min_d1",
]
USE_NEG_FEATURES = True

# Capacité installée renouvelable (registre national, agrégée par département) et
# "potentiel" de production = somme sur les départements de capacité × ressource
# météo locale. Le potentiel estime directement la production solaire/éolienne
# attendue -> driver majeur du prix (un fort potentiel écrase le prix, parfois
# jusqu'au négatif). Bien plus prédictif que la météo brute, et sans surcoût de
# téléchargement (la capacité commune est agrégée au département déjà chargé).
CAPACITY_FEATURES = [
    "solar_potential",
    "wind_potential",
    "solar_capacity_total",
    "wind_capacity_total",
]

# Activé automatiquement quand le pas de temps est sub-horaire (cf. _prepare_features).
SUBHOURLY_TEMPORAL_FEATURES = [
    "quarter_of_hour",
    "minute_of_day_sin",
    "minute_of_day_cos",
]

# Gradients (variations pas-à-pas) des prévisions ENTSO-E. Ce sont les *rampes*
# (montée solaire le matin, chute le soir, rampe de conso) qui causent les pics
# de prix, surtout au pas 15-min -> features les plus utiles pour les spikes.
GRADIENT_FEATURES = [
    "net_load_grad",
    "load_grad",
    "solar_grad",
    "wind_grad",
]

# Lags du prix : prix d'il y a 1 jour et 1 semaine au même créneau. Pas de fuite
# tant que l'horizon de prédiction est >= 24 h (cas prévision J+1). Très informatif
# car le prix est fortement auto-corrélé jour-à-jour et semaine-à-semaine.
LAG_FEATURES = [
    "price_lag_1d",
    "price_lag_7d",
]

# Définition d'un "pic" : prix hors de l'intervalle [q05, q95] du jeu d'entraînement
# (hausses de scarcité OU prix négatifs). Un classifieur dédié apprend à les détecter,
# car un régresseur MSE seul lisse systématiquement les extrêmes.
SPIKE_LOW_Q = 0.05
SPIKE_HIGH_Q = 0.95

MODEL_DIR = Path("results/models")


def model_learn(
    data: pd.DataFrame,
    model_type: str,
    n_splits: int = 5,
    meteo_mode: str = "department",
    fast: bool = True,
    use_neighbour_prices: bool = False,
):
    # use_neighbour_prices : inclut les prix DA des pays voisins. Désactivé par défaut,
    # car ils sont fixés en même temps que le prix FR (couplage EUPHEMIA) : utile pour
    # reconstruire un prix a posteriori, mais inconnu au moment de la prévision J-1.
    # Cache : si un modèle a déjà été entraîné sur la même plage / type / fréquence
    # / mode météo, on le recharge.
    # fast : ne concerne que le TFT (cf. learn_tft) — mode rapide pour les longs
    # historiques. En mode rapide la météo est forcée en 'national' : on le reflète
    # dans le tag de cache pour ne pas collisionner avec un run 'department'.
    print(f"[model_learn] Démarrage : model_type={model_type}, meteo_mode={meteo_mode}, "
          f"{len(data)} lignes en entrée")
    start = pd.to_datetime(data["time"]).min().strftime("%Y%m%d")
    end = pd.to_datetime(data["time"]).max().strftime("%Y%m%d")
    freq_tag = _infer_freq_tag(data["time"])
    tft_fast = (model_type == "TFT") and fast
    effective_meteo_mode = "national" if tft_fast else meteo_mode
    meteo_tag = "dept" if effective_meteo_mode == "department" else "nat"
    print(f"[model_learn] Plage temporelle : {start} -> {end} (pas={freq_tag}, météo={meteo_tag})")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fast_tag = "_fast" if tft_fast else ""
    nb_tag = "_nb" if use_neighbour_prices else ""
    cache_path = MODEL_DIR / (
        f"pricing_from_meteo_{model_type}_{start}_{end}_{freq_tag}_{meteo_tag}{fast_tag}{nb_tag}.pkl"
    )

    if cache_path.exists():
        print(f"[model_learn] Modèle déjà entraîné, rechargement : {cache_path}")
        bundle = joblib.load(cache_path)
        if "scores" in bundle:
            _print_scores(bundle)
        return bundle

    # Le TFT a un pipeline d'entraînement à part (séquentiel, PyTorch).
    if model_type == "TFT":
        from src.models.tft import learn_tft
        bundle = learn_tft(data, meteo_mode=meteo_mode, n_splits=n_splits, fast=fast)
        joblib.dump(bundle, cache_path)
        print(f"[model_learn] Modèle TFT sauvegardé : {cache_path}")
        return bundle

    print("[model_learn] Préparation des features (drop des heures sans prix réel)")
    df = _prepare_features(data, drop_missing_price=True, meteo_mode=meteo_mode)

    # On n'utilise une feature optionnelle que si elle est massivement présente
    # (>= 60% de couverture sur le jeu d'entraînement). Sinon on l'écarte pour
    # éviter d'introduire un signal très bruité par l'imputation.
    meteo_cols = _meteo_cols_of(df)
    fc_features = _select_available(df, ENTSOE_FORECAST_FEATURES)
    nb_features = (
        _select_available(df, NEIGHBOUR_PRICE_FEATURES) if use_neighbour_prices else []
    )
    gas_features = _select_available(df, GAS_FEATURES)
    nuc_features = _select_available(df, NUCLEAR_FEATURES + NUCLEAR_DERIVED_FEATURES)
    cap_features = _select_available(df, CAPACITY_FEATURES)
    grad_features = _select_available(df, GRADIENT_FEATURES)
    lag_features = _select_available(df, LAG_FEATURES)
    neg_features = _select_available(df, NEG_FEATURES)
    # Features intra-heure ajoutées automatiquement par _prepare_features quand le pas est 15-min.
    subhourly = [c for c in SUBHOURLY_TEMPORAL_FEATURES if c in df.columns]
    feature_cols = (
        meteo_cols + fc_features + nb_features + gas_features + nuc_features + cap_features
        + grad_features + lag_features + neg_features + TEMPORAL_FEATURES + subhourly
    )
    print(
        f"[model_learn] Features ({len(feature_cols)}) : "
        f"météo={len(meteo_cols)} ({meteo_tag}), forecasts={fc_features}, voisins={nb_features}, "
        f"gaz={gas_features}, nucléaire={nuc_features}, capacité={cap_features}, gradients={grad_features}, "
        f"lags={lag_features}, temporel={len(TEMPORAL_FEATURES)}, intra-heure={subhourly}"
    )
    print(f"[model_learn] {len(df)} lignes d'entraînement après filtrage")

    X = _build_feature_matrix(df, feature_cols, model_type)
    y = df["price"].values

    # Labels de pic : prix hors [q05, q95] du jeu d'entraînement.
    q_lo = float(np.quantile(y, SPIKE_LOW_Q))
    q_hi = float(np.quantile(y, SPIKE_HIGH_Q))
    is_spike = ((y < q_lo) | (y > q_hi)).astype(int)
    print(
        f"[model_learn] Pics : seuils [{q_lo:.1f}, {q_hi:.1f}] €/MWh, "
        f"{is_spike.sum()} pics / {len(is_spike)} pas ({100 * is_spike.mean():.1f}%)"
    )

    # Validation rolling-origin (TimeSeriesSplit) : moyenne des scores sur n_splits
    # plis chronologiques. Bien plus robuste qu'un split unique 80/20.
    n_splits = min(n_splits, max(2, len(df) // 24))  # garde-fou pour petits jeux
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = {"train": [], "test": []}
    spike_cv = []
    fold_details = []
    last_split_times = None
    last_fold_pred = None
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model = _build_model(model_type)
        # Early stopping sur la fin du train (10 %), jamais sur le pli de test :
        # sinon le nombre d'arbres serait choisi en regardant le test.
        n_es = max(24, int(len(train_idx) * 0.1))
        model = _fit_with_validation(
            model, model_type, X_tr[:-n_es], y_tr[:-n_es], X_tr[-n_es:], y_tr[-n_es:]
        )
        fold_scores = _compute_fold_scores(model, X_tr, y_tr, X_te, y_te)
        cv_scores["train"].append(fold_scores["train"])
        cv_scores["test"].append(fold_scores["test"])
        fold_details.append({
            **fold_scores["test"],
            "test_start": df["time"].iloc[test_idx[0]].isoformat(),
            "test_end": df["time"].iloc[test_idx[-1]].isoformat(),
        })
        if fold == n_splits:
            last_fold_pred = pd.DataFrame({
                "time": df["time"].iloc[test_idx].values,
                "y_true": y_te,
                "y_pred": model.predict(X_te),
            })

        # Classifieur de pic évalué sur le même découpage temporel.
        s_tr, s_te = is_spike[train_idx], is_spike[test_idx]
        spike_line = ""
        if len(np.unique(s_tr)) > 1:
            clf = _build_spike_classifier()
            clf.fit(X_tr, s_tr)
            sm = _spike_metrics(s_te, clf.predict(X_te))
            spike_cv.append(sm)
            spike_line = f" | spike P={sm['precision']:.2f} R={sm['recall']:.2f}"

        last_split_times = (
            df["time"].iloc[train_idx[-1]],
            df["time"].iloc[test_idx[0]],
            df["time"].iloc[test_idx[-1]],
        )
        print(
            f"[model_learn] Pli {fold}/{n_splits} : "
            f"train n={len(train_idx)} R²={fold_scores['train']['r2']:.3f} | "
            f"test n={len(test_idx)} R²={fold_scores['test']['r2']:.3f} "
            f"MAE={fold_scores['test']['mae']:.2f} €/MWh{spike_line}"
        )

    scores = _aggregate_cv_scores(cv_scores)
    scores["folds"] = fold_details
    if spike_cv:
        scores["spike"] = _aggregate_spike_scores(spike_cv)
    if last_split_times is not None:
        scores["time_test_start"] = last_split_times[1].isoformat()
        scores["time_test_end"] = last_split_times[2].isoformat()

    # Modèle final : entraînement sur l'intégralité du jeu de données.
    print("[model_learn] Ré-entraînement final sur l'intégralité du jeu de données")
    model = _build_model(model_type)
    if model_type in ("LightGBM", "LightGBM_hurdle"):
        # Pour LightGBM on garde un petit holdout (10 % des derniers points) pour
        # déclencher l'early stopping et figer le bon nombre d'itérations.
        n_es = max(24, int(len(df) * 0.1))
        model = _fit_with_validation(
            model, model_type, X[:-n_es], y[:-n_es], X[-n_es:], y[-n_es:]
        )
    else:
        model.fit(X, y)

    # Classifieur de pic final (sur tout le jeu) — None si une seule classe présente.
    spike_model = None
    if len(np.unique(is_spike)) > 1:
        spike_model = _build_spike_classifier()
        spike_model.fit(X, is_spike)
        print("[model_learn] Classifieur de pic entraîné")

    bundle = {
        "model": model,
        "model_type": model_type,
        "feature_cols": feature_cols,
        "meteo_mode": meteo_mode,
        "start": start,
        "end": end,
        "freq": freq_tag,
        "n_total": int(len(X)),
        "n_splits": int(n_splits),
        "scores": scores,
        "spike_model": spike_model,
        "last_fold_pred": last_fold_pred,
        "spike_thresholds": {
            "low": q_lo, "high": q_hi,
            "low_q": SPIKE_LOW_Q, "high_q": SPIKE_HIGH_Q,
        },
    }
    _print_scores(bundle)
    joblib.dump(bundle, cache_path)
    print(f"[model_learn] Modèle sauvegardé : {cache_path}")
    return bundle


def _select_available(df: pd.DataFrame, candidates: list[str], min_cov: float = 0.6) -> list[str]:
    # Garde les colonnes présentes ET couvertes à >= min_cov (fraction de non-NaN).
    return [
        c for c in candidates
        if c in df.columns and df[c].notna().mean() >= min_cov
    ]


def _build_spike_classifier():
    # LightGBM en classification binaire, class_weight='balanced' car les pics
    # ne représentent qu'~10 % des pas de temps.
    return lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=30,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def _spike_metrics(y_true, y_pred) -> dict:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_spikes": int(np.sum(y_true)),
    }


def _aggregate_spike_scores(rows: list[dict]) -> dict:
    out = {}
    for metric in ("precision", "recall", "f1"):
        vals = [r[metric] for r in rows]
        out[metric] = float(np.mean(vals))
        out[f"{metric}_std"] = float(np.std(vals))
    out["n_spikes"] = int(np.mean([r["n_spikes"] for r in rows]))
    return out


def _compute_fold_scores(model, X_train, y_train, X_test, y_test) -> dict:
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    return {
        "train": {
            "r2": float(r2_score(y_train, y_train_pred)),
            "mae": float(mean_absolute_error(y_train, y_train_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            "n": int(len(y_train)),
        },
        "test": {
            "r2": float(r2_score(y_test, y_test_pred)),
            "mae": float(mean_absolute_error(y_test, y_test_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            "n": int(len(y_test)),
            **_negative_regime_scores(y_test, y_test_pred),
        },
    }


def _negative_regime_scores(y_true, y_pred) -> dict:
    # Erreur sur les heures à prix <= 0 (biais > 0 = le modèle les surestime) et
    # sur les autres heures, pour vérifier qu'on ne dégrade pas le régime normal.
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    neg = y_true <= NEG_THRESHOLD
    out = {"n_neg": int(neg.sum())}
    if neg.any():
        out["mae_neg"] = float(np.mean(np.abs(y_pred[neg] - y_true[neg])))
        out["bias_neg"] = float(np.mean(y_pred[neg] - y_true[neg]))
    if (~neg).any():
        out["mae_pos"] = float(np.mean(np.abs(y_pred[~neg] - y_true[~neg])))
    return out


def _aggregate_cv_scores(cv_scores: dict) -> dict:
    def agg(rows, metric):
        vals = [r[metric] for r in rows]
        return float(np.mean(vals)), float(np.std(vals))

    out = {}
    for split in ("train", "test"):
        rows = cv_scores[split]
        r2_m, r2_s = agg(rows, "r2")
        mae_m, mae_s = agg(rows, "mae")
        rmse_m, rmse_s = agg(rows, "rmse")
        out[split] = {
            "r2": r2_m,
            "r2_std": r2_s,
            "mae": mae_m,
            "mae_std": mae_s,
            "rmse": rmse_m,
            "rmse_std": rmse_s,
            "n": int(np.mean([r["n"] for r in rows])),
        }
    return out


def _fmt(mean: float, std: float | None, fmt: str = ".3f", unit: str = "") -> str:
    if std is None or np.isnan(std):
        return f"{mean:{fmt}}{unit}"
    return f"{mean:{fmt}} ± {std:{fmt}}{unit}"


def _print_scores(bundle: dict) -> None:
    s = bundle.get("scores") or {}
    train = s.get("train", {})
    test = s.get("test", {})
    n_splits = bundle.get("n_splits")
    print("=" * 72)
    print(f"[model_score] Modèle '{bundle.get('model_type')}' — caractéristiques :")
    print(f"  - Plage entraînement   : {bundle.get('start')} -> {bundle.get('end')}")
    feats = bundle.get("feature_cols") or []
    n_meteo = sum("__" in c for c in feats)
    others = [c for c in feats if "__" not in c]
    print(f"  - Features ({len(feats)}) : {n_meteo} météo par département + {others}")
    if "time_test_start" in s:
        print(f"  - Dernier pli test     : {s.get('time_test_start')} -> {s.get('time_test_end')}")
    print(f"  - Heures totales       : {bundle.get('n_total')}")
    if n_splits:
        print(f"  - Validation           : TimeSeriesSplit, {n_splits} plis (rolling-origin)")
    if train:
        print(
            f"  - Train  R²={_fmt(train['r2'], train.get('r2_std'))}  "
            f"MAE={_fmt(train['mae'], train.get('mae_std'), '.2f', ' €/MWh')}  "
            f"RMSE={_fmt(train['rmse'], train.get('rmse_std'), '.2f', ' €/MWh')}  "
            f"(n moy ≈ {train.get('n')})"
        )
    if test:
        print(
            f"  - Test   R²={_fmt(test['r2'], test.get('r2_std'))}  "
            f"MAE={_fmt(test['mae'], test.get('mae_std'), '.2f', ' €/MWh')}  "
            f"RMSE={_fmt(test['rmse'], test.get('rmse_std'), '.2f', ' €/MWh')}  "
            f"(n moy ≈ {test.get('n')})"
        )
    spike = s.get("spike")
    if spike:
        thr = bundle.get("spike_thresholds") or {}
        seuils = ""
        if "low" in thr:
            seuils = f" (seuils {thr['low']:.0f}/{thr['high']:.0f} €/MWh)"
        print(
            f"  - Pics   Précision={_fmt(spike['precision'], spike.get('precision_std'))}  "
            f"Rappel={_fmt(spike['recall'], spike.get('recall_std'))}  "
            f"F1={_fmt(spike['f1'], spike.get('f1_std'))}{seuils}"
        )
    print("=" * 72)


def model_score(bundle: dict, data: pd.DataFrame | None = None) -> dict:
    # Si data n'est pas fourni, on rappelle simplement les scores stockés au fit.
    # Si data est fourni, on évalue le modèle déjà entraîné sur ce jeu (sans réentraîner)
    # via les mêmes plis temporels, pour comparer fit-CV vs out-of-sample.
    if data is None or (bundle.get("model_type") == "TFT"):
        # Pour le TFT, le ré-scoring out-of-sample n'utilise pas le chemin sklearn :
        # on se contente des scores stockés au fit.
        _print_scores(bundle)
        return bundle.get("scores") or {}

    meteo_mode = bundle.get("meteo_mode", "national")
    df = _prepare_features(data, drop_missing_price=True, meteo_mode=meteo_mode)
    feature_cols = bundle["feature_cols"]
    model_type = bundle.get("model_type") or "Simple"
    X = _build_feature_matrix(df, feature_cols, model_type)
    y = df["price"].values

    n_splits = bundle.get("n_splits") or 5
    n_splits = min(n_splits, max(2, len(df) // 24))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv = {"train": [], "test": []}
    for train_idx, test_idx in tscv.split(X):
        fold = _compute_fold_scores(
            bundle["model"], X[train_idx], y[train_idx], X[test_idx], y[test_idx]
        )
        cv["train"].append(fold["train"])
        cv["test"].append(fold["test"])
    scores = _aggregate_cv_scores(cv)
    eval_bundle = {**bundle, "scores": scores}
    _print_scores(eval_bundle)
    return scores


def predict_price_from_meteo(model_bundle: dict, date) -> pd.Series:
    # date : str "YYYY-MM-DD" ou pd.Timestamp -> on prédit les 24h de cette journée.
    # On honore le pas du modèle (stocké dans le bundle) pour que la météo/forecasts
    # arrivent au bon pas avant prédiction.
    print(f"[predict_price_from_meteo] Démarrage prédiction pour {date}")
    date = pd.to_datetime(date).strftime("%Y-%m-%d")
    next_date = (pd.to_datetime(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    freq = model_bundle.get("freq", "1h")
    print(f"[predict_price_from_meteo] Chargement des données {date} -> {next_date} (freq={freq})")
    data = load_data(date, next_date, freq=freq)
    return predict_price_from_data(model_bundle, data)


def predict_price_from_data(model_bundle: dict, data: pd.DataFrame) -> pd.Series:
    # Prédit à partir d'un DataFrame déjà chargé (mêmes colonnes que load_data).
    print(f"[predict_price_from_data] {len(data)} lignes en entrée")

    model_type = model_bundle.get("model_type") or "Simple"
    if model_type == "TFT":
        from src.models.tft import predict_tft
        return predict_tft(model_bundle, data)

    # meteo_mode hérité du bundle (legacy sans clé -> 'national', l'ancien comportement).
    meteo_mode = model_bundle.get("meteo_mode", "national")
    df = _prepare_features(data, meteo_mode=meteo_mode)

    feature_cols = model_bundle["feature_cols"]
    model = model_bundle["model"]

    # Garde-fou : si des colonnes attendues manquent (ex. bundle entraîné avec ENTSO-E
    # forecasts mais prédict-set sans), on les ajoute en NaN et _build_feature_matrix
    # gérera l'imputation (LightGBM : NaN ; autres : moyenne de colonne sur le set actuel).
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    print(f"[predict_price_from_data] Prédiction sur {len(df)} pas de temps")
    X = _build_feature_matrix(df, feature_cols, model_type)
    preds = model.predict(X)
    print(f"[predict_price_from_data] Prédiction terminée (moyenne={preds.mean():.2f} EUR/MWh)")

    return pd.Series(preds, index=df["time"], name="predicted_price")


def predict_spike_proba(model_bundle: dict, data: pd.DataFrame) -> pd.Series:
    # Probabilité qu'un pas de temps soit un "pic" (prix extrême), via le classifieur
    # dédié stocké dans le bundle. Renvoie une Série indexée par time, vide si le
    # bundle n'a pas de classifieur (modèle ancien, ou une seule classe à l'entraînement).
    spike_model = model_bundle.get("spike_model")
    if spike_model is None:
        print("[predict_spike_proba] Pas de classifieur de pic dans ce modèle.")
        return pd.Series(dtype=float, name="spike_proba")

    meteo_mode = model_bundle.get("meteo_mode", "national")
    df = _prepare_features(data, meteo_mode=meteo_mode)
    feature_cols = model_bundle["feature_cols"]
    model_type = model_bundle.get("model_type") or "Simple"
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    X = _build_feature_matrix(df, feature_cols, model_type)
    proba = spike_model.predict_proba(X)[:, 1]
    return pd.Series(proba, index=df["time"], name="spike_proba")


def _aggregate_meteo(data: pd.DataFrame, meteo_mode: str):
    # Renvoie (df indexé par 'time' avec les colonnes météo, liste de ces colonnes).
    # - meteo_mode="national"   : 1 valeur par variable (moyenne pondérée population).
    # - meteo_mode="department" : 1 colonne par (variable, département) -> pivot large.
    meteo_present = [c for c in METEO_FEATURES if c in data.columns]
    data = data.copy()
    data["time"] = pd.to_datetime(data["time"])

    if meteo_mode == "national":
        has_pop = "population" in data.columns and data["population"].notna().any()
        if has_pop:
            # Moyenne pondérée : sum(x*w)/sum(w) par heure (robuste aux poids NaN).
            w = data["population"].astype(float)
            weighted = data[meteo_present].multiply(w, axis=0)
            weighted["time"] = data["time"].values
            weighted["__w"] = w.values
            num = weighted.groupby("time", as_index=False)[meteo_present + ["__w"]].sum()
            dfm = num[meteo_present].div(num["__w"], axis=0)
            dfm["time"] = num["time"]
            return dfm[["time"] + meteo_present], meteo_present
        dfm = data.groupby("time", as_index=False)[meteo_present].mean()
        return dfm, meteo_present

    if meteo_mode == "department":
        if "code" not in data.columns:
            raise ValueError("meteo_mode='department' requiert la colonne 'code' (département).")
        # Pivot : chaque (variable, département) devient une colonne 'variable__code'.
        wide = data.pivot_table(index="time", columns="code", values=meteo_present, aggfunc="mean")
        wide.columns = [f"{var}__{code}" for var, code in wide.columns]
        wide = wide.reset_index()
        cols = sorted(c for c in wide.columns if c != "time")
        return wide, cols

    raise ValueError(f"meteo_mode inconnu : {meteo_mode!r} (attendu 'national' ou 'department')")


def _meteo_cols_of(df: pd.DataFrame) -> list[str]:
    # Retrouve les colonnes météo d'un df déjà préparé : soit les colonnes pivotées
    # 'variable__code', soit les variables nationales brutes.
    dept = sorted(
        c for c in df.columns
        if "__" in c and c.split("__")[0] in METEO_FEATURES
    )
    if dept:
        return dept
    return [c for c in METEO_FEATURES if c in df.columns]


# Plafond (m/s) appliqué à la vitesse de vent avant de la cuber dans wind_potential.
# Réglé très haut (≈ aucun écrêtage) : config 'cap_declip' retenue. On laisse
# LightGBM apprendre lui-même la saturation de la courbe de puissance plutôt que
# de l'imposer à 12 m/s — ça préserve mieux les creux de prix par vent fort.
_WIND_RATED_SPEED = 100.0


def _renewable_potential(data: pd.DataFrame) -> pd.DataFrame | None:
    # Construit les features "potentiel" renouvelable à partir des lignes brutes
    # (time, code, météo, capacité par département). Pour chaque pas de temps :
    #   solar_potential = Σ_dép  capacité_solaire_dép × rayonnement_dép
    #   wind_potential  = Σ_dép  capacité_éolien_dép  × vitesse_vent_dép³ (écrêtée)
    # Le cube approxime la courbe de puissance éolienne (P ∝ v³ sous le nominal).
    # Renvoie aussi la capacité installée totale (solaire / éolien). None si la
    # capacité n'est pas disponible (données chargées par un loader antérieur).
    if not {"solar_capacity", "wind_capacity"}.issubset(data.columns):
        return None

    d = data.copy()
    d["time"] = pd.to_datetime(d["time"])
    d["solar_capacity"] = d["solar_capacity"].fillna(0.0)
    d["wind_capacity"] = d["wind_capacity"].fillna(0.0)

    out = d.groupby("time", as_index=False).agg(
        solar_capacity_total=("solar_capacity", "sum"),
        wind_capacity_total=("wind_capacity", "sum"),
    )
    if "shortwave_radiation" in d.columns:
        d["_solar"] = d["solar_capacity"] * d["shortwave_radiation"].fillna(0.0)
        sp = d.groupby("time", as_index=False)["_solar"].sum()
        out = out.merge(sp.rename(columns={"_solar": "solar_potential"}), on="time", how="left")
    if "wind_speed_10m" in d.columns:
        ws = d["wind_speed_10m"].clip(lower=0.0, upper=_WIND_RATED_SPEED)
        d["_wind"] = d["wind_capacity"] * ws ** 3
        wp = d.groupby("time", as_index=False)["_wind"].sum()
        out = out.merge(wp.rename(columns={"_wind": "wind_potential"}), on="time", how="left")
    return out


def _prepare_features(
    data: pd.DataFrame,
    drop_missing_price: bool = False,
    meteo_mode: str = "department",
) -> pd.DataFrame:
    # Agrégation par pas de temps. La météo est traitée selon meteo_mode (cf. _aggregate_meteo) :
    # 'department' garde la valeur par département (95x5 colonnes), 'national' agrège en
    # une moyenne pondérée population. Le prix et les prévisions ENTSO-E (nationaux,
    # dupliqués sur les 95 dep) sont ramenés par moyenne simple = identité.
    available_fc = [c for c in ("load_forecast", "wind_forecast", "solar_forecast") if c in data.columns]
    available_nb = [c for c in NEIGHBOUR_PRICE_FEATURES if c in data.columns]
    available_gas = [c for c in GAS_FEATURES if c in data.columns]
    available_nuc = [c for c in NUCLEAR_FEATURES if c in data.columns]
    national_cols = (
        available_fc + available_nb + available_gas + available_nuc
        + (["price"] if "price" in data.columns else [])
    )

    df, meteo_cols = _aggregate_meteo(data, meteo_mode)
    if national_cols:
        df_nat = data.groupby("time", as_index=False)[national_cols].mean()
        df = df.merge(df_nat, on="time", how="left")

    # Potentiel renouvelable = capacité installée par département × ressource météo
    # locale, sommé nationalement (cf. _renewable_potential). Calculé sur les lignes
    # brutes avant agrégation, donc indépendant de meteo_mode.
    potential = _renewable_potential(data)
    if potential is not None:
        df = df.merge(potential, on="time", how="left")

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month
    df["day_of_year"] = df["time"].dt.dayofyear
    # is_weekend = samedi/dimanche OU jour férié français : la consommation et le prix
    # se comportent comme un week-end les jours fériés, on fusionne les deux signaux.
    is_holiday = df["time"].dt.date.map(lambda d: d in _FR_HOLIDAYS)
    df["is_weekend"] = ((df["day_of_week"] >= 5) | is_holiday).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Sub-horaire : si on a au moins une timestamp dont la minute n'est pas 0, on est
    # au pas 15-min (ou plus fin). On ajoute alors le quart d'heure dans la journée
    # (entier + sin/cos) pour que le modèle puisse capter la variation intra-heure.
    minute = df["time"].dt.minute
    if (minute != 0).any():
        df["quarter_of_hour"] = (minute // 15).astype(int)
        minute_of_day = df["time"].dt.hour * 60 + minute
        df["minute_of_day_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
        df["minute_of_day_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)

    # Demande résiduelle prévue = load - (wind+solar). C'est ce qui fixe la techno marginale
    # (gaz, charbon, etc.) donc l'un des prédicteurs les plus directs du prix.
    if {"load_forecast", "wind_forecast", "solar_forecast"}.issubset(df.columns):
        df["net_load_forecast"] = (
            df["load_forecast"]
            - df["wind_forecast"].fillna(0.0)
            - df["solar_forecast"].fillna(0.0)
        )

    # Gradients pas-à-pas : la *rampe* d'une grandeur, pas son niveau. df est trié
    # par time, donc .diff() = variation sur un pas (1 h ou 15 min selon le jeu).
    grad_src = {
        "net_load_grad": "net_load_forecast",
        "load_grad": "load_forecast",
        "solar_grad": "solar_forecast",
        "wind_grad": "wind_forecast",
    }
    for grad_col, src_col in grad_src.items():
        if src_col in df.columns:
            df[grad_col] = df[src_col].diff()

    # Lags de prix : jointure sur timestamp décalé (robuste aux trous, contrairement
    # à un .shift() par nombre de lignes). price_lag_1d = prix au même créneau J-1.
    if "price" in df.columns:
        for lag_col, delta in (
            ("price_lag_1d", pd.Timedelta(days=1)),
            ("price_lag_7d", pd.Timedelta(days=7)),
        ):
            shifted = df[["time", "price"]].rename(columns={"price": lag_col}).copy()
            shifted["time"] = shifted["time"] + delta
            df = df.merge(shifted, on="time", how="left")

    if USE_NEG_FEATURES:
        df = _add_negative_price_features(df)

    if USE_NUCLEAR_DERIVED and "nuclear_planned_unavail" in df.columns:
        u = df[["time", "nuclear_planned_unavail"]]
        for col, delta in (("_u_1d", pd.Timedelta(days=1)), ("_u_2d", pd.Timedelta(days=2))):
            shifted = u.rename(columns={"nuclear_planned_unavail": col}).copy()
            shifted["time"] = shifted["time"] + delta
            df = df.merge(shifted, on="time", how="left")
        df["nuclear_unavail_delta_1d"] = df["nuclear_planned_unavail"] - df["_u_1d"]
        if {"net_load_forecast", "nuclear_gen_d2"}.issubset(df.columns):
            nuclear_avail = df["nuclear_gen_d2"] - (df["nuclear_planned_unavail"] - df["_u_2d"])
            df["thermal_residual"] = df["net_load_forecast"] - nuclear_avail
        df = df.drop(columns=["_u_1d", "_u_2d"])

    df = df.replace([np.inf, -np.inf], np.nan)
    # On droppe les pas de temps entièrement vides côté météo (how='all' : en mode
    # 'department', un seul département manquant ne doit pas tuer l'heure entière —
    # LightGBM gère les NaN, les autres modèles imputent par moyenne de colonne).
    if meteo_cols:
        df = df.dropna(subset=meteo_cols, how="all")
    df = df.dropna(subset=TEMPORAL_FEATURES)
    if drop_missing_price and "price" in df.columns:
        df = df.dropna(subset=["price"])
    return df.reset_index(drop=True)


def _build_model(model_type: str):
    if model_type == "Simple":
        return Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    if model_type == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    if model_type == "GradientBoosting":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(n_estimators=200, random_state=42)
    if model_type == "LightGBM":
        # Tuning prudent : régularisation modérée, learning rate fixe, early stopping
        # géré au moment du fit (cf. _fit_with_validation).
        return lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=63,
            min_data_in_leaf=50,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=5,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    if model_type == "LightGBM_hurdle":
        return HurdleLGBM()
    raise ValueError(f"model_type inconnu : {model_type}")


def _model_handles_nan(model_type: str) -> bool:
    # Seul LightGBM tolère les NaN dans X nativement ; pour les autres on impute.
    return model_type in ("LightGBM", "LightGBM_hurdle")


def _build_feature_matrix(df: pd.DataFrame, feature_cols: list[str], model_type: str):
    X = df[feature_cols].copy()
    if not _model_handles_nan(model_type):
        # Pour Ridge / RF / GBM sklearn : on impute les prévisions manquantes par la moyenne
        # de la colonne (toutes les autres features sont déjà sans NaN à ce stade).
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].mean())
    return X.values


def _fit_with_validation(model, model_type, X_train, y_train, X_val, y_val):
    if model_type == "LightGBM_hurdle":
        return model.fit(X_train, y_train, X_val, y_val)
    if model_type == "LightGBM":
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
    else:
        model.fit(X_train, y_train)
    return model


# ----------------------------------------------------------------------
# Prix négatifs : features et modèle à deux étages
# ----------------------------------------------------------------------
NEG_THRESHOLD = 0.0  # régime "négatif" = prix <= 0 €/MWh


def _add_negative_price_features(df: pd.DataFrame) -> pd.DataFrame:
    # Cf. NEG_FEATURES. df est trié par time, une ligne par pas de temps.
    if {"wind_forecast", "solar_forecast", "load_forecast"}.issubset(df.columns):
        renewables = df["wind_forecast"].fillna(0.0) + df["solar_forecast"].fillna(0.0)
        df["renewable_share"] = renewables / df["load_forecast"]

    dates = df["time"].dt.normalize()
    day = pd.Series(pd.to_datetime(dates.unique()))
    is_holiday = day.dt.date.map(lambda d: d in _FR_HOLIDAYS)
    prev_hol = (day - pd.Timedelta(days=1)).dt.date.map(lambda d: d in _FR_HOLIDAYS)
    next_hol = (day + pd.Timedelta(days=1)).dt.date.map(lambda d: d in _FR_HOLIDAYS)
    # Pont : vendredi après un jeudi férié, ou lundi avant un mardi férié.
    bridge = ((day.dt.dayofweek == 4) & prev_hol) | ((day.dt.dayofweek == 0) & next_hol)
    bridge &= ~is_holiday
    df["is_bridge"] = dates.map(dict(zip(day, bridge.astype(int)))).astype(int)

    if "solar_forecast" in df.columns:
        offday = (df["is_weekend"] == 1) | (df["is_bridge"] == 1)
        df["solar_x_offday"] = df["solar_forecast"] * offday

    if "price" in df.columns:
        daily = df.groupby(dates)["price"].agg(
            neg_hours=lambda p: float((p <= NEG_THRESHOLD).sum()) if p.notna().any() else np.nan,
            price_min="min",
        )
        for lag in (1, 7):
            shifted = daily["neg_hours"].copy()
            shifted.index = shifted.index + pd.Timedelta(days=lag)
            df[f"neg_hours_d{lag}"] = dates.map(shifted).values
        shifted = daily["price_min"].copy()
        shifted.index = shifted.index + pd.Timedelta(days=1)
        df["price_min_d1"] = dates.map(shifted).values
    return df


class HurdleLGBM:
    """Modèle à deux régimes pour les prix négatifs.

    Un régresseur L2 fait la moyenne entre régimes : il place vers +10 € des heures
    qui finissent à 0 ou en dessous. On sépare donc :
      - un classifieur p(x) = P(prix <= 0 | x) ;
      - un régresseur sur le régime normal (prix > 0) ;
      - un régresseur sur le régime négatif (petit modèle, peu d'exemples).
    Prévision = p · E[prix | négatif] + (1 - p) · E[prix | normal] : l'espérance,
    qui minimise l'erreur quadratique comme le modèle d'origine.
    """

    def __init__(self, threshold: float = NEG_THRESHOLD, min_neg: int = 150):
        self.threshold = threshold
        self.min_neg = min_neg

    @staticmethod
    def _es(X_val, y_val):
        if X_val is None or len(X_val) == 0 or len(np.unique(y_val)) < 2:
            return {}
        return dict(eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

    def fit(self, X, y, X_val=None, y_val=None):
        y = np.asarray(y)
        neg = y <= self.threshold
        has_val = X_val is not None and len(X_val) > 0
        if has_val:
            y_val = np.asarray(y_val)
            neg_val = y_val <= self.threshold

        # Probabilités non rééquilibrées (pas de class_weight) : p doit être calibré,
        # puisqu'il pondère l'espérance.
        self.clf_ = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.03, num_leaves=31, min_data_in_leaf=30,
            feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        es_clf = self._es(X_val, neg_val.astype(int)) if has_val else {}
        self.clf_.fit(X, neg.astype(int), **es_clf)

        self.pos_ = lgb.LGBMRegressor(
            n_estimators=2000, learning_rate=0.03, num_leaves=63, min_data_in_leaf=50,
            feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        es_pos = self._es(X_val[~neg_val], y_val[~neg_val]) if has_val else {}
        self.pos_.fit(X[~neg], y[~neg], **es_pos)

        self.neg_ = None
        self.neg_value_ = float(np.median(y[neg])) if neg.any() else 0.0
        if neg.sum() >= self.min_neg:
            self.neg_ = lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.05, num_leaves=15, min_data_in_leaf=20,
                random_state=42, n_jobs=-1, verbose=-1,
            )
            self.neg_.fit(X[neg], y[neg])
        return self

    def predict_proba_neg(self, X) -> np.ndarray:
        return self.clf_.predict_proba(X)[:, 1]

    def predict(self, X) -> np.ndarray:
        p = self.predict_proba_neg(X)
        neg = self.neg_.predict(X) if self.neg_ is not None else self.neg_value_
        return p * neg + (1.0 - p) * self.pos_.predict(X)
