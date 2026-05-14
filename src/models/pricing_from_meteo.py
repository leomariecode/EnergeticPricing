from pathlib import Path
import holidays
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_data


_FR_HOLIDAYS = holidays.country_holidays("FR")


METEO_FEATURES = [
    "temperature_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "cloud_cover",
    "precipitation",
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

MODEL_DIR = Path("results/models")


def model_learn(data: pd.DataFrame, model_type: str, test_size: float = 0.2):
    # Cache : si un modèle a déjà été entraîné sur la même plage et le même type, on le recharge.
    print(f"[model_learn] Démarrage : model_type={model_type}, {len(data)} lignes en entrée")
    start = pd.to_datetime(data["time"]).min().strftime("%Y%m%d")
    end = pd.to_datetime(data["time"]).max().strftime("%Y%m%d")
    print(f"[model_learn] Plage temporelle : {start} -> {end}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = MODEL_DIR / f"pricing_from_meteo_{model_type}_{start}_{end}.pkl"

    if cache_path.exists():
        print(f"[model_learn] Modèle déjà entraîné, rechargement : {cache_path}")
        bundle = joblib.load(cache_path)
        if "scores" in bundle:
            _print_scores(bundle)
        return bundle

    print("[model_learn] Préparation des features (drop des heures sans prix réel)")
    df = _prepare_features(data, drop_missing_price=True)
    feature_cols = METEO_FEATURES + TEMPORAL_FEATURES
    print(f"[model_learn] Features prêtes : {len(df)} lignes, {len(feature_cols)} colonnes")

    X = df[feature_cols].values
    y = df["price"].values

    # Split temporel : on garde les dernières {test_size*100}% comme test (pas de mélange,
    # pour évaluer la généralisation à des dates futures).
    n_test = max(1, int(len(df) * test_size))
    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train, y_test = y[:-n_test], y[-n_test:]
    time_train_end = df["time"].iloc[-n_test - 1]
    time_test_start = df["time"].iloc[-n_test]
    time_test_end = df["time"].iloc[-1]

    print(f"[model_learn] Entraînement {model_type} sur {len(X_train)} heures (test : {len(X_test)})")
    model = _build_model(model_type)
    model.fit(X_train, y_train)
    scores = _compute_scores(
        model, X_train, y_train, X_test, y_test,
        time_train_end=time_train_end,
        time_test_start=time_test_start,
        time_test_end=time_test_end,
    )

    # Ré-entraînement sur l'intégralité du jeu de données : le modèle servi en prédiction
    # bénéficie de toutes les heures disponibles, les scores ci-dessus restent l'évaluation honnête.
    print("[model_learn] Ré-entraînement sur l'intégralité du jeu de données")
    model = _build_model(model_type)
    model.fit(X, y)

    bundle = {
        "model": model,
        "model_type": model_type,
        "feature_cols": feature_cols,
        "start": start,
        "end": end,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_total": int(len(X)),
        "scores": scores,
    }
    _print_scores(bundle)
    joblib.dump(bundle, cache_path)
    print(f"[model_learn] Modèle sauvegardé : {cache_path}")
    return bundle


def _compute_scores(model, X_train, y_train, X_test, y_test, **info) -> dict:
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    scores = {
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
        },
    }
    scores.update({k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in info.items()})
    return scores


def _print_scores(bundle: dict) -> None:
    s = bundle.get("scores") or {}
    train = s.get("train", {})
    test = s.get("test", {})
    print("=" * 72)
    print(f"[model_score] Modèle '{bundle.get('model_type')}' — caractéristiques :")
    print(f"  - Plage entraînement   : {bundle.get('start')} -> {bundle.get('end')}")
    print(f"  - Features ({len(bundle.get('feature_cols') or [])}) : {bundle.get('feature_cols')}")
    if "time_test_start" in s:
        print(f"  - Période test         : {s.get('time_test_start')} -> {s.get('time_test_end')}")
    print(f"  - Heures totales       : {bundle.get('n_total')}  (train={bundle.get('n_train')}, test={bundle.get('n_test')})")
    if train:
        print(
            f"  - Train  R²={train['r2']:.3f}  MAE={train['mae']:.2f} €/MWh  "
            f"RMSE={train['rmse']:.2f} €/MWh  (n={train['n']})"
        )
    if test:
        print(
            f"  - Test   R²={test['r2']:.3f}  MAE={test['mae']:.2f} €/MWh  "
            f"RMSE={test['rmse']:.2f} €/MWh  (n={test['n']})"
        )
    print("=" * 72)


def model_score(bundle: dict, data: pd.DataFrame | None = None, test_size: float = 0.2) -> dict:
    # Si data est fourni, recalcule les scores sur ce jeu (utile pour évaluer un modèle en cache
    # sur de nouvelles données). Sinon, renvoie les scores stockés dans le bundle.
    if data is None:
        _print_scores(bundle)
        return bundle.get("scores") or {}

    df = _prepare_features(data, drop_missing_price=True)
    feature_cols = bundle["feature_cols"]
    X = df[feature_cols].values
    y = df["price"].values
    n_test = max(1, int(len(df) * test_size))
    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train, y_test = y[:-n_test], y[-n_test:]
    scores = _compute_scores(
        bundle["model"], X_train, y_train, X_test, y_test,
        time_train_end=df["time"].iloc[-n_test - 1],
        time_test_start=df["time"].iloc[-n_test],
        time_test_end=df["time"].iloc[-1],
    )
    eval_bundle = {**bundle, "scores": scores, "n_train": len(X_train),
                   "n_test": len(X_test), "n_total": len(X)}
    _print_scores(eval_bundle)
    return scores


def predict_price_from_meteo(model_bundle: dict, date) -> pd.Series:
    # date : str "YYYY-MM-DD" ou pd.Timestamp -> on prédit les 24h de cette journée
    print(f"[predict_price_from_meteo] Démarrage prédiction pour {date}")
    date = pd.to_datetime(date).strftime("%Y-%m-%d")
    next_date = (pd.to_datetime(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"[predict_price_from_meteo] Chargement des données {date} -> {next_date}")
    data = load_data(date, next_date)
    return predict_price_from_data(model_bundle, data)


def predict_price_from_data(model_bundle: dict, data: pd.DataFrame) -> pd.Series:
    # Prédit à partir d'un DataFrame météo déjà chargé (mêmes colonnes que load_data).
    print(f"[predict_price_from_data] {len(data)} lignes en entrée")
    df = _prepare_features(data)

    feature_cols = model_bundle["feature_cols"]
    model = model_bundle["model"]

    print(f"[predict_price_from_data] Prédiction sur {len(df)} heures")
    X = df[feature_cols].values
    preds = model.predict(X)
    print(f"[predict_price_from_data] Prédiction terminée (moyenne={preds.mean():.2f} EUR/MWh)")

    return pd.Series(preds, index=df["time"], name="predicted_price")


def _prepare_features(data: pd.DataFrame, drop_missing_price: bool = False) -> pd.DataFrame:
    # Agrégation nationale : moyenne des features météo sur les 95 départements pour chaque heure.
    # Le prix est dupliqué sur tous les départements -> mean() le rebascule en valeur unique horaire.
    cols_to_agg = METEO_FEATURES + (["price"] if "price" in data.columns else [])
    df = data.groupby("time", as_index=False)[cols_to_agg].mean()

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

    df = df.replace([np.inf, -np.inf], np.nan)
    # On ne droppe que sur les colonnes requises par le modèle.
    # Le prix peut être NaN (heures récentes non publiées par ENTSO-E) : on garde ces lignes
    # pour pouvoir prédire, sauf en entraînement où drop_missing_price=True.
    df = df.dropna(subset=METEO_FEATURES + TEMPORAL_FEATURES)
    if drop_missing_price and "price" in df.columns:
        df = df.dropna(subset=["price"])
    return df.reset_index(drop=True)


def _build_model(model_type: str):
    if model_type == "Simple":
        return Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    if model_type == "RandomForest":
        return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    if model_type == "GradientBoosting":
        return GradientBoostingRegressor(n_estimators=200, random_state=42)
    raise ValueError(f"model_type inconnu : {model_type}")
