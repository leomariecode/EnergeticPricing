"""Chemin TFT (Temporal Fusion Transformer) pour le pricing.

Le TFT modélise explicitement la *séquence* temporelle (contrairement aux modèles
tabulaires de pricing_from_meteo.py qui voient chaque pas de temps isolément). Il
est donc surtout intéressant pour les rampes et les pics intra-horaires.

Dépendances lourdes — non installées par défaut :
    pip install torch pytorch-forecasting lightning

NB : sur Mac Intel (x86_64) + Python 3.13, il n'existe pas de wheel PyTorch
(PyTorch ne publie plus de build macOS x86_64 depuis la 2.2.2, et la 2.2.2 n'a
pas de wheel Python 3.13). Utiliser un environnement Python 3.12, ou une machine
Linux / Apple Silicon.

Ce module est importé paresseusement par pricing_from_meteo.model_learn quand
model_type == "TFT". Si les dépendances manquent, learn_tft/predict_tft lèvent
une RuntimeError explicite plutôt que de casser l'import du reste du projet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.pricing_from_meteo import (
    CAPACITY_FEATURES,
    ENTSOE_FORECAST_FEATURES,
    GAS_FEATURES,
    GRADIENT_FEATURES,
    LAG_FEATURES,
    NEIGHBOUR_PRICE_FEATURES,
    SUBHOURLY_TEMPORAL_FEATURES,
    TEMPORAL_FEATURES,
    _infer_freq_tag,
    _meteo_cols_of,
    _prepare_features,
    _print_scores,
    _select_available,
)

_INSTALL_HINT = (
    "Le modèle TFT requiert PyTorch + pytorch-forecasting + lightning.\n"
    "  pip install torch pytorch-forecasting lightning\n"
    "NB : pas de wheel torch pour Mac Intel + Python 3.13 -> utiliser un env "
    "Python 3.12, ou une machine Linux / Apple Silicon."
)

try:  # imports lourds, optionnels
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss

    _TFT_AVAILABLE = True
    _IMPORT_ERROR = None
except Exception as exc:  # ImportError, ou erreur de compat
    _TFT_AVAILABLE = False
    _IMPORT_ERROR = exc


def tft_available() -> bool:
    return _TFT_AVAILABLE


def _require_tft() -> None:
    if not _TFT_AVAILABLE:
        raise RuntimeError(f"{_INSTALL_HINT}\n(import échoué : {_IMPORT_ERROR})")


# Hyper-paramètres TFT — volontairement modestes (jeu de données ~1-2 ans).
_TFT_PARAMS = dict(
    learning_rate=0.03,
    hidden_size=32,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=16,
)
_MAX_EPOCHS = 40
_BATCH_SIZE = 128
_EARLYSTOP_PATIENCE = 6


def _build_long_df(data: pd.DataFrame, meteo_mode: str, drop_missing_price: bool):
    # Réutilise toute la préparation de features du pipeline tabulaire, puis met
    # le DataFrame au format long attendu par pytorch-forecasting (time_idx + group).
    df = _prepare_features(data, drop_missing_price=drop_missing_price, meteo_mode=meteo_mode)
    df = df.sort_values("time").reset_index(drop=True)

    meteo_cols = _meteo_cols_of(df)
    fc = _select_available(df, ENTSOE_FORECAST_FEATURES)
    nb = _select_available(df, NEIGHBOUR_PRICE_FEATURES)
    gas = _select_available(df, GAS_FEATURES)
    cap = _select_available(df, CAPACITY_FEATURES)
    grad = _select_available(df, GRADIENT_FEATURES)
    lag = _select_available(df, LAG_FEATURES)
    subhourly = [c for c in SUBHOURLY_TEMPORAL_FEATURES if c in df.columns]
    feature_cols = meteo_cols + fc + nb + gas + cap + grad + lag + TEMPORAL_FEATURES + subhourly

    # TFT n'accepte pas les NaN dans les covariables : on impute par la médiane.
    for c in feature_cols:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # time_idx entier consécutif + un identifiant de série unique (on a une seule série FR).
    df["time_idx"] = np.arange(len(df), dtype=int)
    df["series"] = "FR"
    return df, feature_cols


def _make_dataset(df: pd.DataFrame, feature_cols, max_encoder_length, max_prediction_length):
    # Toutes les covariables sont déclarées "known reals" (météo/prévisions/calendrier
    # sont connus à l'avance ; les prix voisins le sont aussi en couplage day-ahead).
    # Le prix est la cible, déclarée "unknown real".
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="price",
        group_ids=["series"],
        max_encoder_length=max_encoder_length,
        min_encoder_length=max_encoder_length // 2,
        max_prediction_length=max_prediction_length,
        min_prediction_length=1,
        static_categoricals=["series"],
        time_varying_known_reals=["time_idx"] + list(feature_cols),
        time_varying_unknown_reals=["price"],
        target_normalizer=GroupNormalizer(groups=["series"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )


def learn_tft(data: pd.DataFrame, meteo_mode: str = "department", n_splits: int = 5) -> dict:
    # Entraîne un Temporal Fusion Transformer. n_splits sert ici à dimensionner la
    # taille du holdout temporel d'évaluation (pas un vrai CV : trop coûteux en TFT).
    _require_tft()
    print(f"[learn_tft] Démarrage TFT — meteo_mode={meteo_mode}, {len(data)} lignes")

    freq_tag = _infer_freq_tag(data["time"])
    steps_per_day = 96 if freq_tag == "15min" else 24
    max_encoder_length = 7 * steps_per_day      # 1 semaine de contexte
    max_prediction_length = steps_per_day       # horizon : 1 jour

    df, feature_cols = _build_long_df(data, meteo_mode, drop_missing_price=True)
    print(f"[learn_tft] {len(df)} pas de temps, {len(feature_cols)} covariables, pas={freq_tag}")
    if len(df) < max_encoder_length + 2 * max_prediction_length:
        raise ValueError("[learn_tft] Jeu de données trop court pour entraîner un TFT.")

    # Holdout temporel : les derniers jours servent de validation.
    val_steps = max(max_prediction_length, n_splits * max_prediction_length)
    training_cutoff = int(df["time_idx"].max()) - val_steps

    training = _make_dataset(
        df[df["time_idx"] <= training_cutoff],
        feature_cols, max_encoder_length, max_prediction_length,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True,
    )
    train_loader = training.to_dataloader(train=True, batch_size=_BATCH_SIZE, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=_BATCH_SIZE, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training, loss=QuantileLoss(), log_interval=0, **_TFT_PARAMS,
    )
    trainer = pl.Trainer(
        max_epochs=_MAX_EPOCHS,
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=_EARLYSTOP_PATIENCE, mode="min")],
        enable_progress_bar=True,
        logger=False,
        enable_checkpointing=False,
    )
    print(f"[learn_tft] Entraînement (max {_MAX_EPOCHS} époques, early stopping)...")
    trainer.fit(tft, train_loader, val_loader)

    # Évaluation sur le holdout : MAE / RMSE / R² sur la prédiction médiane.
    raw = tft.predict(val_loader, mode="prediction")
    y_pred = np.asarray(raw).reshape(-1)
    y_true = np.concatenate([y[0].numpy().reshape(-1) for _, y in iter(val_loader)])
    m = min(len(y_pred), len(y_true))
    y_pred, y_true = y_pred[:m], y_true[:m]
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    scores = {"test": {"r2": r2, "mae": mae, "rmse": rmse, "n": m}}

    bundle = {
        "model": tft,
        "model_type": "TFT",
        "feature_cols": feature_cols,
        "meteo_mode": meteo_mode,
        "freq": freq_tag,
        "start": pd.to_datetime(data["time"]).min().strftime("%Y%m%d"),
        "end": pd.to_datetime(data["time"]).max().strftime("%Y%m%d"),
        "n_total": int(len(df)),
        "n_splits": int(n_splits),
        "scores": scores,
        "spike_model": None,  # le TFT donne déjà des quantiles ; pas de classifieur séparé ici
        "tft_training_dataset": training,
        "max_encoder_length": max_encoder_length,
        "max_prediction_length": max_prediction_length,
    }
    _print_scores(bundle)
    print(f"[learn_tft] Terminé — holdout : R²={r2:.3f}  MAE={mae:.2f}  RMSE={rmse:.2f}")
    return bundle


def predict_tft(model_bundle: dict, data: pd.DataFrame) -> pd.Series:
    # Prédit le prix sur la série fournie. On reconstruit un dataset à partir du
    # dataset d'entraînement stocké dans le bundle, on prédit toutes les fenêtres
    # décodables et on renvoie une Série indexée par 'time'.
    _require_tft()
    print(f"[predict_tft] {len(data)} lignes en entrée")

    df, _ = _build_long_df(data, model_bundle["meteo_mode"], drop_missing_price=False)
    training = model_bundle["tft_training_dataset"]
    model = model_bundle["model"]

    # predict=False -> toutes les fenêtres glissantes ; on récupère l'index temporel.
    dataset = TimeSeriesDataSet.from_dataset(
        training, df, predict=False, stop_randomization=True,
    )
    loader = dataset.to_dataloader(train=False, batch_size=_BATCH_SIZE, num_workers=0)
    raw, index = model.predict(loader, mode="prediction", return_index=True)
    raw = np.asarray(raw)  # (n_fenetres, horizon)

    # Chaque fenêtre démarre au time_idx donné par 'index' ; on rabat la prédiction
    # sur les time_idx correspondants. En cas de recouvrement, on garde la dernière.
    pred_by_idx: dict[int, float] = {}
    horizon = raw.shape[1] if raw.ndim == 2 else 1
    for row, start_idx in zip(raw, index["time_idx"].to_numpy()):
        for h in range(horizon):
            pred_by_idx[int(start_idx) + h] = float(np.atleast_1d(row)[h])

    idx_to_time = dict(zip(df["time_idx"], pd.to_datetime(df["time"])))
    times, values = [], []
    for tidx, val in sorted(pred_by_idx.items()):
        if tidx in idx_to_time:
            times.append(idx_to_time[tidx])
            values.append(val)

    print(f"[predict_tft] {len(values)} pas prédits")
    return pd.Series(values, index=pd.DatetimeIndex(times, name="time"), name="predicted_price")
