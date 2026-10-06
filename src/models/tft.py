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

import time

import numpy as np
import pandas as pd

from src.models.pricing_from_meteo import (
    CAPACITY_FEATURES,
    ENTSOE_FORECAST_FEATURES,
    GAS_FEATURES,
    GRADIENT_FEATURES,
    LAG_FEATURES,
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

    class _EpochLogger(pl.Callback):
        """Callback de logging : trace chaque époque (pertes, chrono, ETA).

        Le Trainer tourne avec logger=False, donc sans ce callback on n'a aucune
        trace persistante de l'entraînement (juste la barre de progression, qui
        s'efface). Ici on imprime une ligne par époque, lisible aussi dans un
        notebook ou un fichier de log.
        """

        def __init__(self, max_epochs: int):
            self.max_epochs = max_epochs
            self._fit_t0 = 0.0
            self._epoch_t0 = 0.0
            self._epoch_secs: list[float] = []

        @staticmethod
        def _metric(trainer, key: str) -> float:
            v = trainer.callback_metrics.get(key)
            try:
                return float(v)
            except (TypeError, ValueError):
                return float("nan")

        def on_fit_start(self, trainer, pl_module):
            self._fit_t0 = time.perf_counter()
            print(f"[learn_tft] >>> Entraînement démarré ({self.max_epochs} époques max)")

        def on_train_epoch_start(self, trainer, pl_module):
            self._epoch_t0 = time.perf_counter()

        def on_validation_epoch_end(self, trainer, pl_module):
            # Appelé après chaque validation ; on saute la sanity-check du début.
            if trainer.sanity_checking:
                return
            now = time.perf_counter()
            epoch = trainer.current_epoch + 1
            dt = now - self._epoch_t0
            self._epoch_secs.append(dt)
            elapsed = now - self._fit_t0
            avg = sum(self._epoch_secs) / len(self._epoch_secs)
            eta = avg * max(0, self.max_epochs - epoch)
            train_loss = self._metric(trainer, "train_loss_epoch")
            if np.isnan(train_loss):  # pas encore agrégé à la 1re époque
                train_loss = self._metric(trainer, "train_loss_step")
            val_loss = self._metric(trainer, "val_loss")
            print(
                f"[learn_tft] Époque {epoch:>3}/{self.max_epochs} | "
                f"train_loss={train_loss:7.4f} val_loss={val_loss:7.4f} | "
                f"{dt:5.1f}s/époque | écoulé {elapsed/60:5.1f} min | "
                f"ETA ~{eta/60:5.1f} min",
                flush=True,
            )

        def on_fit_end(self, trainer, pl_module):
            total = time.perf_counter() - self._fit_t0
            n = trainer.current_epoch + 1
            print(
                f"[learn_tft] <<< Entraînement terminé : {n} époques en "
                f"{total/60:.1f} min ({total/max(n,1):.1f}s/époque en moyenne)",
                flush=True,
            )

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
# Variante "rapide" (fast=True) — modèle plus léger pour les longs historiques
# (≈10 ans) et le CPU : ~4x moins de paramètres, donc un temps/époque très
# inférieur, au prix d'un peu de capacité. À coupler avec meteo_mode="national".
_TFT_PARAMS_FAST = dict(
    learning_rate=0.05,
    hidden_size=16,
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=8,
)
_MAX_EPOCHS = 40
_BATCH_SIZE = 128
_EARLYSTOP_PATIENCE = 6

# Taille (en jours) de la fenêtre de contexte de l'encodeur, selon le mode.
# Le mode rapide raccourcit le contexte (3 j au lieu de 7) -> encodeur LSTM
# plus court, donc moins de calcul par batch.
_ENCODER_DAYS = 7
_ENCODER_DAYS_FAST = 3
_FAST_BATCH_SIZE = 256  # batch plus large en mode rapide : moins d'overhead Python


def _build_long_df(data: pd.DataFrame, meteo_mode: str, drop_missing_price: bool):
    # Réutilise toute la préparation de features du pipeline tabulaire, puis met
    # le DataFrame au format long attendu par pytorch-forecasting (time_idx + group).
    df = _prepare_features(data, drop_missing_price=drop_missing_price, meteo_mode=meteo_mode)
    df = df.sort_values("time").reset_index(drop=True)

    meteo_cols = _meteo_cols_of(df)
    fc = _select_available(df, ENTSOE_FORECAST_FEATURES)
    nb = []  # prix voisins exclus : inconnus en prévision J-1 (cf. model_learn)
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


def learn_tft(
    data: pd.DataFrame,
    meteo_mode: str = "department",
    n_splits: int = 5,
    fast: bool = True,
) -> dict:
    # Entraîne un Temporal Fusion Transformer. n_splits sert ici à dimensionner la
    # taille du holdout temporel d'évaluation (pas un vrai CV : trop coûteux en TFT).
    #
    # fast=True (défaut) : mode rapide, pensé pour les longs historiques (≈10 ans)
    # et le CPU. Il force la météo NATIONALE (≈30 covariables au lieu de ≈500 en
    # mode 'department'), raccourcit la fenêtre de contexte (cf. _ENCODER_DAYS_FAST),
    # élargit le batch et allège le modèle (cf. _TFT_PARAMS_FAST). Plusieurs fois
    # plus rapide par époque. fast=False rétablit le TFT "pleine capacité".
    _require_tft()

    if fast and meteo_mode == "department":
        print("[learn_tft] Mode rapide -> météo forcée en 'national' "
              "(au lieu de 'department', ~500 covariables -> ~30)")
        meteo_mode = "national"
    params = _TFT_PARAMS_FAST if fast else _TFT_PARAMS
    encoder_days = _ENCODER_DAYS_FAST if fast else _ENCODER_DAYS
    batch_size = _FAST_BATCH_SIZE if fast else _BATCH_SIZE

    print(f"[learn_tft] Démarrage TFT — fast={fast}, meteo_mode={meteo_mode}, "
          f"{len(data)} lignes")

    freq_tag = _infer_freq_tag(data["time"])
    steps_per_day = 96 if freq_tag == "15min" else 24
    max_encoder_length = encoder_days * steps_per_day  # fenêtre de contexte
    max_prediction_length = steps_per_day              # horizon : 1 jour

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
    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_batches = int(np.ceil(n_train / batch_size))
    print(
        f"[learn_tft] {n_train} fenêtres d'entraînement, {n_val} de validation | "
        f"batch_size={batch_size} -> {n_batches} batches/époque"
    )

    tft = TemporalFusionTransformer.from_dataset(
        training, loss=QuantileLoss(), log_interval=0, **params,
    )
    print(
        f"[learn_tft] TFT instancié : {sum(p.numel() for p in tft.parameters()):,} "
        f"paramètres, accelerator=auto"
    )
    trainer = pl.Trainer(
        max_epochs=_MAX_EPOCHS,
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=_EARLYSTOP_PATIENCE, mode="min"),
            _EpochLogger(_MAX_EPOCHS),
        ],
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
        "fast": bool(fast),
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
    # pytorch-forecasting 1.x : predict() renvoie un namedtuple Prediction
    # (output, x, index, decoder_lengths, y) -> on accède aux champs par nom.
    prediction = model.predict(loader, mode="prediction", return_index=True)
    raw = np.asarray(prediction.output)  # (n_fenetres, horizon)
    index = prediction.index

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
