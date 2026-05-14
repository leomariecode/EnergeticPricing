import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
from sklearn.model_selection import train_test_split

ENERGY_SOURCES = ["Fioul", "Charbon", "Gaz", "Eolien", "Solaire", "Nucleaire"]
LAGS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [6, 12, 24, 168]


def pricing_model(data_RTE, data_price, pricing_model_type):
    df = prepare_energy_pricing_from_mix_features(data_RTE, data_price)

    if pricing_model_type == "Simple":
        return simple_model(df)


def simple_model(df):
    target_col = "Price (EUR/MWhe)"
    exclude = [target_col, "Date"]
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].values
    y = df[target_col].values
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model, feature_cols


def _add_temporal_features(df):
    df["hour"] = df["Date"].dt.hour
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["day_of_year"] = df["Date"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["year"] = df["Date"].dt.year
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def _add_production_features(df):
    df["total_production"] = sum(df[s] for s in ENERGY_SOURCES)
    df["part_renouvelable"] = (df["Eolien"] + df["Solaire"]) / df["total_production"]
    df["part_fossile"] = (df["Fioul"] + df["Charbon"] + df["Gaz"]) / df["total_production"]
    df["part_nucleaire"] = df["Nucleaire"] / df["total_production"]
    df["ratio_fossile_renouvelable"] = (
        (df["Fioul"] + df["Charbon"] + df["Gaz"]) / (df["Eolien"] + df["Solaire"] + 1)
    )
    return df


def _add_lag_and_rolling_features(df):
    for lag in LAGS:
        df[f"price_lag_{lag}h"] = df["Price (EUR/MWhe)"].shift(lag)
        df[f"total_prod_lag_{lag}h"] = df["total_production"].shift(lag)
    for window in ROLLING_WINDOWS:
        df[f"price_rolling_{window}h"] = (
            df["Price (EUR/MWhe)"].rolling(window, min_periods=1).mean()
        )
    return df


def prepare_energy_pricing_from_mix_features(df_prod: pd.DataFrame, df_price: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare les features pour l'entraînement du modèle de pricing.

    Colonnes attendues dans df_prod :
        Date, Consommation, Fioul, Charbon, Gaz, Eolien, Solaire, Nucleaire

    Colonnes attendues dans df_price :
        Datetime, Price (EUR/MWhe)
    """
    df = df_prod.merge(df_price, left_on="Date", right_on="Datetime", how="inner")
    df = df[["Date", "Consommation"] + ENERGY_SOURCES + ["Price (EUR/MWhe)"]].copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df = _add_temporal_features(df)
    df = _add_production_features(df)
    df = _add_lag_and_rolling_features(df)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df


def predict_price_from_mix(model, feature_cols, energy_consumption):
    """
    Prédit le prix de l'électricité à partir d'un mix énergétique.

    Parameters
    ----------
    model : trained sklearn model
    feature_cols : list[str]
        Les colonnes de features utilisées à l'entraînement (retournées par simple_model).
    energy_consumption : pd.DataFrame
        DataFrame contenant l'historique récent + la ligne à prédire.
        Doit contenir au minimum 168 lignes (1 semaine) pour calculer les rolling features.
        La prédiction est faite sur la DERNIÈRE ligne.

        Colonnes requises :
        ┌─────────────────┬──────────────────────────────────────────────────┐
        │ Colonne         │ Description                                      │
        ├─────────────────┼──────────────────────────────────────────────────┤
        │ Date            │ datetime de l'heure (ex: "2023-06-15 14:00:00") │
        │ Consommation    │ Consommation en MW                               │
        │ Fioul           │ Production fioul en MW                           │
        │ Charbon         │ Production charbon en MW                         │
        │ Gaz             │ Production gaz en MW                             │
        │ Eolien          │ Production éolien en MW                          │
        │ Solaire         │ Production solaire en MW                         │
        │ Nucleaire       │ Production nucléaire en MW                       │
        │ Price (EUR/MWhe)│ Prix historique (pour les lags/rolling)          │
        └─────────────────┴──────────────────────────────────────────────────┘

        Note : le prix de la dernière ligne peut être NaN (c'est ce qu'on prédit),
        mais les lignes précédentes doivent avoir un prix renseigné pour les lags.
    """
    df = energy_consumption.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df = _add_temporal_features(df)
    df = _add_production_features(df)
    df = _add_lag_and_rolling_features(df)

    df = df.replace([np.inf, -np.inf], np.nan)

    last_row = df.iloc[[-1]][feature_cols]
    return model.predict(last_row.values)[0]