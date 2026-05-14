import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.loader import load_data
from src.models.pricing_from_meteo import model_learn, model_score, predict_price_from_data


class EnergyMarketV2:

    def __init__(self, start, end):
        self.data = None
        self.model = None
        self.model_type = None
        self.start = start
        self.end = end

    def initialize(self):
        print(f"[EnergyMarketV2.initialize] start={self.start}, end={self.end}")
        self.data = load_data(self.start, self.end)
        print(f"[EnergyMarketV2.initialize] Données chargées : {len(self.data)} lignes")

    def learn(self, model_type):
        print(f"[EnergyMarketV2.learn] Apprentissage modèle {model_type}")
        self.model_type = model_type
        self.model = model_learn(self.data, model_type)
        print("[EnergyMarketV2.learn] Modèle prêt")

    def score(self, recompute_on_data: bool = False):
        # Affiche les scores du modèle. Par défaut, ceux stockés au moment du fit.
        if self.model is None:
            raise RuntimeError("Aucun modèle entraîné : appeler learn() d'abord.")
        data = self.data if recompute_on_data else None
        return model_score(self.model, data=data)

    def predict_from_meteo(self, date):
        # Prédit les 24h d'une journée. Compare au prix réel quand disponible,
        # heure par heure : aucune heure n'est moyennée ni interpolée.
        print(f"[EnergyMarketV2.predict_from_meteo] date={date}")
        if self.data is None:
            raise RuntimeError("self.data est None : appeler initialize() d'abord.")
        if self.model is None:
            raise RuntimeError("self.model est None : appeler learn() d'abord.")

        date_ts = pd.to_datetime(date).normalize()
        start_ts = pd.to_datetime(self.start).normalize()
        end_ts = pd.to_datetime(self.end).normalize()
        if not (start_ts <= date_ts <= end_ts):
            raise ValueError(
                f"date {date_ts.date()} hors plage [{start_ts.date()}, {end_ts.date()}]"
            )

        times = pd.to_datetime(self.data["time"])
        mask = times.dt.normalize() == date_ts
        day_data = self.data.loc[mask]
        if day_data.empty:
            raise ValueError(
                f"Aucune donnée météo pour {date_ts.date()} dans self.data. "
                "Vérifie que la plage [start, end] couvre cette date."
            )

        predicted = predict_price_from_data(self.model, day_data)

        # Le prix réel ENTSO-E peut être manquant pour les heures les plus récentes :
        # on ne le construit que sur les heures où il existe vraiment (NaN exclus).
        real = None
        if "price" in day_data.columns:
            real_df = (
                day_data.drop_duplicates(subset="time")
                .loc[:, ["time", "price"]]
                .dropna(subset=["price"])
                .sort_values("time")
            )
            if not real_df.empty:
                real = real_df.set_index("time")["price"]

        self._plot_prices(date_ts, real, predicted)
        self._print_comparison(date_ts, real, predicted)
        return predicted, real

    def _plot_prices(self, date_ts, real, predicted):
        fig, ax = plt.subplots(figsize=(11, 5))

        ax.plot(
            predicted.index, predicted.values,
            marker="x", markersize=7, linestyle="--", linewidth=1.8,
            color="#E67E22", label="Prix prédit (modèle)",
        )
        if real is not None and not real.empty:
            ax.plot(
                real.index, real.values,
                marker="o", markersize=6, linestyle="-", linewidth=2,
                color="#2E86AB", label="Prix réel (ENTSO-E)",
            )
            title = f"Prix horaire le {date_ts.strftime('%A %d %B %Y')} — réel vs prédit"
        else:
            title = (
                f"Prix horaire le {date_ts.strftime('%A %d %B %Y')} — "
                "prédit uniquement (réel non publié)"
            )

        # Axe horizontal : on n'affiche que les heures (HH:MM) et pas la date,
        # puisque tout le graphique correspond à la même journée.
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_xlim(
            date_ts,
            date_ts + pd.Timedelta(hours=23, minutes=59),
        )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Heure de la journée", fontsize=11)
        ax.set_ylabel("Prix (EUR/MWh)", fontsize=11)
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.12)
        ax.legend(loc="best", frameon=True, framealpha=0.9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        fig.autofmt_xdate(rotation=0, ha="center")
        fig.tight_layout()
        plt.show()

    def _print_comparison(self, date_ts, real, predicted):
        n_pred = len(predicted)
        if real is None or real.empty:
            print(
                f"Le {date_ts.date()} : prédit sur {n_pred}h "
                f"(moyenne prédite = {predicted.mean():.2f} EUR/MWh). "
                "Aucun prix réel ENTSO-E disponible sur cette journée."
            )
            return

        # Jointure heure-par-heure : on ne compare que les heures où le réel existe.
        df = pd.concat(
            [real.rename("real"), predicted.rename("pred")], axis=1, join="inner"
        )
        if df.empty:
            print(
                f"Le {date_ts.date()} : prédit sur {n_pred}h mais aucune heure réelle "
                "ne coïncide avec une heure prédite."
            )
            return

        err = df["pred"] - df["real"]
        mae = err.abs().mean()
        rmse = float(np.sqrt((err ** 2).mean()))
        n_real = len(df)
        print(
            f"Le {date_ts.date()} : {n_real}/{n_pred}h avec prix réel — "
            f"moy réel={df['real'].mean():.2f}, moy prédit={df['pred'].mean():.2f}, "
            f"MAE={mae:.2f} EUR/MWh, RMSE={rmse:.2f} EUR/MWh."
        )
        if n_real < n_pred:
            print(
                f"  -> {n_pred - n_real}h restantes prédites sans contrepartie réelle "
                "(non agrégées dans les métriques ci-dessus)."
            )
