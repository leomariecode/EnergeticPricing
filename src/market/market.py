import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.loader import load_data
from src.models.pricing_from_meteo import (
    model_learn,
    model_score,
    predict_price_from_data,
    predict_spike_proba,
)


_SUPPORTED_FREQS = ("1h", "15min")
_SUPPORTED_METEO_MODES = ("department", "national")

# Horizon maximal de prédiction au-delà d'aujourd'hui : Open-Meteo donne plusieurs jours
# de forecast, ENTSO-E ne publie le day-ahead que J+1, et la qualité de la prévision
# météo se dégrade vite -> 3 jours est un compromis raisonnable.
MAX_FORWARD_DAYS = 3


class EnergyMarket:

    def __init__(self, start, end, freq: str = "1h", meteo_mode: str = "department"):
        # freq : "1h" (par défaut) ou "15min". Détermine le pas de temps cible utilisé
        # pour tout le pipeline (prix, prévisions ENTSO-E, météo, features, modèle).
        # En "15min", les prix DA récents (post-oct 2025) restent au pas natif 15 min ;
        # les heures plus anciennes sont upsamplées via ffill (le prix horaire s'applique
        # à chaque slot 15 min). La météo est fetchée en horaire et interpolée à 15 min.
        #
        # meteo_mode : "department" (défaut) garde la météo par département (95x5 colonnes,
        # le modèle voit la structure spatiale) ; "national" l'agrège en une moyenne
        # pondérée population (baseline plus compacte).
        if freq not in _SUPPORTED_FREQS:
            raise ValueError(f"freq doit être dans {_SUPPORTED_FREQS}, reçu '{freq}'")
        if meteo_mode not in _SUPPORTED_METEO_MODES:
            raise ValueError(
                f"meteo_mode doit être dans {_SUPPORTED_METEO_MODES}, reçu '{meteo_mode}'"
            )
        self.data = None
        self.model = None
        self.model_type = None
        self.start = start
        self.end = end
        self.freq = freq
        self.meteo_mode = meteo_mode

    def initialize(self):
        print(
            f"[EnergyMarket.initialize] start={self.start}, end={self.end}, "
            f"freq={self.freq}, meteo_mode={self.meteo_mode}"
        )
        self.data = load_data(self.start, self.end, freq=self.freq)
        print(f"[EnergyMarket.initialize] Données chargées : {len(self.data)} lignes")

    def learn(self, model_type):
        print(f"[EnergyMarket.learn] Apprentissage modèle {model_type} (météo {self.meteo_mode})")
        self.model_type = model_type
        self.model = model_learn(self.data, model_type, meteo_mode=self.meteo_mode)
        print("[EnergyMarket.learn] Modèle prêt")
        # Récap visuel : dernier jour (pas natif), dernier mois (moy. journalière),
        # dernière année (moy. journalière). Chaque titre porte le nom du modèle.
        self._plot_training_summary()

    def _ensure_data_covers(self, date_ts: pd.Timestamp) -> None:
        # Étend self.data jusqu'à date_ts inclus si nécessaire (en re-chargeant la
        # tranche manquante via load_data, qui sait gérer la météo forecast + ENTSO-E).
        data_max = pd.to_datetime(self.data["time"]).max().normalize()
        if date_ts <= data_max:
            return
        extra_start = (data_max + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        extra_end = date_ts.strftime("%Y-%m-%d")
        print(
            f"[EnergyMarket] Extension du dataset : {extra_start} -> {extra_end} "
            f"(au-delà du END configuré, fetch Open-Meteo + ENTSO-E)"
        )
        extra = load_data(extra_start, extra_end, freq=self.freq)
        if extra.empty:
            print("[EnergyMarket] Aucune nouvelle donnée récupérée pour l'extension.")
            return
        # Dédup sur (time, code) pour ne pas dupliquer si load_data renvoie un chevauchement.
        subset = ["time", "code"] if "code" in self.data.columns else ["time"]
        self.data = (
            pd.concat([self.data, extra], ignore_index=True)
            .drop_duplicates(subset=subset, keep="last")
            .sort_values(subset)
            .reset_index(drop=True)
        )
        print(
            f"[EnergyMarket] Dataset étendu : {len(self.data)} lignes "
            f"(max time = {pd.to_datetime(self.data['time']).max()})"
        )

    def _build_eval_df(self) -> pd.DataFrame:
        # Construit une série temporelle indexée par time avec colonnes 'predicted' et 'real'.
        pred = predict_price_from_data(self.model, self.data)
        real = (
            self.data.drop_duplicates(subset="time")
            .loc[:, ["time", "price"]]
            .set_index("time")["price"]
        )
        df = pd.concat([pred.rename("predicted"), real.rename("real")], axis=1)
        df.index = pd.to_datetime(df.index)
        df.index.name = "time"
        return df.sort_index()

    def _plot_training_summary(self):
        if self.model is None:
            return
        df = self._build_eval_df()
        if df.empty:
            print("[EnergyMarket.learn] Pas de données pour le récap visuel.")
            return

        end_ts = pd.to_datetime(self.end).normalize()
        model_name = self.model_type or "?"

        # 1) Dernier jour au pas natif (1h ou 15min selon self.freq).
        day_mask = df.index.normalize() == end_ts
        day_slice = df.loc[day_mask]
        if day_slice.empty:
            # Fallback : le dernier jour effectivement présent dans les données.
            last_present = df.index.normalize().max()
            day_slice = df.loc[df.index.normalize() == last_present]
            end_ts = last_present
        self._plot_eval_window(
            day_slice,
            title=f"[{model_name}] Dernier jour — {end_ts.strftime('%A %d %B %Y')} ({self.freq})",
            xfmt="%H:%M",
            xlocator=mdates.HourLocator(interval=2),
            xlim=(end_ts, end_ts + pd.Timedelta(hours=23, minutes=59)),
            xlabel="Heure de la journée",
        )

        # 2) Dernier mois (30 jours glissants), moyenne par jour.
        month_start = end_ts - pd.Timedelta(days=30)
        month_slice = df.loc[df.index >= month_start]
        month_daily = month_slice.resample("D").mean()
        self._plot_eval_window(
            month_daily,
            title=f"[{model_name}] Dernier mois (moyenne journalière) — "
                  f"{month_start.strftime('%d %b %Y')} → {end_ts.strftime('%d %b %Y')}",
            xfmt="%d %b",
            xlocator=mdates.DayLocator(interval=3),
            xlim=(month_start, end_ts),
            xlabel="Jour",
        )

        # 3) Dernière année (365 jours glissants), moyenne par jour.
        year_start = end_ts - pd.Timedelta(days=365)
        year_slice = df.loc[df.index >= year_start]
        year_daily = year_slice.resample("D").mean()
        self._plot_eval_window(
            year_daily,
            title=f"[{model_name}] Dernière année (moyenne journalière) — "
                  f"{year_start.strftime('%b %Y')} → {end_ts.strftime('%b %Y')}",
            xfmt="%b %Y",
            xlocator=mdates.MonthLocator(interval=1),
            xlim=(year_start, end_ts),
            xlabel="Mois",
        )

    def _plot_eval_window(self, df, title, xfmt, xlocator, xlim, xlabel):
        # df : DataFrame indexé par datetime, colonnes 'predicted' et 'real' (NaN possible).
        fig, ax = plt.subplots(figsize=(12, 4.5))

        ax.plot(
            df.index, df["predicted"].values,
            marker="x", markersize=5, linestyle="--", linewidth=1.5,
            color="#E67E22", label="Prix prédit (modèle)",
        )
        # Pour le réel : on ne trace que les points effectivement disponibles, sans interpoler.
        real = df["real"].dropna()
        if not real.empty:
            ax.plot(
                real.index, real.values,
                marker="o", markersize=4, linestyle="-", linewidth=1.8,
                color="#2E86AB", label="Prix réel (ENTSO-E)",
            )

        ax.xaxis.set_major_locator(xlocator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(xfmt))
        if xlim is not None:
            ax.set_xlim(*xlim)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Prix (EUR/MWh)", fontsize=10)
        ax.grid(True, which="major", alpha=0.3)
        ax.legend(loc="best", frameon=True, framealpha=0.9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        fig.autofmt_xdate(rotation=30, ha="right")
        fig.tight_layout()
        plt.show()

    def score(self, recompute_on_data: bool = False):
        # Affiche les scores du modèle. Par défaut, ceux stockés au moment du fit.
        if self.model is None:
            raise RuntimeError("Aucun modèle entraîné : appeler learn() d'abord.")
        data = self.data if recompute_on_data else None
        return model_score(self.model, data=data)

    def predict_from_meteo(self, date):
        # Prédit les 24h d'une journée. Compare au prix réel quand disponible,
        # heure par heure : aucune heure n'est moyennée ni interpolée.
        # On accepte une date jusqu'à MAX_FORWARD_DAYS après aujourd'hui : si
        # elle est au-delà du dataset déjà chargé, on étend self.data à la volée
        # (Open-Meteo forecast pour la météo, ENTSO-E publiera les forecasts
        # load/wind/solar pour les jours à venir ; le prix réel restera NaN
        # tant qu'il n'est pas publié, c'est attendu).
        print(f"[EnergyMarket.predict_from_meteo] date={date}")
        if self.data is None:
            raise RuntimeError("self.data est None : appeler initialize() d'abord.")
        if self.model is None:
            raise RuntimeError("self.model est None : appeler learn() d'abord.")

        date_ts = pd.to_datetime(date).normalize()
        start_ts = pd.to_datetime(self.start).normalize()
        today = pd.Timestamp.today().normalize()
        max_allowed = today + pd.Timedelta(days=MAX_FORWARD_DAYS)
        if date_ts < start_ts:
            raise ValueError(
                f"date {date_ts.date()} antérieure au start configuré ({start_ts.date()})"
            )
        if date_ts > max_allowed:
            raise ValueError(
                f"date {date_ts.date()} au-delà de la limite (aujourd'hui + "
                f"{MAX_FORWARD_DAYS} jours = {max_allowed.date()})"
            )

        # Si la date demandée dépasse le dataset, on charge le complément.
        self._ensure_data_covers(date_ts)

        times = pd.to_datetime(self.data["time"])
        if not (times.dt.normalize() == date_ts).any():
            raise ValueError(
                f"Aucune donnée météo pour {date_ts.date()} dans self.data "
                "après extension. Vérifie la disponibilité Open-Meteo / ENTSO-E."
            )

        # On prédit sur TOUT l'historique puis on slice la journée demandée : les
        # features de lag (prix J-1, J-7) et de gradient ont besoin du contexte
        # qui précède la journée — un slice préalable les casserait.
        predicted_full = predict_price_from_data(self.model, self.data)
        spike_full = predict_spike_proba(self.model, self.data)

        day_mask = predicted_full.index.normalize() == date_ts
        predicted = predicted_full.loc[day_mask]
        spike = spike_full.loc[spike_full.index.normalize() == date_ts] if not spike_full.empty else spike_full

        # Le prix réel ENTSO-E peut être manquant pour les heures les plus récentes :
        # on ne le construit que sur les heures où il existe vraiment (NaN exclus).
        real = None
        if "price" in self.data.columns:
            real_df = (
                self.data.loc[times.dt.normalize() == date_ts]
                .drop_duplicates(subset="time")
                .loc[:, ["time", "price"]]
                .dropna(subset=["price"])
                .sort_values("time")
            )
            if not real_df.empty:
                real = real_df.set_index("time")["price"]

        self._plot_prices(date_ts, real, predicted)
        self._print_comparison(date_ts, real, predicted)
        self._print_spike_report(date_ts, spike)
        return predicted, real

    def _print_spike_report(self, date_ts, spike):
        # Résume le risque de pic sur la journée à partir du classifieur dédié.
        if spike is None or spike.empty:
            return
        hi = spike[spike >= 0.5]
        if hi.empty:
            print(
                f"Risque de pic le {date_ts.date()} : faible "
                f"(proba max = {spike.max():.0%} à {spike.idxmax():%H:%M})."
            )
            return
        creneaux = ", ".join(f"{t:%H:%M} ({p:.0%})" for t, p in hi.items())
        print(
            f"Risque de pic le {date_ts.date()} : {len(hi)} créneau(x) flaggé(s) "
            f"(proba ≥ 50%) -> {creneaux}"
        )

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
