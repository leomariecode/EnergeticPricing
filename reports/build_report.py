"""
Génère reports/EnergeticPricing_report.pdf à partir des sorties de
experiments/run_ablations.py (dernier ablation_*.csv / folds_*.csv).

    python reports/build_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "EnergeticPricing_report.pdf"
REPO_URL = "https://github.com/leomariecode/EnergeticPricing"

DATA_START = pd.Timestamp("2023-01-01")  # START de experiments/run_ablations.py
MATURE_MONTHS = 18  # plis dont la fenêtre d'entraînement couvre >= 18 mois

# Palette catégorielle validée (mode clair) + encres neutres.
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

LABELS = {
    ("naive_D-1", "benchmark"): "Naive: same hour D-1",
    ("naive_D-7", "benchmark"): "Naive: same hour D-7",
    ("Simple", "forecast_base"): "Ridge · weather + ENTSO-E forecasts",
    ("Simple", "forecast_nocap"): "Ridge · + gas",
    ("Simple", "forecast_full"): "Ridge · + gas + capacity",
    ("Simple", "forecast_full_national"): "Ridge · full, national weather",
    ("Simple", "reconstruction"): "Ridge · + neighbour prices*",
    ("LightGBM", "forecast_base"): "LightGBM · weather + ENTSO-E forecasts",
    ("LightGBM", "forecast_nocap"): "LightGBM · + gas",
    ("LightGBM", "forecast_full"): "LightGBM · + gas + capacity",
    ("LightGBM", "forecast_full_national"): "LightGBM · full, national weather",
    ("LightGBM", "forecast_full_l1"): "LightGBM · full, L1 loss",
    ("LightGBM", "reconstruction"): "LightGBM · + neighbour prices*",
    ("LightGBM", "forecast_full_nuke_gen"): "LightGBM · full + nuclear output D-2",
    ("LightGBM", "forecast_full_nuke_delta"): "LightGBM · full + all nuclear features",
}
# Lignes de la figure 1 (les Ridge divergents restent dans le tableau).
FIG1_ROWS = [
    ("naive_D-7", "benchmark"), ("naive_D-1", "benchmark"),
    ("Simple", "forecast_base"), ("Simple", "forecast_nocap"),
    ("LightGBM", "forecast_base"), ("LightGBM", "forecast_nocap"),
    ("LightGBM", "forecast_full_national"), ("LightGBM", "forecast_full_l1"),
    ("LightGBM", "forecast_full_nuke_delta"),
    ("LightGBM", "forecast_full"), ("LightGBM", "reconstruction"),
]


def _latest(pattern: str) -> Path:
    files = sorted(RES.glob(pattern))
    if not files:
        sys.exit(f"Aucun fichier {pattern} dans {RES} : lancer experiments/run_ablations.py")
    return files[-1]


def _style_axes(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def fig_mae(summary: pd.DataFrame, path: Path):
    # Deux barres par modèle (même unité, même axe) : MAE sur les 5 plis et sur
    # les plis "matures" (>= 18 mois d'historique d'entraînement).
    d = summary.set_index(["model", "config"]).loc[FIG1_ROWS].reset_index()
    y = np.arange(len(d))
    h = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.barh(y + h / 2, d["mae_all"], height=h - 0.04, color="#a9c8ef", label="All 5 folds")
    ax.barh(y - h / 2, d["mae_mature"], height=h - 0.04, color=BLUE,
            label=f"Folds with ≥ {MATURE_MONTHS} months of training data")
    for yi, (a, m) in enumerate(zip(d["mae_all"], d["mae_mature"])):
        ax.text(a + 0.3, yi + h / 2, f"{a:.1f}", va="center", fontsize=7, color=INK2)
        ax.text(m + 0.3, yi - h / 2, f"{m:.1f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[(m, c)] for m, c in zip(d["model"], d["config"])])
    for lbl, c in zip(ax.get_yticklabels(), d["config"]):
        if c == "benchmark":
            lbl.set_color(INK2)
    ax.axhline(1.5, color=GRID, linewidth=0.8)
    ax.set_xlabel("Out-of-sample MAE (EUR/MWh)", fontsize=8, color=INK2)
    _style_axes(ax, grid_axis="x")
    ax.tick_params(axis="y", labelsize=7.5, colors=INK)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right", bbox_to_anchor=(1, 1), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fig_week(tag: str, path: Path, days: int = 14):
    lf = pd.read_parquet(RES / f"last_fold_{tag}.parquet")
    lf["time"] = pd.to_datetime(lf["time"])
    lf = lf.set_index("time").sort_index()
    # Fenêtre de 2 semaines la plus volatile du dernier pli.
    end = lf["y_true"].rolling(f"{days}D").std().idxmax()
    w = lf.loc[end - pd.Timedelta(days=days):end]
    fig, ax = plt.subplots(figsize=(7.2, 1.95))
    ax.plot(w.index, w["y_true"], color=INK, linewidth=1.2, label="Actual day-ahead price")
    ax.plot(w.index, w["y_pred"], color=BLUE, linewidth=1.2,
            label="LightGBM forecast (ex-ante information only)")
    ax.axhline(0, color=GRAY, linewidth=0.6)
    ax.set_ylabel("EUR/MWh", fontsize=8, color=INK2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    _style_axes(ax)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return w.index.min(), w.index.max()


def fig_importance(tag: str, path: Path, top: int = 10):
    imp = pd.read_csv(RES / f"importance_{tag}.csv").head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 1.95))
    ax.barh(imp["group"], 100 * imp["share"], color=BLUE, height=0.6)
    for yi, v in enumerate(100 * imp["share"]):
        ax.text(v + 0.5, yi, f"{v:.1f}%", va="center", fontsize=7.5, color=INK)
    ax.set_xlabel("Share of total split gain (%)", fontsize=8, color=INK2)
    _style_axes(ax, grid_axis="x")
    ax.tick_params(axis="y", labelsize=7.5, colors=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fig_folds(folds: pd.DataFrame, path: Path):
    series = [
        (("naive_D-1", "benchmark"), GRAY, "Naive D-1"),
        (("Simple", "forecast_nocap"), ORANGE, "Ridge (best linear)"),
        (("LightGBM", "forecast_full"), BLUE, "LightGBM (main model)"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 2.05))
    width = 0.26
    labels = None
    for k, ((m, c), col, lab) in enumerate(series):
        f = folds[(folds["model"] == m) & (folds["config"] == c)].sort_values("fold")
        x = np.arange(len(f)) + (k - 1) * width
        ax.bar(x, f["mae"], width=width - 0.03, color=col, label=lab)
        if labels is None:
            labels = [
                f"{pd.Timestamp(s):%b %y}–{pd.Timestamp(e):%b %y}\n"
                f"train ≈ {tm:.0f} mo"
                for s, e, tm in zip(f["test_start"], f["test_end"], f["train_months"])
            ]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("MAE (EUR/MWh)", fontsize=8, color=INK2)
    _style_axes(ax)
    ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


# ----------------------------------------------------------------------
# Document
# ----------------------------------------------------------------------
def build():
    FIG.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(_latest("ablation_*.csv"))
    folds = pd.read_csv(_latest("folds_*.csv"))
    folds["train_months"] = (
        (pd.to_datetime(folds["test_start"]) - DATA_START).dt.days / 30.44
    )
    folds["mature"] = folds["train_months"] >= MATURE_MONTHS

    summary = (
        folds.groupby(["model", "config"])
        .apply(lambda g: pd.Series({
            "mae_all": g["mae"].mean(),
            "mae_mature": g.loc[g["mature"], "mae"].mean(),
            "r2_mature": g.loc[g["mature"], "r2"].mean(),
        }), include_groups=False)
        .reset_index()
        .merge(res[["model", "config", "test_rmse", "test_r2", "n_features",
                    "spike_recall", "spike_precision", "ampl_ratio"]],
               on=["model", "config"], how="left")
    )

    def row(m, c):
        r = summary[(summary["model"] == m) & (summary["config"] == c)]
        return r.iloc[0]

    naive1 = row("naive_D-1", "benchmark")
    best = row("LightGBM", "forecast_full")
    recon = row("LightGBM", "reconstruction")
    nocap = row("LightGBM", "forecast_nocap")
    base = row("LightGBM", "forecast_base")
    nat = row("LightGBM", "forecast_full_national")
    l1 = row("LightGBM", "forecast_full_l1")
    ridge = row("Simple", "forecast_nocap")
    nuke_gen = row("LightGBM", "forecast_full_nuke_gen")
    nuke = row("LightGBM", "forecast_full_nuke_delta")
    imp_nuke = pd.read_csv(RES / "importance_LightGBM_forecast_full_nuke_delta.csv").set_index("group")["share"]
    skill_mature = 1 - best["mae_mature"] / naive1["mae_mature"]
    skill_all = 1 - best["mae_all"] / naive1["mae_all"]
    n_mature = int(folds.loc[folds["model"] == "naive_D-1", "mature"].sum())
    mature_start = pd.Timestamp(folds.loc[folds["mature"], "test_start"].min())
    test_end = pd.Timestamp(folds["test_end"].max())
    test_start = pd.Timestamp(folds["test_start"].min())

    fig_mae(summary, FIG / "mae.png")
    w0, w1 = fig_week("LightGBM_forecast_full", FIG / "week.png")
    fig_importance("LightGBM_forecast_full", FIG / "importance.png")
    fig_folds(folds, FIG / "folds.png")
    imp = pd.read_csv(RES / "importance_LightGBM_forecast_full.csv").set_index("group")["share"]

    # --- Styles -------------------------------------------------------
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=9.2,
                          leading=12.6, textColor=colors.HexColor(INK), spaceAfter=5)
    small = ParagraphStyle("small", parent=body, fontSize=7.8, leading=10,
                           textColor=colors.HexColor(INK2))
    h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=18, leading=22,
                        spaceAfter=2)
    sub = ParagraphStyle("sub", parent=body, fontSize=9.5, textColor=colors.HexColor(INK2),
                         spaceAfter=9)
    h2 = ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=11.5,
                        leading=15, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor(BLUE))
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=11, bulletIndent=2, spaceAfter=2.5)
    cell = ParagraphStyle("cell", parent=body, fontSize=8, leading=10, spaceAfter=0)
    cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")

    def P(t, s=body):
        return Paragraph(t, s)

    def B(items):
        return [Paragraph(t, bullet, bulletText="•") for t in items]

    def table(data, widths, zebra=True, bold_rows=()):
        rows = [[P(str(c), cellb if i == 0 or i in bold_rows else cell) for c in r]
                for i, r in enumerate(data)]
        t = Table(rows, colWidths=widths, repeatRows=1)
        st = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3fb")),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor(BLUE)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]
        if zebra:
            for i in range(2, len(rows), 2):
                st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f7f7f5")))
        t.setStyle(TableStyle(st))
        return t

    def img(p, w=17.0):
        iw, ih = ImageReader(str(p)).getSize()
        return Image(str(p), width=w * cm, height=w * cm * ih / iw)

    def link(url):
        return f"<link href='{url}' color='{BLUE}'>{url.replace('https://', '')}</link>"

    S = []
    # --- En-tête ------------------------------------------------------
    S += [
        P("Forecasting French Day-Ahead Power Prices", h1),
        P(f"Weather- and fundamentals-driven ML model for the EPEX Spot FR auction · Léo Marie · "
          f"{link(REPO_URL)}", sub),
    ]
    S.append(table([
        ["Target", "Information set", "Backtest", "Main model", "vs. naive D-1"],
        ["FR day-ahead price, 24 hourly values (EUR/MWh)",
         "Only data known before gate closure (D-1, 12:00 CET)",
         f"{test_start:%b %Y} – {test_end:%b %Y}, 5 expanding folds",
         f"LightGBM, {int(best['n_features'])} features",
         f"<b>MAE −{100 * skill_mature:.0f}%</b> ({naive1['mae_mature']:.1f} → "
         f"{best['mae_mature']:.1f}) once ≥ {MATURE_MONTHS} months of history"],
    ], [3.5 * cm, 3.6 * cm, 3.3 * cm, 3.0 * cm, 3.8 * cm], zebra=False))
    S.append(Spacer(1, 4))

    # --- 1. Problème ---------------------------------------------------
    S.append(P("1 · Problem", h2))
    S.append(P(
        "The French day-ahead price is set every day at 12:00 CET by the coupled European auction "
        "(EPEX Spot / EUPHEMIA) for each hour of the next day (15-minute products since Oct. 2025). It is the "
        "reference for spot hedging, battery and flexibility scheduling, and the anchor of intraday trading. "
        "It is driven by <b>residual load</b> (demand minus wind and solar), by the <b>marginal technology</b> "
        "(gas plants set the price when nuclear, hydro and renewables do not cover demand) and by "
        "<b>cross-border coupling</b>. The project forecasts the 24 hourly prices of D+1 using <b>only "
        "information available before gate closure</b>, and measures what each block of information is worth "
        "against naive persistence."
    ))

    # --- 2. Données ---------------------------------------------------
    S.append(P("2 · Data", h2))
    S.append(table([
        ["Source", "Variables", "Granularity", "Use in the model"],
        ["ENTSO-E Transparency Platform",
         "FR day-ahead price (target); day-ahead forecasts of load, wind and solar; DA prices of "
         "DE, BE, ES, IT-North, CH", "Hourly (15 min since Oct. 2025)",
         "Target; residual load and its ramps. Neighbour prices only in the <i>reconstruction</i> "
         "upper bound"],
        ["Open-Meteo (ERA5 archive + forecast API)",
         "Temperature, 10 m wind speed, shortwave radiation, cloud cover, precipitation",
         "Hourly, 96 départements (prefecture coordinates)",
         "480 spatial weather features (5 variables × 96 départements)"],
        ["ODRE national installation registry",
         "Solar and wind capacity per département with commissioning date", "Daily steps",
         "Point-in-time installed capacity; <i>renewable potential</i> = Σ<sub>dept</sub> capacity × "
         "radiation (solar) or × wind speed<super>3</super> (wind)"],
        ["ICE Endex TTF front-month (via Yahoo Finance)", "Natural-gas price (EUR/MWh)", "Daily close",
         "Marginal-cost proxy, lagged 2 days (last settlement known at gate closure)"],
        ["ENTSO-E · nuclear", "Actual nuclear output; planned unavailability messages (REMIT)",
         "Hourly · per event", "Output of D-2, planned outage MW at D, their change vs D-1, "
         "residual load net of available nuclear (tested, not retained)"],
        ["INSEE · French calendar", "Population by département · public holidays",
         "Static · daily", "Population-weighted national weather; holidays treated as weekends"],
    ], [3.4 * cm, 5.0 * cm, 2.9 * cm, 5.9 * cm]))
    S.append(P(
        f"Sample: Jan 2023 – Apr 2026, about 29,200 hourly observations. All API calls are cached as parquet; "
        "the dataset and every result in this note rebuild with one command.", small))

    # --- 3. Méthode ----------------------------------------------------
    S.append(P("3 · Methodology", h2))
    S += B([
        "<b>No look-ahead.</b> Each feature is checked against what is known at D-1 noon: price lags at "
        "D-1 and D-7, gas settlement of D-2, nuclear output of D-2, ENTSO-E day-ahead forecasts, calendar. Neighbour DA prices clear "
        "<i>simultaneously</i> with France, so they are excluded from all forecasting models and kept only "
        "in a <i>reconstruction</i> run that shows how large this leak is.",
        "<b>Features from market structure:</b> residual-load forecast and hour-to-hour ramps (morning and "
        "evening solar ramps), capacity-weighted renewable potential, cyclical hour and month encodings, "
        "French public holidays.",
        "<b>Validation:</b> rolling-origin cross-validation with 5 expanding windows of about 6.5 months of "
        "test each. LightGBM early-stops on the tail of the <i>training</i> window, never on the test fold.",
        "<b>Benchmarks:</b> naive persistence (same hour D-1 and D-7) on the same folds. Ridge regression is "
        "the linear baseline, LightGBM the main model, and an auxiliary LightGBM classifier flags price spikes "
        "(outside the 5–95% quantiles, including negative prices).",
    ])

    # --- 4. Résultats -------------------------------------------------
    S.append(PageBreak())
    S.append(P("4 · Results", h2))
    S.append(KeepTogether([
        img(FIG / "mae.png"),
        P("Figure 1 – Out-of-sample MAE by model and feature set (lower is better). Dark bars: the "
          f"{n_mature} folds tested from {mature_start:%b %Y} onwards, trained on ≥ {MATURE_MONTHS} months. "
          "*Uses neighbour prices that are unknown before the auction: an upper bound, not a forecast.",
          small),
    ]))

    order = [("naive_D-1", "benchmark"), ("naive_D-7", "benchmark"),
             ("Simple", "forecast_nocap"), ("Simple", "forecast_full"),
             ("LightGBM", "forecast_base"), ("LightGBM", "forecast_nocap"),
             ("LightGBM", "forecast_full_national"), ("LightGBM", "forecast_full_l1"),
             ("LightGBM", "forecast_full_nuke_gen"), ("LightGBM", "forecast_full_nuke_delta"),
             ("LightGBM", "forecast_full"), ("LightGBM", "reconstruction")]
    tab = [["Model · features", "MAE, all folds", f"MAE, ≥ {MATURE_MONTHS} mo", "R², all folds",
            f"R², ≥ {MATURE_MONTHS} mo"]]
    for m, c in order:
        r = row(m, c)
        r2_all = f"{r['test_r2']:.2f}" if r["test_r2"] > -1 else "&lt; 0 (diverges)"
        tab.append([LABELS[(m, c)], f"{r['mae_all']:.1f}", f"{r['mae_mature']:.1f}", r2_all,
                    f"{r['r2_mature']:.2f}"])
    S.append(table(tab, [6.6 * cm, 2.4 * cm, 2.6 * cm, 2.8 * cm, 2.6 * cm],
                   bold_rows=[order.index(("LightGBM", "forecast_full")) + 1]))
    S.append(P("EUR/MWh. R² below zero means a few extreme errors dominate: the linear model extrapolates "
               "on the trending capacity features.", small))

    S.append(P("Key findings", h2))
    S += B([
        f"<b>The model beats persistence by {100 * skill_mature:.0f}% once it has seen a full seasonal "
        f"cycle</b>: MAE {best['mae_mature']:.1f} vs {naive1['mae_mature']:.1f} EUR/MWh, R² "
        f"{best['r2_mature']:.2f}, on every one of the last {n_mature} folds. With 6 to 13 months of "
        f"history it does <i>worse</i> than persistence (Figure 2), which is why the 5-fold average gain is "
        f"only {100 * skill_all:.0f}%. In practice it needs at least 18 months of training data.",
        f"<b>Installed capacity is the most valuable block.</b> Capacity-weighted renewable potential cuts "
        f"the MAE from {nocap['mae_mature']:.1f} to {best['mae_mature']:.1f} on mature folds and stabilises "
        f"the short-history folds (from {nocap['mae_all']:.1f} to {best['mae_all']:.1f} over all folds). "
        f"Département-level weather beats national averages ({nat['mae_mature']:.1f} → "
        f"{best['mae_mature']:.1f}).",
        f"<b>What did not help:</b> the 2-day-lagged gas price ({base['mae_mature']:.1f} without vs "
        f"{nocap['mae_mature']:.1f} with) and an L1 loss ({l1['mae_mature']:.1f}). A Ridge on weather "
        f"and ENTSO-E forecasts is a strong linear baseline ({ridge['mae_mature']:.1f}); LightGBM's edge "
        "comes from using the capacity features, which make the linear model unstable.",
        f"<b>Nuclear availability adds nothing over yesterday's price (2024-2026).</b> Nuclear output of "
        f"D-2, planned outages, their day-on-day change and the residual load left to thermal plants "
        f"({nuke_gen['mae_mature']:.2f} / {nuke['mae_mature']:.2f} vs {best['mae_mature']:.2f}). The model "
        f"uses the thermal residual load ({100 * imp_nuke.get('thermal_residual', 0):.0f}% of gain) but as a "
        "substitute for residual load: the fleet moves over weeks, so its state is already priced in D-1. "
        "Caveat: ENTSO-E only serves the <i>last</i> revision of each REMIT message, republished in Oct. 2025, "
        "so forced outages were excluded to avoid look-ahead.",
        f"<b>Neighbour prices are a leakage trap.</b> Adding them drops the MAE to {recon['mae_all']:.1f} "
        "and a single coupled neighbour (Belgium) takes most of the model's importance. Results that include "
        "such features overstate what can be achieved before the auction. This project's own first version "
        "made that mistake.",
        f"<b>Extremes remain hard.</b> Forecasts are smoothed (prediction std / actual std = "
        f"{best['ampl_ratio']:.2f}), and the spike classifier catches only {100 * best['spike_recall']:.0f}% "
        f"of hours outside the 5–95% band, with {100 * best['spike_precision']:.0f}% precision.",
    ])

    S.append(Spacer(1, 4))
    S.append(KeepTogether([
        img(FIG / "folds.png"),
        P("Figure 2 – MAE per test fold, with the length of the training window. The model needs a full "
          "seasonal cycle; once it has one, it beats persistence on every fold.", small),
    ]))
    S.append(Spacer(1, 1))
    S.append(KeepTogether([
        img(FIG / "week.png"),
        P(f"Figure 3 – Hourly forecast vs actual over the most volatile two weeks of the last test fold "
          f"({w0:%d %b %Y} – {w1:%d %b %Y}), out-of-sample. The level and daily shape are tracked; the "
          "negative-price troughs on sunny, windy weekends are missed.", small),
    ]))
    S.append(Spacer(1, 1))
    S.append(KeepTogether([
        img(FIG / "importance.png"),
        P(f"Figure 4 – Feature importance of the main model (share of split gain; département weather grouped "
          f"by variable). Yesterday's price ({100 * imp.get('price_lag_1d', 0):.0f}%) and the residual-load "
          f"forecast ({100 * imp.get('net_load_forecast', 0):.0f}%) dominate, followed by département wind "
          "speed and installed capacity.", small),
    ]))

    # --- 5. Limites / prochaines étapes -------------------------------
    S.append(P("5 · Limitations and next steps", h2))
    S += B([
        "<b>Weather is reanalysis, not forecast.</b> Historical weather is ERA5 (what actually happened), "
        "whereas a live model sees D-1 weather forecasts, so backtest errors are slightly optimistic. Next "
        "step: train on archived forecast runs (Open-Meteo historical forecasts, ECMWF).",
        "<b>Point-in-time fundamentals:</b> nuclear features need a REMIT feed with the full revision "
        "history (as-of D-1 view, including forced outages); hydro reservoir levels, interconnection "
        "capacities and the EUA carbon price are still missing. They are the likeliest sources of the "
        "missed spikes.",
        "<b>Only a point forecast.</b> Trading and battery dispatch need distributions: quantile LightGBM "
        "or the Temporal Fusion Transformer already implemented in the code (quantile loss), scored with "
        "pinball loss.",
        "<b>From forecast error to P&amp;L:</b> backtest a simple strategy, such as battery arbitrage or a "
        "day-ahead vs intraday position, to express the MAE gain in EUR, and move to the native 15-minute "
        "resolution.",
    ])
    S.append(Spacer(1, 5))
    S.append(P(
        "<b>Stack:</b> Python, pandas, LightGBM, scikit-learn, PyTorch Forecasting (TFT), entsoe-py, "
        "Open-Meteo API. <b>Reproduce:</b> <font face='Courier'>python experiments/run_ablations.py</font> "
        f"then <font face='Courier'>python reports/build_report.py</font>. <b>Code:</b> {link(REPO_URL)}",
        small))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor(INK2))
        canvas.drawString(2 * cm, 1.1 * cm, "Léo Marie · French day-ahead price forecasting")
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"{doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.6 * cm, bottomMargin=1.7 * cm,
        title="Forecasting French Day-Ahead Power Prices", author="Léo Marie",
    )
    doc.build(S, onFirstPage=footer, onLaterPages=footer)
    print(f"PDF écrit : {OUT}")


if __name__ == "__main__":
    build()
