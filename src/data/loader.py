import os
from pathlib import Path
import time
import pandas as pd
import json
from shapely.geometry import shape
import requests
from entsoe import EntsoePandasClient

DATA_DIR = Path("data")
METEO_CACHE_DIR = DATA_DIR /"cache"/"meteo"
ENTSOE_CACHE_DIR = DATA_DIR / "cache" / "entsoe"
GAS_CACHE_DIR = DATA_DIR / "cache" / "gas"

# Registre national des installations de production (capacité solaire/éolien
# par commune). On l'agrège au département pour pondérer la météo par la
# puissance installée — bien plus prédictif qu'une météo brute.
REGISTRY_PATH = (
    DATA_DIR / "ernergy_production"
    / "registre-national-installation-production-stockage-electricite-agrege.csv"
)

SUPPORTED_FREQS = ("1h", "15min")


def _scan_cache_files(cache_dir: Path, prefix: str):
    # Liste tous les fichiers du dossier au format '{prefix}_YYYY-MM-DD_YYYY-MM-DD.parquet'
    # et renvoie [(start_ts, end_ts, path), ...]. Les fichiers dont le nom ne parse pas
    # comme deux dates ISO en fin sont ignorés silencieusement.
    if not cache_dir.exists():
        return []
    found = []
    for p in cache_dir.glob(f"{prefix}_*.parquet"):
        parts = p.stem.split("_")
        if len(parts) < 2:
            continue
        try:
            e_ts = pd.to_datetime(parts[-1])
            s_ts = pd.to_datetime(parts[-2])
        except (ValueError, TypeError):
            continue
        found.append((s_ts, e_ts, p))
    return found


def _find_covering_cache(cache_dir: Path, prefix: str, start: str, end: str) -> pd.DataFrame | None:
    # Cherche un cache dont la plage [s,e] couvre [start, end].
    # Si trouvé, renvoie un slice du DataFrame chargé ; sinon None.
    s_d = pd.to_datetime(start).date()
    e_d = pd.to_datetime(end).date()
    candidates = _scan_cache_files(cache_dir, prefix)
    # En cas de plusieurs caches couvrants, on prend le plus large (rare mais propre).
    candidates = sorted(
        [c for c in candidates if c[0].date() <= s_d and c[1].date() >= e_d],
        key=lambda c: (c[0], -c[1].toordinal()),
    )
    if not candidates:
        return None
    s, e, p = candidates[0]
    print(f"[cache] {prefix} : hit (couverture) -> {p.name} (slice {s_d}..{e_d})")
    df = pd.read_parquet(p)
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    s_ts = pd.Timestamp(s_d)
    e_ts = pd.Timestamp(e_d) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return df.loc[(df["time"] >= s_ts) & (df["time"] <= e_ts)].reset_index(drop=True)


def load_data(start: str, end: str, freq: str = "1h", nuclear: bool = False) -> pd.DataFrame:
    # start / end au format ISO "YYYY-MM-DD" ; freq = "1h" ou "15min".
    # Pour "15min" : prix et forecasts ENTSO-E gardés en 15-min natifs (post-oct 2025)
    # ou upsamplés par ffill (avant). La météo (toujours fetchée en horaire) est
    # interpolée linéairement à 15-min en sortie.
    if freq not in SUPPORTED_FREQS:
        raise ValueError(f"freq doit être dans {SUPPORTED_FREQS}, reçu '{freq}'")
    print(f"[load_data] Démarrage : start={start}, end={end}, freq={freq}")

    geo_path = DATA_DIR / "Geographie" / "departements.geojson"
    pop_path = DATA_DIR / "Geographie" / "donnees_departements.csv"
    pref_path = DATA_DIR / "Geographie" / "prefectures_france.csv"

    print(f"[load_data] Lecture du geojson : {geo_path}")
    DEP_CODE = [str(i).zfill(2) for i in range(1, 96) if i != 20] + ["2A", "2B"]
    with open(geo_path, encoding="utf-8") as f:
        features = json.load(f)["features"]
    departements = pd.DataFrame([
        {"code": ft["properties"]["code"], "geometry": shape(ft["geometry"])}
        for ft in features if ft["properties"]["code"] in DEP_CODE
    ])
    # Fichier INSEE : DEP = code département ("01".."95", "2A"/"2B"),
    # PMUN = population municipale (référence légale), PTOT = population totale.
    pop_dep = pd.read_csv(pop_path, sep=";", dtype={"DEP": str})
    pop_dep = pop_dep.rename(columns={"DEP": "code", "PMUN": "population"})[
        ["code", "population"]
    ]
    departements = departements.merge(pop_dep, on="code")
    print(f"[load_data] {len(departements)} départements à traiter")

    # Coordonnées de la préfecture par département : on interroge Open-Meteo sur le
    # chef-lieu (là où se concentre la consommation) plutôt que sur le point
    # représentatif du polygone. Fallback sur representative_point si absent.
    pref_coords = {}
    if pref_path.exists():
        prefectures = pd.read_csv(pref_path, dtype={"code_dept": str})
        pref_coords = (
            prefectures.set_index("code_dept")[["latitude", "longitude"]]
            .to_dict("index")
        )
        print(f"[load_data] Préfectures chargées : {len(pref_coords)} coordonnées")
    else:
        print(f"[load_data] {pref_path} absent -> point représentatif du polygone")

    df_meteo_list = []
    for i, code in enumerate(DEP_CODE, start=1):
        departement = departements[departements["code"] == code]
        if departement.empty:
            print(f"[load_data] ({i}/{len(DEP_CODE)}) dep {code} introuvable dans le geojson, skip")
            continue
        print(f"[load_data] ({i}/{len(DEP_CODE)}) météo dep {code}...")
        # Open-Meteo ne sert que de l'horaire (archive ERA5 + forecast). L'upsample
        # éventuel à 15-min se fait plus bas en une passe après le merge.
        coords = pref_coords.get(code)
        lat = coords["latitude"] if coords else None
        lon = coords["longitude"] if coords else None
        geometry = departement["geometry"].iloc[0]
        df_dep = load_meteo_france(start, end, geometry, code=code, lat=lat, lon=lon)
        df_dep["code"] = code
        # Population du département : sert de poids pour l'agrégation nationale de la météo
        # (un département à 2M habitants pèse beaucoup plus dans la conso nationale qu'un à 80k).
        df_dep["population"] = float(departement["population"].iloc[0])
        df_meteo_list.append(df_dep)

    print(f"[load_data] Concat de {len(df_meteo_list)} dataframes météo")
    df_meteo = pd.concat(df_meteo_list, ignore_index=True)

    print("[load_data] Attribution de la capacité installée renouvelable par département")
    df_meteo = _attach_installed_capacity(df_meteo)

    if freq == "15min":
        # Open-Meteo ne sert que de l'horaire. On upsample chaque dep à 15-min
        # via interpolation linéaire pour pouvoir joindre proprement aux prix/forecasts 15-min.
        # La météo varie lentement (temp/vent/rayonnement) donc l'approximation est saine ;
        # la précipitation est interpolée linéairement aussi (approximation acceptable).
        print("[load_data] Upsampling météo horaire -> 15 min par département")
        df_meteo = _upsample_meteo_to_15min(df_meteo)

    print("[load_data] Chargement des prix ENTSO-E")
    df_price = load_entsoe_data(start, end, freq=freq)

    print("[load_data] Chargement des prévisions ENTSO-E (load + wind/solar)")
    df_fc = load_entsoe_forecasts(start, end, freq=freq)

    print("[load_data] Chargement des prix des pays voisins")
    df_nb = load_entsoe_neighbour_prices(start, end, freq=freq)

    print("[load_data] Chargement du prix du gaz (TTF)")
    df_gas = load_gas_price(start, end, freq=freq)

    # Nucléaire : optionnel (téléchargement REMIT lent, et sans gain mesuré sur
    # 2024-2026, cf. experiments/run_ablations.py).
    if nuclear:
        print("[load_data] Chargement des features nucléaires (production J-2, arrêts planifiés)")
        df_nuc = load_nuclear_features(start, end, freq=freq)
    else:
        df_nuc = pd.DataFrame({"time": pd.Series(dtype="datetime64[ns]")})

    # Left-merge depuis la météo : on garde toutes les heures pour lesquelles on a la météo,
    # le prix peut être NaN (typiquement les heures les plus récentes que l'ENTSO-E
    # n'a pas encore publiées). On veut pouvoir prédire ces heures-là.
    print(
        f"[load_data] Merge météo ({len(df_meteo)}) <-> prix ({len(df_price)}) "
        f"<-> forecasts ({len(df_fc)}) <-> voisins ({len(df_nb)}) <-> gaz ({len(df_gas)})"
    )
    df = (
        df_meteo
        .merge(df_price, on="time", how="left")
        .merge(df_fc, on="time", how="left")
        .merge(df_nb, on="time", how="left")
        .merge(df_gas, on="time", how="left")
        .merge(df_nuc, on="time", how="left")
    )

    n_price = df["price"].notna().sum()
    n_load = df["load_forecast"].notna().sum() if "load_forecast" in df.columns else 0
    print(
        f"[load_data] Terminé : {len(df)} lignes (freq={freq}, "
        f"prix réel : {n_price} ; load_forecast : {n_load})"
    )
    return df


def _upsample_meteo_to_15min(df: pd.DataFrame) -> pd.DataFrame:
    # Le merge produit du horaire (météo hourly + ENTSO-E déjà au pas voulu).
    # Pour le pas 15-min on densifie par groupby(code) en interpolant linéairement
    # les colonnes météo (continues) et en ffill les colonnes nationales (prix +
    # forecasts ENTSO-E, qui sont des step-fonctions horaires côté source).
    print(f"[load_data] Upsample météo → 15-min (interpolation linéaire)")
    meteo_cols = ["temperature_2m", "wind_speed_10m", "shortwave_radiation",
                  "cloud_cover", "precipitation"]
    # Colonnes nationales (step-fonctions horaires) : prix, forecasts, prix voisins.
    national_base = ["price", "load_forecast", "wind_forecast", "solar_forecast"]
    national_cols = [
        c for c in df.columns
        if c in national_base or c.startswith("price_")
    ]
    static_cols = [c for c in ("population",) if c in df.columns]
    # Capacité installée : palier mensuel par département -> ffill dans le groupe.
    capacity_cols = [c for c in ("solar_capacity", "wind_capacity") if c in df.columns]

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])

    parts = []
    for code, g in df.groupby("code", sort=False):
        g = g.set_index("time").sort_index()
        new_idx = pd.date_range(g.index.min(), g.index.max(), freq="15min")
        g15 = g.reindex(new_idx)
        for c in meteo_cols:
            if c in g15.columns:
                g15[c] = g15[c].astype(float).interpolate("linear", limit_direction="both")
        for c in national_cols:
            if c in g15.columns:
                g15[c] = g15[c].ffill(limit=3)
        for c in static_cols:
            if c in g15.columns:
                g15[c] = g15[c].ffill().bfill()
        for c in capacity_cols:
            if c in g15.columns:
                g15[c] = g15[c].ffill().bfill()
        g15["code"] = code
        g15.index.name = "time"
        parts.append(g15.reset_index())
    return pd.concat(parts, ignore_index=True)




def load_entsoe_data(start: str, end: str, freq: str = "1h") -> pd.DataFrame:
    # ENTSO-E renvoie du 15-min en FR depuis oct. 2025 (et de l'horaire avant).
    # freq="1h"     -> on agrège à l'heure (mean) quelle que soit la source.
    # freq="15min"  -> on resample en 15-min puis ffill (le prix horaire d'avant
    #                  MTU s'applique aux 4 slots de l'heure).
    # Cache : si un fichier 'prices_{freq}_S_E.parquet' couvre déjà [start, end],
    # on en sert un slice (sans repasser par l'API).
    ENTSOE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    covering = _find_covering_cache(ENTSOE_CACHE_DIR, f"prices_{freq}", start, end)
    if covering is not None:
        print(f"[load_entsoe_data] cache hit, {len(covering)} points {freq}")
        return covering

    print(f"[load_entsoe_data] Requête ENTSO-E FR : {start} -> {end} (freq={freq})")

    client = EntsoePandasClient(api_key=ENTSOE_TOKEN)
    start_ts = pd.Timestamp(start, tz="Europe/Paris")
    end_ts = pd.Timestamp(end, tz="Europe/Paris") + pd.Timedelta(days=1)

    try:
        prices = client.query_day_ahead_prices("FR", start=start_ts, end=end_ts)
    except Exception as exc:
        print(f"[load_entsoe_data] ENTSO-E a refusé la requête ({exc}); retour vide.")
        return pd.DataFrame(columns=["time", "price"])

    print(f"[load_entsoe_data] {len(prices)} points reçus")
    prices = _resample_to_freq(prices, freq).reset_index()
    prices.columns = ["time", "price"]
    prices["time"] = pd.to_datetime(prices["time"]).dt.tz_localize(None)
    prices = prices.dropna(subset=["price"]).reset_index(drop=True)

    if not prices.empty:
        cache_path = ENTSOE_CACHE_DIR / f"prices_{freq}_{start}_{end}.parquet"
        prices.to_parquet(cache_path, index=False)
        print(
            f"[load_entsoe_data] Prix disponibles : {prices['time'].min()} -> {prices['time'].max()}"
            f" ({len(prices)} points {freq}) — cache écrit : {cache_path.name}"
        )
    return prices


def _resample_to_freq(s, freq: str):
    # Helper de resampling pour les séries ENTSO-E (Series ou DataFrame tz-aware).
    # - freq="1h"     : downsample (ou pas-à-pas) via .resample("1h").mean()
    # - freq="15min"  : pour upsampler du 1h vers 15-min, on prend .mean() (qui laisse NaN
    #                   aux nouveaux slots) puis ffill avec limite 3 pour répliquer
    #                   le prix horaire sur les 4 quart d'heure correspondants.
    if freq == "1h":
        return s.resample("1h").mean()
    return s.resample("15min").mean().ffill(limit=3)


ARCHIVE_LAG_DAYS = 6  # marge de sécurité sur le délai ~5j d'Open-Meteo archive

# Clé API ENTSO-E (Transparency Platform) : lue dans l'environnement, ou dans un
# fichier .env à la racine (non versionné, cf. .env.example).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
ENTSOE_TOKEN = os.environ.get("ENTSOE_API_KEY")

# Zones de marché frontalières de la France (libellés ENTSO-E -> suffixe de colonne).
# Toutes sont couplées au day-ahead français : leur prix porte l'info de tension
# du marché chez le voisin (interconnexions, imports/exports).
NEIGHBOUR_ZONES = {
    "DE_LU": "DE",   # Allemagne-Luxembourg
    "BE": "BE",      # Belgique
    "ES": "ES",      # Espagne
    "IT_NORD": "IT",  # Italie Nord (zone frontalière FR)
    "CH": "CH",      # Suisse
    "GB": "GB",      # Grande-Bretagne (IFA / ElecLink)
}


def load_entsoe_neighbour_prices(start: str, end: str, freq: str = "1h") -> pd.DataFrame:
    # Prix day-ahead des pays voisins de la France. Renvoie un DataFrame
    # 'time, price_DE, price_BE, price_ES, price_IT, price_CH, price_GB'.
    # Cache : fichier 'neighbours_{freq}_S_E.parquet' réutilisé si couvre la plage.
    ENTSOE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    covering = _find_covering_cache(ENTSOE_CACHE_DIR, f"neighbours_{freq}", start, end)
    if covering is not None:
        print(f"[load_entsoe_neighbours] cache hit, {len(covering)} points {freq}")
        return covering

    print(f"[load_entsoe_neighbours] {start} -> {end} (freq={freq})")
    client = EntsoePandasClient(api_key=ENTSOE_TOKEN)
    start_ts = pd.Timestamp(start, tz="Europe/Paris")
    end_ts = pd.Timestamp(end, tz="Europe/Paris") + pd.Timedelta(days=1)

    out = pd.DataFrame(columns=["time"])
    for zone, suffix in NEIGHBOUR_ZONES.items():
        col = f"price_{suffix}"
        try:
            p = client.query_day_ahead_prices(zone, start=start_ts, end=end_ts)
            p = _resample_to_freq(p, freq)
            p.index = p.index.tz_localize(None)
            dfp = pd.DataFrame({"time": p.index, col: p.values})
            out = out.merge(dfp, on="time", how="outer") if not out.empty else dfp
            print(f"[load_entsoe_neighbours] {zone} -> {col} : {len(dfp)} points")
        except Exception as exc:
            print(f"[load_entsoe_neighbours] {zone} indisponible ({exc})")

    if out.empty or "time" not in out.columns:
        cols = ["time"] + [f"price_{s}" for s in NEIGHBOUR_ZONES.values()]
        return pd.DataFrame(columns=cols)

    out = out.sort_values("time").reset_index(drop=True)
    cache_path = ENTSOE_CACHE_DIR / f"neighbours_{freq}_{start}_{end}.parquet"
    out.to_parquet(cache_path, index=False)
    print(f"[load_entsoe_neighbours] cache écrit : {cache_path.name} ({len(out)} points)")
    return out


# Ticker Yahoo Finance du future TTF (gaz naturel, référence européenne, en EUR/MWh).
# Le gaz fixe le prix marginal de l'électricité dès qu'une centrale CCGT est en
# marge -> c'est l'un des drivers les plus directs du prix day-ahead français.
GAS_TICKER = "TTF=F"


# Décalage de publication du gaz : le day-ahead électricité de J se clôt à midi
# en J-1, avant la clôture TTF de J-1. La dernière clôture connue est donc celle
# de J-2 -> on décale de 2 jours pour ne pas injecter d'info future.
GAS_PUBLICATION_LAG_DAYS = 2


def load_gas_price(start: str, end: str, freq: str = "1h") -> pd.DataFrame:
    # Prix du gaz TTF au pas freq, décalé de GAS_PUBLICATION_LAG_DAYS (cf. ci-dessus).
    fetch_start = (pd.Timestamp(start) - pd.Timedelta(days=GAS_PUBLICATION_LAG_DAYS)).strftime("%Y-%m-%d")
    gas = _load_gas_price_raw(fetch_start, end, freq=freq)
    if gas.empty:
        return gas
    gas = gas.copy()
    gas["time"] = pd.to_datetime(gas["time"]) + pd.Timedelta(days=GAS_PUBLICATION_LAG_DAYS)
    end_excl = pd.Timestamp(end) + pd.Timedelta(days=1)
    mask = (gas["time"] >= pd.Timestamp(start)) & (gas["time"] < end_excl)
    return gas.loc[mask].reset_index(drop=True)


def _load_gas_price_raw(start: str, end: str, freq: str = "1h") -> pd.DataFrame:
    # Prix du gaz TTF récupéré via Yahoo Finance. Yahoo ne sert que du journalier
    # (jours ouvrés) : on densifie au pas freq par ffill — la clôture d'un jour
    # s'applique aux pas du lendemain et la clôture du vendredi couvre le week-end.
    # Renvoie un DataFrame 'time, gas_price'.
    # Cache : fichier 'gas_{freq}_S_E.parquet' réutilisé s'il couvre la plage.
    GAS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    covering = _find_covering_cache(GAS_CACHE_DIR, f"gas_{freq}", start, end)
    if covering is not None:
        print(f"[load_gas_price] cache hit, {len(covering)} points {freq}")
        return covering

    print(f"[load_gas_price] Requête Yahoo Finance {GAS_TICKER} : {start} -> {end} (freq={freq})")
    step = "15min" if freq == "15min" else "1h"
    full_idx = pd.date_range(
        start, pd.Timestamp(end) + pd.Timedelta(days=1), freq=step, inclusive="left"
    )

    try:
        import yfinance as yf
        # On élargit de 5 jours côté start pour disposer d'une clôture antérieure
        # à ffill sur le premier jour (marché du gaz fermé week-end / jours fériés).
        hist = yf.Ticker(GAS_TICKER).history(
            start=(pd.Timestamp(start) - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
            end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        )
    except Exception as exc:
        print(f"[load_gas_price] Yahoo Finance indisponible ({exc}); retour vide.")
        return pd.DataFrame(columns=["time", "gas_price"])

    if hist.empty or "Close" not in hist.columns:
        print("[load_gas_price] Aucune donnée gaz reçue; retour vide.")
        return pd.DataFrame(columns=["time", "gas_price"])

    # Clôture journalière, ramenée en dates tz-naïves (minuit) pour s'aligner sur
    # le reste des données (heure locale Paris, tz-naïve).
    daily = hist["Close"].copy()
    daily.index = pd.to_datetime(daily.index).tz_localize(None).normalize()
    daily = daily[~daily.index.duplicated(keep="last")].sort_index()

    # Densification au pas freq : on ffill la clôture journalière sur tous les pas.
    gas = daily.reindex(daily.index.union(full_idx)).ffill().reindex(full_idx)
    out = pd.DataFrame({"time": full_idx, "gas_price": gas.values})
    out = out.dropna(subset=["gas_price"]).reset_index(drop=True)

    if not out.empty:
        cache_path = GAS_CACHE_DIR / f"gas_{freq}_{start}_{end}.parquet"
        out.to_parquet(cache_path, index=False)
        print(
            f"[load_gas_price] Gaz disponible : {out['time'].min()} -> {out['time'].max()}"
            f" ({len(out)} points {freq}) — cache écrit : {cache_path.name}"
        )
    return out


# ----------------------------------------------------------------------
# Nucléaire
# ----------------------------------------------------------------------
# Deux sources ENTSO-E, avec des garanties différentes sur la fuite d'information :
#  - la production nucléaire réalisée : celle de J-2 est publiée avant l'enchère
#    de J-1 midi -> aucune fuite. Le parc bouge lentement (arrêts de plusieurs
#    semaines), donc J-2 est un bon proxy de la disponibilité à J.
#  - les messages d'indisponibilité (REMIT) : l'API ne sert que la DERNIÈRE révision
#    de chaque message, et le stock a été republié (created_doc_time ≈ oct. 2025
#    même pour 2023) -> impossible de reconstituer ce qui était connu la veille.
#    On ne garde donc que les arrêts PLANIFIÉS (annoncés des semaines à l'avance) ;
#    seules leurs prolongations fuitent. Les arrêts fortuits sont exclus.
NUCLEAR_CACHE_DIR = DATA_DIR / "cache" / "nuclear"


def load_nuclear_features(start: str, end: str, freq: str = "1h") -> pd.DataFrame:
    # Renvoie 'time, nuclear_gen_d2, nuclear_gen_d2_trend, nuclear_planned_unavail'
    # au pas freq. Les colonnes manquantes (API indisponible) sont simplement absentes.
    step = "15min" if freq == "15min" else "1h"
    full_idx = pd.date_range(
        start, pd.Timestamp(end) + pd.Timedelta(days=1), freq=step, inclusive="left"
    )
    out = pd.DataFrame({"time": full_idx})

    gen = _nuclear_generation_daily(start, end)
    if gen is not None:
        # Moyenne journalière de J-2 et sa variation sur une semaine (J-2 vs J-9).
        feats = pd.DataFrame({
            "date": gen.index + pd.Timedelta(days=2),
            "nuclear_gen_d2": gen.values,
            "nuclear_gen_d2_trend": (gen - gen.shift(7, freq="D").reindex(gen.index)).values,
        })
        out["date"] = out["time"].dt.normalize()
        out = out.merge(feats, on="date", how="left").drop(columns="date")

    unavail = _nuclear_planned_unavailability(start, end)
    if unavail is not None:
        hourly = unavail.reindex(pd.date_range(full_idx.min().floor("h"),
                                               full_idx.max().floor("h"), freq="1h"),
                                 fill_value=0.0)
        out["nuclear_planned_unavail"] = hourly.reindex(out["time"].dt.floor("h")).values
    return out


def _nuclear_generation_daily(start: str, end: str) -> pd.Series | None:
    # Production nucléaire FR (MW), moyenne journalière en heure locale Paris.
    # On remonte 10 jours avant start pour disposer de J-2 et J-9 dès le premier jour.
    NUCLEAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fetch_start = (pd.Timestamp(start) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    covering = _find_covering_cache(NUCLEAR_CACHE_DIR, "gen", fetch_start, end)
    if covering is None:
        print(f"[load_nuclear] Production nucléaire ENTSO-E : {fetch_start} -> {end}")
        client = EntsoePandasClient(api_key=ENTSOE_TOKEN)
        try:
            g = client.query_generation(
                "FR",
                start=pd.Timestamp(fetch_start, tz="Europe/Paris"),
                end=pd.Timestamp(end, tz="Europe/Paris") + pd.Timedelta(days=1),
                psr_type="B14",  # B14 = nucléaire
            )
        except Exception as exc:
            print(f"[load_nuclear] production indisponible ({exc})")
            return None
        if isinstance(g, pd.DataFrame):
            g = g.iloc[:, 0]
        g = g.resample("1h").mean()
        g.index = g.index.tz_convert("Europe/Paris").tz_localize(None)
        covering = pd.DataFrame({"time": g.index, "nuclear_gen": g.values})
        covering.to_parquet(NUCLEAR_CACHE_DIR / f"gen_{fetch_start}_{end}.parquet", index=False)
    s = covering.set_index("time")["nuclear_gen"]
    return s.resample("D").mean()


def _nuclear_planned_unavailability(start: str, end: str) -> pd.Series | None:
    # MW nucléaires en arrêt planifié, par heure. Un même réacteur peut apparaître
    # dans plusieurs messages qui se chevauchent : on prend le max par réacteur et
    # par heure, puis on somme sur le parc.
    NUCLEAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = NUCLEAR_CACHE_DIR / f"unavail_{start}_{end}.parquet"
    if cache_path.exists():
        msgs = pd.read_parquet(cache_path)
    else:
        msgs = _fetch_nuclear_unavailability_messages(start, end)
        if msgs is None:
            return None
        msgs.to_parquet(cache_path, index=False)

    msgs = msgs[(msgs["businesstype"] == "Planned maintenance")
                & (msgs["docstatus"].fillna("") != "Cancelled")]
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) + pd.Timedelta(days=1)
    rows = []
    for m in msgs.itertuples(index=False):
        s, e = max(m.start, lo), min(m.end, hi)
        if s >= e:
            continue
        hours = pd.date_range(s.floor("h"), e, freq="1h", inclusive="left")
        mw = float(m.nominal_power) - float(m.avail_qty)
        rows.append(pd.DataFrame({"unit": m.production_resource_name, "time": hours, "mw": mw}))
    if not rows:
        return pd.Series(dtype=float)
    per_unit = pd.concat(rows).groupby(["unit", "time"])["mw"].max()
    total = per_unit.groupby("time").sum().clip(lower=0.0)
    print(f"[load_nuclear] Arrêts planifiés : {len(msgs)} messages, "
          f"moyenne {total.mean():.0f} MW indisponibles")
    return total


def _fetch_nuclear_unavailability_messages(start: str, end: str) -> pd.DataFrame | None:
    # Requêtes mois par mois (l'API plafonne le nombre de documents par appel et
    # répond lentement) ; chaque mois est mis en cache pour pouvoir reprendre.
    client = EntsoePandasClient(api_key=ENTSOE_TOKEN)
    month_dir = NUCLEAR_CACHE_DIR / "unavail_months"
    month_dir.mkdir(parents=True, exist_ok=True)
    months = pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS")
    parts = []
    for m0 in months:
        month_path = month_dir / f"{m0:%Y-%m}.pkl"
        if month_path.exists():
            parts.append(pd.read_pickle(month_path))
            continue
        m1 = m0 + pd.offsets.MonthBegin(1)
        print(f"[load_nuclear] Indisponibilités {m0:%Y-%m}")
        try:
            u = client.query_unavailability_of_generation_units(
                "FR", start=pd.Timestamp(m0, tz="Europe/Paris"),
                end=pd.Timestamp(m1, tz="Europe/Paris"), docstatus=None,
            )
        except Exception as exc:
            print(f"[load_nuclear] {m0:%Y-%m} indisponible ({exc})")
            continue
        nuclear = u[u["plant_type"] == "Nuclear"].reset_index()
        if m1 <= pd.Timestamp.today().normalize():  # mois en cours : pas de cache (incomplet)
            nuclear.to_pickle(month_path)
        parts.append(nuclear)
    if not parts:
        return None
    msgs = pd.concat(parts, ignore_index=True)
    # Un message chevauchant plusieurs mois est renvoyé plusieurs fois : on garde
    # sa révision la plus récente.
    msgs = msgs.sort_values("revision").drop_duplicates("mrid", keep="last")
    for col in ("start", "end"):
        msgs[col] = pd.to_datetime(msgs[col], utc=True).dt.tz_convert("Europe/Paris").dt.tz_localize(None)
    msgs["avail_qty"] = pd.to_numeric(msgs["avail_qty"], errors="coerce").fillna(0.0)
    msgs["nominal_power"] = pd.to_numeric(msgs["nominal_power"], errors="coerce")
    keep = ["mrid", "revision", "businesstype", "docstatus", "production_resource_name",
            "nominal_power", "avail_qty", "start", "end"]
    return msgs[keep].dropna(subset=["nominal_power"]).reset_index(drop=True)


# Filières du registre national retenues, et nom de la colonne de capacité produite.
CAPACITY_FILIERES = {"Solaire": "solar_capacity", "Eolien": "wind_capacity"}


def load_installed_capacity() -> pd.DataFrame:
    # Lit le registre national des installations et renvoie la capacité installée
    # CUMULÉE (en MW) solaire et éolien, par département et par date de mise en
    # service. Chaque ligne = un palier : la capacité d'un département à un instant t
    # s'obtient par merge_asof (somme des mises en service <= t), ce qui évite
    # d'injecter la capacité future dans une prédiction passée.
    if not REGISTRY_PATH.exists():
        print(f"[load_installed_capacity] registre absent ({REGISTRY_PATH}); capacités ignorées.")
        return pd.DataFrame(columns=["code", "date", "solar_capacity", "wind_capacity"])

    # Deux formats d'export coexistent (data.gouv en camelCase, ODRE en minuscules) :
    # on repère les colonnes utiles par leur nom normalisé.
    aliases = {
        "codedepartement": "code",
        "filiere": "filiere",
        "puismaxinstallee": "puisMaxInstallee",
        "datemiseenservice (format date)": "date",
        "datemiseenservice_date": "date",
    }
    header = pd.read_csv(REGISTRY_PATH, sep=";", encoding="utf-8-sig", nrows=0).columns
    rename = {c: aliases[c.lower()] for c in header if c.lower() in aliases}
    reg = pd.read_csv(
        REGISTRY_PATH, sep=";", encoding="utf-8-sig", usecols=list(rename),
        dtype=str, low_memory=False,
    )
    reg = reg.rename(columns=rename)
    reg = reg[reg["filiere"].isin(CAPACITY_FILIERES)].copy()
    reg["date"] = pd.to_datetime(reg["date"], errors="coerce")
    reg["puisMaxInstallee"] = pd.to_numeric(reg["puisMaxInstallee"], errors="coerce")
    reg = reg.dropna(subset=["code", "date", "puisMaxInstallee"])
    # puisMaxInstallee est en kW dans le registre -> MW (homogène aux forecasts ENTSO-E).
    reg["mw"] = reg["puisMaxInstallee"] / 1000.0
    reg["kind"] = reg["filiere"].map(CAPACITY_FILIERES)

    # Un palier par (département, date) : delta de capacité solaire / éolien ce jour.
    events = (
        reg.pivot_table(index=["code", "date"], columns="kind", values="mw", aggfunc="sum")
        .reset_index()
        .sort_values(["code", "date"])
    )
    for col in CAPACITY_FILIERES.values():
        if col not in events.columns:
            events[col] = 0.0
        # Cumul chronologique au sein du département.
        events[col] = events.groupby("code")[col].cumsum().ffill().fillna(0.0)

    print(
        f"[load_installed_capacity] {len(events)} paliers — "
        f"solaire {events.groupby('code')['solar_capacity'].last().sum():.0f} MW, "
        f"éolien {events.groupby('code')['wind_capacity'].last().sum():.0f} MW (capacité actuelle)"
    )
    return events[["code", "date", "solar_capacity", "wind_capacity"]]


def _attach_installed_capacity(df_meteo: pd.DataFrame) -> pd.DataFrame:
    # Ajoute à df_meteo (lignes (time, code)) les colonnes 'solar_capacity' et
    # 'wind_capacity' = capacité installée du département à cette date, via un
    # merge_asof backward sur la date de mise en service.
    cap = load_installed_capacity()
    if cap.empty:
        return df_meteo

    df_meteo = df_meteo.copy()
    df_meteo["time"] = pd.to_datetime(df_meteo["time"])
    left = df_meteo.sort_values("time").reset_index(drop=True)
    cap = cap.sort_values("date").reset_index(drop=True)
    merged = pd.merge_asof(
        left, cap,
        left_on="time", right_on="date",
        by="code", direction="backward",
    )
    # Avant la 1re mise en service connue d'un département -> capacité nulle.
    merged["solar_capacity"] = merged["solar_capacity"].fillna(0.0)
    merged["wind_capacity"] = merged["wind_capacity"].fillna(0.0)
    return merged.drop(columns=["date"])


def load_entsoe_forecasts(start: str, end: str, freq: str = "1h") -> pd.DataFrame:
    # Récupère les prévisions ENTSO-E pour FR : conso (load) + éolien + solaire,
    # au pas freq ("1h" ou "15min"). Ces prévisions sont le signal le plus prédictif
    # du prix DA (équilibre offre/demande anticipé).
    # Cache : si un fichier 'forecasts_{freq}_S_E.parquet' couvre [start, end],
    # on en sert un slice (sans repasser par l'API).
    ENTSOE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    covering = _find_covering_cache(ENTSOE_CACHE_DIR, f"forecasts_{freq}", start, end)
    if covering is not None:
        print(f"[load_entsoe_forecasts] cache hit, {len(covering)} points {freq}")
        return covering

    print(f"[load_entsoe_forecasts] FR : {start} -> {end} (freq={freq})")

    client = EntsoePandasClient(api_key=ENTSOE_TOKEN)
    start_ts = pd.Timestamp(start, tz="Europe/Paris")
    end_ts = pd.Timestamp(end, tz="Europe/Paris") + pd.Timedelta(days=1)

    out = pd.DataFrame(columns=["time"])

    try:
        load_fc = client.query_load_forecast("FR", start=start_ts, end=end_ts)
        if isinstance(load_fc, pd.DataFrame):
            load_fc = load_fc.iloc[:, 0]
        load_fc = _resample_to_freq(load_fc, freq)
        load_fc.index = load_fc.index.tz_localize(None)
        out = pd.DataFrame({"time": load_fc.index, "load_forecast": load_fc.values})
        print(f"[load_entsoe_forecasts] load_forecast : {len(load_fc)} points {freq}")
    except Exception as exc:
        print(f"[load_entsoe_forecasts] load_forecast indisponible ({exc})")

    try:
        ws = client.query_wind_and_solar_forecast("FR", start=start_ts, end=end_ts)
        ws = _resample_to_freq(ws, freq)
        ws.index = ws.index.tz_localize(None)

        # Les colonnes selon les libellés ENTSO-E ('Solar', 'Wind Onshore', 'Wind Offshore' ou 'Wind').
        cols = {c.lower(): c for c in ws.columns}
        solar = ws[cols["solar"]] if "solar" in cols else pd.Series(0.0, index=ws.index)
        wind = pd.Series(0.0, index=ws.index)
        for key, col in cols.items():
            if "wind" in key:
                wind = wind.add(ws[col].fillna(0.0), fill_value=0.0)

        df_ws = pd.DataFrame({
            "time": ws.index,
            "solar_forecast": solar.values,
            "wind_forecast": wind.values,
        })
        out = out.merge(df_ws, on="time", how="outer") if not out.empty else df_ws
        print(f"[load_entsoe_forecasts] wind_solar_forecast : {len(ws)} points {freq}")
    except Exception as exc:
        print(f"[load_entsoe_forecasts] wind_solar_forecast indisponible ({exc})")

    if out.empty:
        return pd.DataFrame(columns=["time", "load_forecast", "wind_forecast", "solar_forecast"])

    out = out.sort_values("time").reset_index(drop=True)
    cache_path = ENTSOE_CACHE_DIR / f"forecasts_{freq}_{start}_{end}.parquet"
    out.to_parquet(cache_path, index=False)
    print(f"[load_entsoe_forecasts] cache écrit : {cache_path.name} ({len(out)} points)")
    return out


def load_meteo_france(
    start: str,
    end: str,
    geometry,
    code: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    max_retries: int = 12,
) -> pd.DataFrame:
    # start / end au format ISO "YYYY-MM-DD" (inclus)
    # Bascule automatiquement entre l'API archive (ERA5, retard ~5 jours) et l'API forecast
    # (qui couvre les derniers jours + prochains jours). On découpe la plage en deux si besoin.
    # lat/lon : coordonnées de la préfecture si fournies, sinon point représentatif du polygone.
    if lat is not None and lon is not None:
        latitude, longitude = float(lat), float(lon)
    else:
        point = geometry.representative_point()
        latitude, longitude = point.y, point.x

    start_d = pd.to_datetime(start).date()
    end_d = pd.to_datetime(end).date()
    today = pd.Timestamp.today().normalize().date()
    archive_cutoff = today - pd.Timedelta(days=ARCHIVE_LAG_DAYS)

    # 1) On cherche d'abord un cache dont la plage couvre [start, end] : si un
    #    fichier 'dep_{code}_S_E.parquet' avec S<=start et E>=end existe, on en
    #    sert un slice (cas typique : l'utilisateur ressort une sous-plage d'un
    #    cache déjà téléchargé sur une plage plus large).
    cache_path = None
    cached_archive: pd.DataFrame | None = None
    if code is not None:
        METEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        covering = _find_covering_cache(METEO_CACHE_DIR, f"dep_{code}", start, end)
        if covering is not None and not covering.empty:
            return covering

        # 2) Sinon : si un cache pour la plage exacte existe mais est partiel (max < end),
        #    on en récupère la portion archive et on re-télécharge la partie forecast.
        cache_path = METEO_CACHE_DIR / f"dep_{code}_{start}_{end}.parquet"
        if cache_path.exists():
            cached_full = pd.read_parquet(cache_path)
            cached_max = pd.to_datetime(cached_full["time"]).max().date()
            print(
                f"[load_meteo_france] cache partiel ({cached_max}<{end_d}) -> "
                f"refetch forecast"
            )
            cutoff_ts = pd.Timestamp(archive_cutoff) + pd.Timedelta(hours=23, minutes=59)
            cached_archive = cached_full[pd.to_datetime(cached_full["time"]) <= cutoff_ts]
        elif end_d > archive_cutoff:
            # 3) Pas de cache exact, mais peut-être un cache plus large qui couvre
            #    au moins la portion archive [start, archive_cutoff]. Si oui, on
            #    n'a plus qu'à re-télécharger la portion forecast récente.
            archive_covering = _find_covering_cache(
                METEO_CACHE_DIR, f"dep_{code}",
                start, archive_cutoff.isoformat(),
            )
            if archive_covering is not None and not archive_covering.empty:
                cached_archive = archive_covering
                print(
                    f"[load_meteo_france] cache archive partiel hit "
                    f"(archive {start}..{archive_cutoff} servi depuis le cache) "
                    f"-> refetch forecast uniquement"
                )

    chunks: list[pd.DataFrame] = []
    if start_d <= archive_cutoff:
        if cached_archive is not None and not cached_archive.empty:
            chunks.append(cached_archive)
        else:
            archive_end = min(end_d, archive_cutoff)
            chunks.append(
                _fetch_meteo(
                    "https://archive-api.open-meteo.com/v1/archive",
                    latitude, longitude,
                    start_d.isoformat(), archive_end.isoformat(),
                    max_retries,
                )
            )
    if end_d > archive_cutoff:
        forecast_start = max(start_d, archive_cutoff + pd.Timedelta(days=1))
        chunks.append(
            _fetch_meteo(
                "https://api.open-meteo.com/v1/forecast",
                latitude, longitude,
                forecast_start.isoformat(), end_d.isoformat(),
                max_retries,
            )
        )

    df = (
        pd.concat(chunks, ignore_index=True)
        .drop_duplicates(subset="time")
        .sort_values("time")
        .reset_index(drop=True)
    )

    if cache_path is not None:
        df.to_parquet(cache_path, index=False)
        print(f"[load_meteo_france] cache write -> {cache_path.name}")

    return df


def _fetch_meteo(
    url: str,
    latitude: float,
    longitude: float,
    start: str,
    end: str,
    max_retries: int,
) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation,cloud_cover,precipitation",
        "timezone": "Europe/Paris",
    }

    for attempt in range(1, max_retries + 1):
        r = requests.get(url, params=params)

        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            wait = int(retry_after) if retry_after and retry_after.isdigit() else min(180 * attempt, 300)
            print(
                f"[_fetch_meteo] HTTP 429 (rate-limit) tentative {attempt}/{max_retries}, "
                f"attente {wait}s..."
            )
            time.sleep(wait)
            continue

        if r.status_code != 200:
            raise RuntimeError(
                f"[_fetch_meteo] HTTP {r.status_code} ({url} {start}->{end}) : {r.text}"
            )

        data = r.json()
        if "hourly" not in data:
            reason = data.get("reason") or data.get("error") or data
            raise RuntimeError(
                f"[_fetch_meteo] Réponse API invalide ({url} {start}->{end}) : {reason}."
            )

        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        return df

    raise RuntimeError(
        f"[_fetch_meteo] Rate-limit non résolu après {max_retries} tentatives ({url} {start}->{end})."
    )