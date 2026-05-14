from pathlib import Path
import time
from src.setup.config import RTE_FILE_YEARS, SOURCES
from src.setup.constant import RTE_FILE_ROOT_NAME,PRICE_FILE_NAME
import src.data.utils as utils
import pandas as pd
import geopandas as gpd
import requests
from entsoe import EntsoePandasClient

METEO_CACHE_DIR = Path("data/cache/meteo")

def load_RTE(steps:list,data_path: Path = Path("data")) -> pd.DataFrame:
    df_list = []

    for year in RTE_FILE_YEARS:
        df = _load_RTE_data(data_path, year)
        df = utils.clean_RTE_data(df, SOURCES)
        df = _load_datetime(steps,df)
        df_list.append(df)

    return pd.concat(df_list, ignore_index=True)


def load_price(data_path: Path = Path("data")) -> pd.DataFrame:

    price_path = data_path / PRICE_FILE_NAME
    df = pd.read_csv(price_path)
    df=df[["Datetime (Local)","Price (EUR/MWhe)"]]
    df =df.rename(columns={"Datetime (Local)" : "Datetime"})
    df["Price (EUR/MWhe)"] = df["Price (EUR/MWhe)"].astype(float)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["year"] = df["Datetime"].dt.year
    df["month"] = df["Datetime"].dt.month
    df["hour"] = df["Datetime"].dt.hour
    df["day"] = df["Datetime"].dt.day
    
    return(df)



def _load_RTE_data(data_path: Path, year: str) -> pd.DataFrame:
    path = data_path / (RTE_FILE_ROOT_NAME + year + ".xlsx")
    return pd.read_excel(path)


def _load_datetime(steps : list, df: pd.DataFrame) -> pd.DataFrame:

    for step in steps:
    
        df[step] = getattr(df["Date"].dt, step)
    return df


def load_data(start: str, end: str) -> pd.DataFrame:
    # start / end au format ISO "YYYY-MM-DD"
    print(f"[load_data] Démarrage : start={start}, end={end}")

    geo_path = "/Users/maximeanno/Documents/ProjetPerso/EnergeticPricing/data/Geographie/departements.geojson"
    print(f"[load_data] Lecture du geojson : {geo_path}")
    departements = gpd.read_file(geo_path)
    DEP_CODE = [str(i).zfill(2) for i in range(1, 96) if i != 20] + ["2A", "2B"]
    departements = departements[departements["code"].isin(DEP_CODE)]
    print(f"[load_data] {len(departements)} départements à traiter")

    df_meteo_list = []
    for i, code in enumerate(DEP_CODE, start=1):
        departement = departements[departements["code"] == code]
        if departement.empty:
            print(f"[load_data] ({i}/{len(DEP_CODE)}) dep {code} introuvable dans le geojson, skip")
            continue
        print(f"[load_data] ({i}/{len(DEP_CODE)}) météo dep {code}...")
        df_dep = load_meteo_france(start, end, departement, code=code)
        df_dep["code"] = code
        df_meteo_list.append(df_dep)

    print(f"[load_data] Concat de {len(df_meteo_list)} dataframes météo")
    df_meteo = pd.concat(df_meteo_list, ignore_index=True)

    print("[load_data] Chargement des prix ENTSO-E")
    df_price = load_entsoe_data(start, end)

    # Left-merge depuis la météo : on garde toutes les heures pour lesquelles on a la météo,
    # le prix peut être NaN (typiquement les heures les plus récentes que l'ENTSO-E
    # n'a pas encore publiées). On veut pouvoir prédire ces heures-là.
    print(f"[load_data] Merge météo ({len(df_meteo)} lignes) <-> prix ({len(df_price)} lignes)")
    df = df_meteo.merge(df_price, on="time", how="left")
    n_price = df["price"].notna().sum()
    print(
        f"[load_data] Terminé : {len(df)} lignes (dont {n_price} avec un prix réel ENTSO-E)"
    )
    return df




def load_entsoe_data(start: str, end: str) -> pd.DataFrame:
    # start / end au format ISO "YYYY-MM-DD"
    # ENTSO-E renvoie du 15 min en FR depuis oct. 2025 -> resample horaire (moyenne).
    # Le day-ahead n'est publié que la veille vers 13h CET : un appel jusqu'à "aujourd'hui"
    # peut ne renvoyer les prix que jusqu'à hier 23h (voire aujourd'hui 23h selon l'horaire d'appel).
    # On élargit la fenêtre côté droite d'un jour pour bien récupérer la dernière journée publiée.
    print(f"[load_entsoe_data] Requête ENTSO-E FR : {start} -> {end}")

    TOKEN = "REDACTED_ENTSOE_KEY"

    client = EntsoePandasClient(api_key=TOKEN)
    start_ts = pd.Timestamp(start, tz="Europe/Paris")
    # ENTSO-E exclut la borne droite : on ajoute 1 jour pour couvrir l'intégralité du dernier jour demandé.
    end_ts = pd.Timestamp(end, tz="Europe/Paris") + pd.Timedelta(days=1)

    try:
        prices = client.query_day_ahead_prices("FR", start=start_ts, end=end_ts)
    except Exception as exc:
        print(f"[load_entsoe_data] ENTSO-E a refusé la requête ({exc}); retour vide.")
        return pd.DataFrame(columns=["time", "price"])

    print(f"[load_entsoe_data] {len(prices)} points reçus (avant resample)")
    prices = prices.resample("1h").mean()
    print(f"[load_entsoe_data] {len(prices)} points après resample horaire")

    prices = prices.reset_index()
    prices.columns = ["time", "price"]
    prices["time"] = pd.to_datetime(prices["time"]).dt.tz_localize(None)
    # On droppe les NaN pour ne pas créer de fausses "heures réelles" lors du merge.
    prices = prices.dropna(subset=["price"]).reset_index(drop=True)

    if not prices.empty:
        print(
            f"[load_entsoe_data] Prix disponibles : {prices['time'].min()} -> {prices['time'].max()}"
        )
    return prices


ARCHIVE_LAG_DAYS = 6  # marge de sécurité sur le délai ~5j d'Open-Meteo archive


def load_meteo_france(
    start: str,
    end: str,
    departement: gpd.GeoDataFrame,
    code: str | None = None,
    max_retries: int = 5,
) -> pd.DataFrame:
    # start / end au format ISO "YYYY-MM-DD" (inclus)
    # Bascule automatiquement entre l'API archive (ERA5, retard ~5 jours) et l'API forecast
    # (qui couvre les derniers jours + prochains jours). On découpe la plage en deux si besoin.
    point = departement.geometry.representative_point().iloc[0]
    latitude, longitude = point.y, point.x

    start_d = pd.to_datetime(start).date()
    end_d = pd.to_datetime(end).date()
    today = pd.Timestamp.today().normalize().date()
    archive_cutoff = today - pd.Timedelta(days=ARCHIVE_LAG_DAYS)

    # On cache la portion archive (figée dans le temps) ; la portion forecast est
    # re-téléchargée à chaque appel car elle change tous les jours.
    cache_path = None
    cached_archive: pd.DataFrame | None = None
    if code is not None:
        METEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = METEO_CACHE_DIR / f"dep_{code}_{start}_{end}.parquet"
        if cache_path.exists():
            cached_full = pd.read_parquet(cache_path)
            cached_max = pd.to_datetime(cached_full["time"]).max().date()
            if cached_max >= end_d:
                # Le cache couvre déjà toute la plage demandée (cas purement historique).
                print(f"[load_meteo_france] cache hit -> {cache_path.name}")
                return cached_full
            # Sinon : on récupère uniquement la portion archive depuis le cache,
            # et on re-télécharge la partie forecast.
            print(
                f"[load_meteo_france] cache partiel ({cached_max}<{end_d}) -> "
                f"refetch forecast"
            )
            cutoff_ts = pd.Timestamp(archive_cutoff) + pd.Timedelta(hours=23, minutes=59)
            cached_archive = cached_full[pd.to_datetime(cached_full["time"]) <= cutoff_ts]

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
            wait = int(retry_after) if retry_after and retry_after.isdigit() else min(60 * attempt, 300)
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