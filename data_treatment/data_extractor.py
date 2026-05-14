import cdsapi
from typing import List 
import pandas as pd 
import os 
import geopandas as gpd
import xarray as xr


VARIABLE_LIST = ["near_surface_wind_speed","precipitation","near_surface_air_temperature","surface_downwelling_longwave_radiation"]


def download_copernicus_data(variable : str, area : List,output_dir : str):
        
    dataset = "projections-cmip6"
    request = {
        "temporal_resolution": "monthly",
        "experiment": "ssp5_8_5",
        "variable": variable,
        "model": "ec_earth3_cc",
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "year": [
            "2015", "2016", "2017",
            "2018", "2019", "2020",
            "2021", "2022", "2023",
            "2024", "2025", "2026",
            "2027", "2028", "2029",
            "2030", "2031", "2032",
            "2033", "2034", "2035",
            "2036", "2037", "2038",
            "2039", "2040", "2041",
            "2042", "2043", "2044",
            "2045", "2046", "2047",
            "2048", "2049", "2050",
           
        ],
        "area": area
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download(f"{output_dir}/{variable}.zip")
    dezip_data(f"{output_dir}/{variable}.zip")

def dezip_data(file_path : str):
    import zipfile
    import shutil
    output_dir = os.path.dirname(file_path)
    variable = os.path.basename(file_path).replace(".zip", "")
    tmp_dir = os.path.join(output_dir, f"_tmp_{variable}")
    os.makedirs(tmp_dir, exist_ok=True)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)
    os.remove(file_path)
    nc_files = [f for f in os.listdir(tmp_dir) if f.endswith(".nc")]
    if nc_files:
        os.rename(os.path.join(tmp_dir, nc_files[0]), os.path.join(output_dir, f"{variable}.nc"))
    shutil.rmtree(tmp_dir)

def get_area_from_departement(departement_code: str, buffer: float = 1) -> List[float]:
    """Calcule [Nord, Ouest, Sud, Est] depuis la géométrie du département + buffer."""
    departements_path = "/Users/maximeanno/Documents/hydros_refactos/data/departements_data/departements.geojson"
    departements_gdf = gpd.read_file(departements_path)
    departement_gdf = departements_gdf[departements_gdf["code"].isin(departement_code)]
    if departement_gdf.crs is None:
        departement_gdf = departement_gdf.set_crs("EPSG:4326")
    else:
        departement_gdf = departement_gdf.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = departement_gdf.total_bounds
    # Format Copernicus : [Nord, Ouest, Sud, Est]
    return [round(maxy + buffer, 4), round(minx - buffer, 4), round(miny - buffer, 4), round(maxx + buffer, 4)]


def join_data_and_save_csv(dep_list : List[str]):
    departements_path = "/Users/maximeanno/Documents/hydros_refactos/data/departements_data/departements.geojson"
    departements_gdf = gpd.read_file(departements_path)
    if departements_gdf.crs is None:
        departements_gdf = departements_gdf.set_crs("EPSG:4326")
    else:
        departements_gdf = departements_gdf.to_crs("EPSG:4326")

    # Charger les données NetCDF
    data_path = f"/Users/maximeanno/Documents/ProjetPerso/EnergeticPricing/data/copernicus"
    df = pd.DataFrame()
    for variable in VARIABLE_LIST:
        nc_file = f"{data_path}/{variable}.nc"
        ds = xr.open_dataset(nc_file,engine="netcdf4")
        df_variable = ds.to_dataframe().reset_index()
        if df.empty:
            df = df_variable
        else:
            df = pd.merge(df, df_variable, on=["lat","time","lon","lon_bnds","lat_bnds","time_bnds","bnds"], how="inner")
    if df["lon"].max() > 180:
        df["lon"] = df["lon"].apply(lambda x: x - 360 if x > 180 else x)

    # Convertir les coordonnées en géométrie
    geometry = gpd.points_from_xy(df["lon"], df["lat"])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    # Garder time, lat, lon + les variables climatiques + geometry
    data_cols = [col for col in gdf.columns if col in ['time', 'lat', 'lon', 'sfcWind', 'pr', 'tas', 'rlds', 'geometry']]
    gdf = gdf[data_cols]

    # Jointure spatiale par département puis concaténation
    all_results = []
    for num in dep_list:
        departement_gdf = departements_gdf[departements_gdf["code"] == num]
        joined_gdf = gpd.sjoin_nearest(gdf, departement_gdf, how="inner")

        cols_to_drop = [c for c in ["index_right", "nom", "geometry"] if c in joined_gdf.columns]
        joined_gdf = joined_gdf.drop(columns=cols_to_drop)
        joined_gdf = joined_gdf.rename(columns={"code": "departement"})
        all_results.append(joined_gdf)
        print(f"  Département {num} : {len(joined_gdf)} lignes")

    result = pd.concat(all_results, ignore_index=True)

    result = result.rename(columns={"time": "date", "lat": "latitude", "lon": "longitude"})

    # Sauvegarder
    output_csv_path = f"/Users/maximeanno/Documents/ProjetPerso/EnergeticPricing/data/copernicus/monthly_departement.csv"
    result.to_csv(output_csv_path, index=False)
    print(f"Sauvegardé : {output_csv_path} ({len(result)} lignes)")

def create_copernicus_data_csv():

    dep_list = [f"{s:02d}" for s in range(1,96)]

    
        
    area = get_area_from_departement(dep_list)
    print(f"Bounding box calculée : {area}")
    output_dir = f"/Users/maximeanno/Documents/ProjetPerso/EnergeticPricing/data/copernicus"
    print(f"\nCréation du dossier : {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    for variable in VARIABLE_LIST:
        print(f"Téléchargement de la variable : {variable}")
        download_copernicus_data(variable, area, output_dir)
        print(f"[ Variable '{variable}' téléchargée.")
    print(f"Jointure spatiale et sauvegarde CSV...")
    join_data_and_save_csv(dep_list)
    print(f"CSV sauvegardé.")

    print("\nTraitement terminé.")



if __name__ == "__main__":

   create_copernicus_data_csv()
   