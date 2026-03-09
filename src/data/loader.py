from pathlib import Path
from src.setup.config import RTE_FILE_YEARS, SOURCES
from src.setup.constant import RTE_FILE_ROOT_NAME
import src.data.utils as utils
import pandas as pd 


def load(steps:list,data_path: Path = Path("data")) -> pd.DataFrame:
    df_list = []

    for year in RTE_FILE_YEARS:
        df = _load_RTE_data(data_path, year)
        df = utils.clean_RTE_data(df, SOURCES)
        df = _load_datetime(steps,df)
        df_list.append(df)

    return pd.concat(df_list, ignore_index=True)


def _load_RTE_data(data_path: Path, year: str) -> pd.DataFrame:
    path = data_path / (RTE_FILE_ROOT_NAME + year + ".xlsx")
    return pd.read_excel(path)


def _load_datetime(steps : list, df: pd.DataFrame) -> pd.DataFrame:
    for step in steps:
    
        df[step] = getattr(df["Date"].dt, step)
    return df