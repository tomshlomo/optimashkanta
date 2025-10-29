import numpy as np
import pandas as pd

from optimashkanta.model import Cols
from optimashkanta.model import EconomicPrediction


EYAL_PATH = "/Users/tom.shlomo/workspace/personal/optimashkanta/src/optimashkanta/data/eyal_data/שלמה_1397896.5_7120_1.7_11.01_מידע_מלא.xls"
THETA_PATH = "/Users/tom.shlomo/workspace/personal/" "optimashkanta/data/theta.csv"

CASE_TO_SHEET = {
    1.0: 4,
    0.5: 5,
    0.75: 6,
    0.88: 7,
    1.12: 8,
    1.25: 9,
    1.5: 10,
}


def load_rbi_and_inflation(scenario: float) -> pd.DataFrame:
    df = pd.read_excel(
        EYAL_PATH,
        sheet_name=CASE_TO_SHEET[scenario],
        usecols="C:D",
        nrows=360,
    )
    df.columns = [Cols.RBI, Cols.INFLATION]
    return df * 100


def load_ogen(scenario: float) -> pd.DataFrame:
    df = pd.read_excel(
        EYAL_PATH,
        sheet_name=CASE_TO_SHEET[scenario],
        usecols="G:I",
        nrows=6,
    )
    df.columns = [Cols.MONTH, Cols.OGEN_TZMOODA, Cols.OGEN_LO_TZMOODA]
    return df


def interpolate_ogen(df_ogen: pd.DataFrame, df: pd.DataFrame) -> None:
    for col in [Cols.OGEN_TZMOODA, Cols.OGEN_LO_TZMOODA]:
        df[col] = np.interp(
            df.index.values,
            df_ogen[Cols.MONTH].values,
            df_ogen.loc[:, Cols.OGEN_TZMOODA].values,
        )


def load_theta() -> pd.DataFrame:
    return pd.read_csv(THETA_PATH, index_col=0)


def predict_avgs_by_theta(df: pd.DataFrame, theta: pd.DataFrame) -> None:
    df["one"] = 1.0
    x = df.loc[:, [Cols.RBI, "one"]]
    y = pd.DataFrame(x.values @ theta.values, columns=theta.columns)
    df.loc[:, theta.columns] = y
    df.loc[:, theta.columns] = 2.0
    pass


def add_deflation_coef(df: pd.DataFrame):
    df[Cols.DEFLATION_COEF] = np.insert(
        np.cumprod(1 / (1 + df.loc[df.index[:-1], Cols.INFLATION].values / 12)),
        0,
        1.0,
        axis=0,
    )


def load_economic_prediction(scenario: float) -> EconomicPrediction:
    df_ogen = load_ogen(scenario)
    df = load_rbi_and_inflation(scenario)
    interpolate_ogen(df_ogen, df)
    theta = load_theta()
    predict_avgs_by_theta(df, theta)
    df = df / 100
    add_deflation_coef(df)
    return df


def load_all_economic_predictions() -> dict[float, EconomicPrediction]:
    return {scenario: load_economic_prediction(scenario) for scenario in CASE_TO_SHEET}
