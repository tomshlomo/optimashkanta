from itertools import product
from typing import TypeAlias

import numpy as np
import pandas as pd

from optimashkanta.model import Cols
from optimashkanta.model import EconomicPrediction
from optimashkanta.model import get_avg_col_from_duration


ArrayLike: TypeAlias = pd.Series | pd.DataFrame


def all_close(
    val: ArrayLike, expected: ArrayLike, tol: ArrayLike | float | int
) -> bool:
    return bool(((val - expected).abs() <= tol).all().all())


def load_main_prediction() -> EconomicPrediction:
    df = pd.read_csv("src/optimashkanta/tests/main_prediction.csv", header=None)
    df.columns = [Cols.RBI, Cols.INFLATION, Cols.OGEN_TZMOODA, Cols.OGEN_LO_TZMOODA]
    df /= 100
    add_all_avg_cols(df)
    df2 = pd.DataFrame(np.repeat(df.loc[[359], :].values, 359, 0), columns=df.columns)
    df = pd.concat([df, df2], ignore_index=True)
    return df  # type:ignore[no-any-return]


def get_all_avg_cols() -> set[str]:
    return {
        get_avg_col_from_duration(i, is_tzmooda)
        for i, is_tzmooda in product(range(360), [True, False])
    }


def add_all_avg_cols(df: EconomicPrediction):
    for col in get_all_avg_cols():
        df[col] = 3 / 100
