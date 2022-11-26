from typing import TypeAlias

import pandas as pd

from optimashkanta.model import Cols
from optimashkanta.model import EconomicPrediction


ArrayLike: TypeAlias = pd.Series | pd.DataFrame


def all_close(
    val: ArrayLike, expected: ArrayLike, tol: ArrayLike | float | int
) -> bool:
    return bool(((val - expected).abs() <= tol).all().all())


def load_main_prediction() -> EconomicPrediction:
    df = pd.read_csv("src/optimashkanta/tests/main_prediction.csv", header=None)
    df.columns = [Cols.RBI, Cols.INFLATION, Cols.OGEN_TZMOODA, Cols.OGEN_LO_TZMOODA]
    df /= 100
    return df  # type:ignore[no-any-return]
