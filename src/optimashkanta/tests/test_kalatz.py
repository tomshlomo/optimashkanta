import numpy as np
import pandas as pd
import pytest

from optimashkanta.model import Cols
from optimashkanta.model import Kalatz
from optimashkanta.tests.utils import all_close


@pytest.mark.parametrize(
    "first_month",
    [0, 1, 100],
)
def test_simulate(first_month: int) -> None:
    loan = Kalatz(
        value=192026,
        first_month=first_month,
        duration=360,
        yearly_rate=4.65 / 100,
    )
    economic_prediction = pd.DataFrame()
    df = loan.simulate(economic_prediction=economic_prediction)
    assert all_close(df[Cols.MONTHLY_RATE], 4.65 / 12 / 100, 1e-6)
    assert all_close(df[Cols.PMT], -990.16, 1e-1)
    assert all_close(
        df.loc[[0 + first_month, 1 + first_month, 359 + first_month], Cols.VAL],
        np.array([192026, 192026 * (1 + 4.65 / 12 / 100) - 990.16, 990.16]),
        np.array([1e-6, 1e-1, 1e1]),
    )


@pytest.mark.parametrize(
    "first_month",
    [0, 1, 100],
)
def test_piraon_mookdam(first_month: int) -> None:
    prime = Kalatz(
        value=192026,
        first_month=first_month,
        duration=360,
        yearly_rate=4.65 / 100,
    )
    economic_prediction = pd.DataFrame()
    df = prime.simulate(economic_prediction=economic_prediction)
    assert abs(df.at[84 + first_month, Cols.AMLAT_PIRAON_MOODKAM] - 20708) < 1
