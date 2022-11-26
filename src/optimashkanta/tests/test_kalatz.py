import numpy as np
import pandas as pd

from optimashkanta.model import Cols
from optimashkanta.model import Kalatz
from optimashkanta.tests.utils import all_close


def test_simulate():
    loan = Kalatz(
        value=192026,
        first_month=0,
        duration=360,
        yearly_rate=4.65 / 100,
    )
    economic_prediction = pd.DataFrame()
    df = loan.simulate(economic_prediction=economic_prediction)
    assert all_close(df[Cols.MONTHLY_RATE], 4.65 / 12 / 100, 1e-6)
    assert all_close(df[Cols.PMT], -990.16, 1e-1)
    assert all_close(
        df.loc[[0, 1, 359], Cols.VAL],
        np.array([192026, 192026 * (1 + 4.65 / 12 / 100) - 990.16, 990.16]),
        np.array([1e-6, 1e-1, 1e1]),
    )
