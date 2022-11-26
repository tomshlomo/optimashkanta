import numpy as np
import pandas as pd

from optimashkanta.model import Cols
from optimashkanta.model import Kalatz
from optimashkanta.model import Prime
from optimashkanta.tests.utils import all_close
from optimashkanta.tests.utils import load_main_prediction


def test_constant_rbi():
    kalatz = Kalatz(
        value=192026,
        first_month=0,
        duration=360,
        yearly_rate=4.65 / 100,
    )
    prime = Prime(
        value=192026,
        first_month=0,
        duration=360,
        initial_yearly_rate=4.65 / 100,
    )
    economic_prediction = pd.DataFrame(
        index=pd.RangeIndex(0, 360),
    )
    economic_prediction[Cols.RBI] = 10.0

    df_kalatz = kalatz.simulate(economic_prediction=economic_prediction)
    df_prime = prime.simulate(economic_prediction=economic_prediction)
    assert all_close(df_kalatz, df_prime, 1e-6)


def test_with_main_prediction():
    prime = Prime(
        value=739905,
        first_month=0,
        duration=360,
        initial_yearly_rate=4.35 / 100,
    )
    economic_prediction = load_main_prediction()
    df = prime.simulate(economic_prediction=economic_prediction)
    assert all_close(
        df.loc[[0, 7 * 12], Cols.PMT],
        np.array([-3685, -3973]),
        1e1,
    )
