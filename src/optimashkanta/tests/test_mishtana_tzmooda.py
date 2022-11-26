import numpy as np

from optimashkanta.model import Cols
from optimashkanta.model import MishtanaTzmooda
from optimashkanta.tests.utils import all_close
from optimashkanta.tests.utils import load_main_prediction


def test_with_main_prediction():
    prime = MishtanaTzmooda(
        value=182706,
        first_month=0,
        duration=360,
        initial_yearly_rate=3.33 / 100,
        changes_every=12 * 5,
    )
    economic_prediction = load_main_prediction()
    df = prime.simulate(economic_prediction=economic_prediction)
    assert all_close(
        df.loc[[0, 7 * 12 - 1], Cols.PMT],
        np.array([-805, -991]),
        1e1,
    )
    pass
