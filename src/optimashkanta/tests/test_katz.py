import numpy as np

from optimashkanta.model import Cols
from optimashkanta.model import Katz
from optimashkanta.tests.utils import all_close
from optimashkanta.tests.utils import load_main_prediction


def test_with_main_prediction():
    prime = Katz(
        value=283259,
        first_month=0,
        duration=292,
        initial_yearly_rate=2.82 / 100,
    )
    economic_prediction = load_main_prediction()
    df = prime.simulate(economic_prediction=economic_prediction)
    assert all_close(
        df.loc[[0, 7 * 12 - 1], Cols.PMT],
        np.array([-1345, -1611]),
        1e1,
    )
    pass
