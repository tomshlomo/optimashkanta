import numpy as np
import pytest

from optimashkanta.model import Cols
from optimashkanta.model import Katz
from optimashkanta.tests.utils import all_close
from optimashkanta.tests.utils import load_main_prediction


@pytest.mark.parametrize(
    "first_month",
    [0, 1, 10],
)
def test_with_main_prediction(first_month: int) -> None:
    prime = Katz(
        value=283259,
        first_month=first_month,
        duration=292,
        yearly_rate=2.82 / 100,
    )
    economic_prediction = load_main_prediction()
    df = prime.simulate(economic_prediction=economic_prediction)
    assert all_close(
        df.loc[[0 + first_month, 7 * 12 - 1 + first_month], Cols.PMT],
        np.array([-1345, -1611]),
        1e1,
    )
    pass
