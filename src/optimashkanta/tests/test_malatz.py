from optimashkanta.model import Cols
from optimashkanta.model import Malatz
from optimashkanta.tests.utils import load_main_prediction


def test_piraon_mookdam() -> None:
    prime = Malatz(
        value=192026,
        first_month=0,
        duration=360,
        initial_yearly_rate=4.65 / 100,
        changes_every=120,
    )
    economic_prediction = load_main_prediction()
    df = prime.simulate(economic_prediction=economic_prediction)
    assert abs(df.at[84, Cols.PIRAON_MOODKAM] - 5345) < 1
