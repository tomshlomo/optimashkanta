from itertools import product

import pandas as pd
import plotly.express as px
import plotly.io as pio
from tqdm import tqdm

from optimashkanta.economic_prediction import load_all_economic_predictions
from optimashkanta.peimot import to_peimot


pio.renderers.default = "browser"

from optimashkanta.model import Cols
from optimashkanta.model import Kalatz
from optimashkanta.model import Matz
from optimashkanta.model import Prime
from optimashkanta.model import Tamhil


tamhils = {
    ### leumi ###
    "leumi_1_kalatz": Tamhil(
        loans={
            "kalatz": Kalatz(
                value=1397874,
                first_month=0,
                duration=360,
                yearly_rate=5.71 / 100,
            ),
        }
    ),
    "leumi_2_shlish": Tamhil(
        loans={
            "kalatz": Kalatz(
                value=465958,
                first_month=0,
                duration=360,
                yearly_rate=5.56 / 100,
            ),
            "prime": Prime(
                value=465958,
                first_month=0,
                duration=360,
                spread=(0.25 + 1.5) / 100,
            ),
            "matz": Matz(
                value=465958,
                first_month=0,
                duration=360,
                initial_yearly_rate=3.78 / 100,
                changes_every=60,
            ),
        }
    ),
    "leumi_3_half": Tamhil(
        loans={
            "kalatz": Kalatz(
                value=698937,
                first_month=0,
                duration=360,
                yearly_rate=5.71 / 100,
            ),
            "prime": Prime(
                value=698937,
                first_month=0,
                duration=360,
                spread=(0.5 + 1.5) / 100,
            ),
        }
    ),
    "leumi_4_my": Tamhil(
        loans={
            "kalatz": Kalatz(
                value=475286,
                first_month=0,
                duration=360,
                yearly_rate=4.41 / 100,
            ),
            "matz1": Matz(
                value=394036,
                first_month=0,
                duration=360,
                initial_yearly_rate=2.83 / 100,
                changes_every=60,
            ),
            "prime": Prime(
                value=461283,
                first_month=0,
                duration=360,
                spread=(-0.65 + 1.5) / 100,
            ),
            "matz2": Matz(
                value=67269,
                first_month=0,
                duration=120,
                initial_yearly_rate=2.83 / 100,
                changes_every=60,
            ),
        }
    ),
    ### poalim ###
    "poalim_1_kalatz": Tamhil(
        loans={
            "kalatz": Kalatz(
                value=1397896,
                first_month=0,
                duration=360,
                yearly_rate=4.965 / 100,
            ),
        }
    ),
    "poalim_2_shlish": Tamhil(
        loans={
            "kalatz": Kalatz(
                value=465966,
                first_month=0,
                duration=360,
                yearly_rate=5.267 / 100,
            ),
            "prime": Prime(
                value=465965,
                first_month=0,
                duration=360,
                spread=(0.28 + 1.5) / 100,
            ),
            "matz": Matz(
                value=465965,
                first_month=0,
                duration=360,
                initial_yearly_rate=2.94 / 100,
                changes_every=60,
            ),
        }
    ),
    "poalim_3_half": Tamhil(
        loans={
            "kalatz": Kalatz(
                value=698948,
                first_month=0,
                duration=360,
                yearly_rate=5.646 / 100,
            ),
            "prime": Prime(
                value=698948,
                first_month=0,
                duration=360,
                spread=(-0.15 + 1.5) / 100,
            ),
        }
    ),
    "poalim_4_my": Tamhil(
        loans={
            "kalatz": Kalatz(
                value=475286,
                first_month=0,
                duration=360,
                yearly_rate=4.43 / 100,
            ),
            "matz1": Matz(
                value=394036,
                first_month=0,
                duration=360,
                initial_yearly_rate=3.12 / 100,
                changes_every=60,
            ),
            "prime": Prime(
                value=461305,
                first_month=0,
                duration=360,
                spread=(1.5 - 0.65) / 100,
            ),
            "matz2": Matz(
                value=67269,
                first_month=0,
                duration=120,
                initial_yearly_rate=3.12 / 100,
                changes_every=60,
            ),
        }
    ),
    ### discont ###
    "discont_1_kalatz": Tamhil(
        loans={
            "kalatz": Kalatz(
                value=1397896,
                first_month=0,
                duration=360,
                yearly_rate=4.67 / 100,
            ),
        }
    ),
    "discont_2_shlish": Tamhil(
        loans={
            "kalatz": Kalatz(
                value=466059,
                first_month=0,
                duration=360,
                yearly_rate=4.86 / 100,
            ),
            "prime": Prime(
                value=465919,
                first_month=0,
                duration=360,
                spread=(-0.44 + 1.5) / 100,
            ),
            "matz": Matz(
                value=465919,
                first_month=0,
                duration=360,
                initial_yearly_rate=2.46 / 100,
                changes_every=60,
            ),
        }
    ),
    "discont_3_half": Tamhil(
        loans={
            "kalatz": Kalatz(
                value=698948,
                first_month=0,
                duration=360,
                yearly_rate=4.53 / 100,
            ),
            "prime": Prime(
                value=698948,
                first_month=0,
                duration=360,
                spread=(-0.04 + 1.5) / 100,
            ),
        }
    ),
    "discont_4_my": Tamhil(
        loans={
            "kalatz": Kalatz(
                value=475286,
                first_month=0,
                duration=360,
                yearly_rate=4.4 / 100,
            ),
            "matz1": Matz(
                value=394036,
                first_month=0,
                duration=360,
                initial_yearly_rate=3.1 / 100,
                changes_every=60,
            ),
            "prime": Prime(
                value=461305,
                first_month=0,
                duration=360,
                spread=(-0.65 + 1.5) / 100,
            ),
            "matz2": Matz(
                value=67269,
                first_month=0,
                duration=120,
                initial_yearly_rate=3.1 / 100,
                changes_every=60,
            ),
        }
    ),
    "plan": Tamhil(
        loans={
            "kalatz": Kalatz(
                value=475286,
                first_month=0,
                duration=360,
                yearly_rate=4.59 / 100,
            ),
            "matz1": Matz(
                value=394036,
                first_month=0,
                duration=360,
                initial_yearly_rate=3.29 / 100,
                changes_every=60,
            ),
            "prime": Prime(
                value=461283,
                first_month=0,
                duration=360,
                spread=(-0.5 + 1.5) / 100,
            ),
            "matz2": Matz(
                value=67269,
                first_month=0,
                duration=120,
                initial_yearly_rate=4.29 / 100,
                changes_every=60,
            ),
        }
    ),
}

tamhils = {name: to_peimot(tamhil) for name, tamhil in tamhils.items()}


def get_data() -> pd.DataFrame:
    economic_predictions = load_all_economic_predictions()
    dfs = []
    keys = []
    for (scen_name, economic_prediction), (tamhil_name, tamhil) in tqdm(
        product(
            economic_predictions.items(),
            tamhils.items(),
        ),
        total=len(economic_predictions) * len(tamhils),
    ):
        dfs.append(tamhil.simulate(economic_prediction))
        keys.append((scen_name, tamhil_name))
    return pd.concat(dfs, keys=keys, names=["scen", "tamhil"])


def plot(df: pd.DataFrame):
    cols_to_plot = [
        Cols.RATIO,
        Cols.DEFLATED_RATIO,
        Cols.PMT,
        Cols.DEFLATED_PMT,
        Cols.TOTAL_PRICE,
        Cols.DEFLATED_TOTAL_PRICE,
        Cols.PIRAON_MOOKDAM_PRICE,
        Cols.DEFLATED_PIRAON_MOOKDAM_PRICE,
        Cols.AMLAT_PIRAON_MOODKAM,
        Cols.DEFLATED_AMLAT_PIRAON_MOOKDAM,
    ]
    df.xs(key=84, level="month").loc[:, cols_to_plot].to_excel("results.xlsx")
    df = df.reset_index()
    df["bank"] = df["tamhil"].apply(lambda s: s.split("_")[0])
    for col in cols_to_plot:
        fig = px.line(
            df,
            x="month",
            y=col,
            color="tamhil",
            line_dash="bank",
            facet_col="scen",
            facet_col_wrap=3,
        )
        fig.show()
        fig.write_html(col + ".html")
    pass


def main():
    df = get_data()
    plot(df)
    pass


if __name__ == "__main__":
    main()
