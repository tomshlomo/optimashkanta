from itertools import product

import pandas as pd
import plotly.express as px
import plotly.io as pio
from tqdm import tqdm

from optimashkanta.economic_prediction import load_all_economic_predictions
from optimashkanta.model import Cols
from optimashkanta.model import Kalatz
from optimashkanta.model import Katz
from optimashkanta.model import Matz
from optimashkanta.model import Prime
from optimashkanta.model import Tamhil
from optimashkanta.reader import get_all_tamhils


pio.renderers.default = "browser"


def get_data() -> pd.DataFrame:
    economic_predictions = load_all_economic_predictions()
    tamhils = get_all_tamhils()
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
    for col in cols_to_plot:
        fig = px.line(
            df,
            x="month",
            y=col,
            color="tamhil",
            facet_col="scen",
            facet_col_wrap=3,
        )
        fig.show()
        fig.write_html(col + ".html")


def main():
    df = get_data()
    plot(df)
    pass


if __name__ == "__main__":
    main()
