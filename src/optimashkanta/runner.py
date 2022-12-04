from itertools import product

import pandas as pd
import plotly.express as px
import plotly.io as pio
from tqdm import tqdm

from optimashkanta.economic_prediction import load_all_economic_predictions
from optimashkanta.model import Kalatz
from optimashkanta.model import Katz
from optimashkanta.model import Matz
from optimashkanta.model import Prime
from optimashkanta.model import Tamhil


pio.renderers.default = "browser"

tamhils = {
    "mid_short": Tamhil(
        {
            "kalatz": Kalatz(
                value=192026,
                first_month=0,
                duration=360,
                yearly_rate=4.65 / 100,
            ),
            "matz": Matz(
                value=182706,
                first_month=0,
                duration=360,
                initial_yearly_rate=3.33 / 100,
                changes_every=60,
            ),
            "prime": Prime(
                value=739905,
                first_month=0,
                duration=360,
                initial_yearly_rate=4.35 / 100,
            ),
            "katz": Katz(
                value=283259,
                first_month=0,
                duration=292,
                yearly_rate=2.82 / 100,
            ),
        }
    ),
    "mid_high_short": Tamhil(
        {
            "kalatz": Kalatz(
                value=116493,
                first_month=0,
                duration=336,
                yearly_rate=4.6 / 100,
            ),
            "matz": Matz(
                value=281376,
                first_month=0,
                duration=360,
                initial_yearly_rate=3.33 / 100,
                changes_every=60,
            ),
            "prime": Prime(
                value=582456,
                first_month=0,
                duration=360,
                initial_yearly_rate=4.01 / 100,
            ),
            "katz": Katz(
                value=417571,
                first_month=0,
                duration=253,
                yearly_rate=2.78 / 100,
            ),
        }
    ),
    "high_short": Tamhil(
        {
            "matz": Matz(
                value=138123,
                first_month=0,
                duration=360,
                initial_yearly_rate=3.33 / 100,
                changes_every=60,
            ),
            "prime": Prime(
                value=565313,
                first_month=0,
                duration=360,
                initial_yearly_rate=3.98 / 100,
            ),
            "katz": Katz(
                value=694460,
                first_month=0,
                duration=264,
                yearly_rate=2.79 / 100,
            ),
        }
    ),
    "mid_long": Tamhil(
        {
            "kalatz": Kalatz(
                value=127528,
                first_month=0,
                duration=360,
                yearly_rate=4.65 / 100,
            ),
            "katz": Katz(
                value=465965,
                first_month=0,
                duration=325,
                yearly_rate=2.85 / 100,
            ),
            "prime": Prime(
                value=804403,
                first_month=0,
                duration=349,
                initial_yearly_rate=4.48 / 100,
            ),
        }
    ),
    "mig_high_long": Tamhil(
        {
            "kalatz": Kalatz(
                value=915,
                first_month=0,
                duration=348,
                yearly_rate=4.62 / 100,
            ),
            "katz": Katz(
                value=698948,
                first_month=0,
                duration=296,
                yearly_rate=2.83 / 100,
            ),
            "prime": Prime(
                value=698033,
                first_month=0,
                duration=338,
                initial_yearly_rate=4.26 / 100,
            ),
        }
    ),
    "high_long": Tamhil(
        {
            "katz1": Katz(
                value=326260,
                first_month=0,
                duration=288,
                yearly_rate=2.82 / 100,
            ),
            "katz2": Katz(
                value=605671,
                first_month=0,
                duration=276,
                yearly_rate=3.01 / 100,
            ),
            "prime": Prime(
                value=465965,
                first_month=0,
                duration=336,
                initial_yearly_rate=3.71 / 100,
            ),
        }
    ),
}

if __name__ == "__main__":
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
    df = pd.concat(dfs, keys=keys, names=["scen", "tamhil"])
    figs = [
        (
            "ratio.html",
            px.line(
                df.reset_index(),
                x="month",
                y="ratio",
                color="scen",
                facet_col="tamhil",
                facet_col_wrap=3,
            ),
        ),
        (
            "total_price.html",
            px.line(
                df.reset_index(),
                x="month",
                y="total_price",
                color="scen",
                facet_col="tamhil",
                facet_col_wrap=3,
            ),
        ),
        (
            "amlat_piraon.html",
            px.line(
                df.reset_index(),
                x="month",
                y="amlat_piraon_mookdam",
                color="scen",
                facet_col="tamhil",
                facet_col_wrap=3,
            ),
        ),
        (
            "pmt.html",
            px.line(
                df.reset_index(),
                x="month",
                y="pmt",
                color="scen",
                facet_col="tamhil",
                facet_col_wrap=3,
            ),
        ),
        (
            "def_ratio.html",
            px.line(
                df.reset_index(),
                x="month",
                y="deflated_ratio",
                color="scen",
                facet_col="tamhil",
                facet_col_wrap=3,
            ),
        ),
        (
            "def_total_price.html",
            px.line(
                df.reset_index(),
                x="month",
                y="deflated_total_price",
                color="scen",
                facet_col="tamhil",
                facet_col_wrap=3,
            ),
        ),
        (
            "deflated_amlat_piraon.html",
            px.line(
                df.reset_index(),
                x="month",
                y="deflated_amlat_piraon_mookdam",
                color="scen",
                facet_col="tamhil",
                facet_col_wrap=3,
            ),
        ),
        (
            "def_pmt.html",
            px.line(
                df.reset_index(),
                x="month",
                y="deflated_pmt",
                color="scen",
                facet_col="tamhil",
                facet_col_wrap=3,
            ),
        ),
    ]
    for name, fig in figs:
        fig.show()
        fig.write_html(name)

    figs = [
        (
            "ratio.html",
            px.line(
                df.reset_index(),
                x="month",
                y="ratio",
                color="tamhil",
                facet_col="scen",
                facet_col_wrap=3,
            ),
        ),
        (
            "total_price.html",
            px.line(
                df.reset_index(),
                x="month",
                y="total_price",
                color="tamhil",
                facet_col="scen",
                facet_col_wrap=3,
            ),
        ),
        (
            "amlat_piraon.html",
            px.line(
                df.reset_index(),
                x="month",
                y="amlat_piraon_mookdam",
                color="tamhil",
                facet_col="scen",
                facet_col_wrap=3,
            ),
        ),
        (
            "pmt.html",
            px.line(
                df.reset_index(),
                x="month",
                y="pmt",
                color="tamhil",
                facet_col="scen",
                facet_col_wrap=3,
            ),
        ),
        (
            "def_ratio.html",
            px.line(
                df.reset_index(),
                x="month",
                y="deflated_ratio",
                color="tamhil",
                facet_col="scen",
                facet_col_wrap=3,
            ),
        ),
        (
            "def_total_price.html",
            px.line(
                df.reset_index(),
                x="month",
                y="deflated_total_price",
                color="tamhil",
                facet_col="scen",
                facet_col_wrap=3,
            ),
        ),
        (
            "deflated_amlat_piraon.html",
            px.line(
                df.reset_index(),
                x="month",
                y="deflated_amlat_piraon_mookdam",
                color="tamhil",
                facet_col="scen",
                facet_col_wrap=3,
            ),
        ),
        (
            "def_pmt.html",
            px.line(
                df.reset_index(),
                x="month",
                y="deflated_pmt",
                color="tamhil",
                facet_col="scen",
                facet_col_wrap=3,
            ),
        ),
    ]
    for name, fig in figs:
        fig.show()
        fig.write_html("all_" + name)
    pass
