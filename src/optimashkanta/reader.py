import glob
from pathlib import Path
from typing import Optional

import pandas as pd

from optimashkanta.economic_prediction import load_economic_prediction
from optimashkanta.model import Cols
from optimashkanta.model import Kalatz
from optimashkanta.model import Katz
from optimashkanta.model import Loan
from optimashkanta.model import Malatz
from optimashkanta.model import Matz
from optimashkanta.model import Prime
from optimashkanta.model import Tamhil
from optimashkanta.peimot import to_peimot


EYAL_DATA_FOLDER = "/Users/tom.shlomo/workspace/personal/optimashkanta/src/optimashkanta/data/eyal_data"


def get_all_paths() -> list[Path]:
    return [Path(file) for file in glob.glob(f"{EYAL_DATA_FOLDER}/*.xlsx")]


class UnknownLoanType(Exception):
    pass


def get_all_tamhils() -> dict[str, Tamhil]:
    files = get_all_paths()
    tamhils = {}
    for file in files:
        tamhils_tmp = file_to_tamhils(file)
        for tamhil_index, tamhil in tamhils_tmp.items():
            tamhil = to_peimot(tamhil)
            tamhils[f"{file.name}_{tamhil_index}"] = tamhil
    return tamhils


def file_to_tamhils(file: Path) -> dict[int, Tamhil]:
    sheet_index = 1
    tamhils = {}
    while True:
        tamhil = sheet_to_tamhil(file, sheet_index)
        if tamhil is None:
            break
        tamhils[sheet_index] = tamhil
        sheet_index += 1
    return tamhils


def sheet_to_tamhil(file: Path, sheet_index: int) -> Optional[Tamhil]:
    df = pd.read_excel(
        file,
        sheet_name=sheet_index,
        usecols="A:P",
        nrows=10,
    )
    if "מספר הלוואה" not in df.columns:
        return None
    df = df.loc[0 : df["מספר הלוואה"].isna().idxmax() - 1, :]
    loans = {}
    for i, row in df.iterrows():
        loan = to_loan(row)
        loan_name = f"{type(loan).__name__}_{i + 1}"
        loans[loan_name] = loan
    return Tamhil(loans=loans)


def to_loan(row: pd.Series) -> Loan:
    value = row['סכום ההלוואה [ש"ח]']
    duration = row["זמן [חודשים]"]
    yearly_rate = row["ריבית יעד [%]"] / 100
    type_ = row["סוג ההלוואה"]
    first_month = 0
    economic_prediction = load_economic_prediction(1.0)
    if type_ == "קבועה לא צמודה":
        return Kalatz(
            value=value,
            duration=duration,
            yearly_rate=yearly_rate,
            first_month=first_month,
        )
    elif type_ == "משתנה צמודה כל חמש":
        return Matz(
            value=value,
            duration=duration,
            initial_yearly_rate=yearly_rate,
            changes_every=5 * 12,
            first_month=first_month,
        )
    elif type_ == "פריים":
        spread = yearly_rate - economic_prediction.at[0, Cols.RBI]
        return Prime(
            value=value,
            duration=duration,
            # initial_yearly_rate=yearly_rate,
            spread=spread,
            first_month=first_month,
        )
    elif type_ == "קבועה צמודה":
        return Katz(
            value=value,
            duration=duration,
            yearly_rate=yearly_rate,
            first_month=first_month,
        )
    elif type_ == "משתנה לא צמודה כל חמש":
        return Malatz(
            value=value,
            duration=duration,
            initial_yearly_rate=yearly_rate,
            changes_every=5 * 12,
            first_month=first_month,
        )
    raise UnknownLoanType(f"unknown loan type: {type_}")


if __name__ == "__main__":
    # sheet_to_tamhil(
    #     Path('/Users/tom.shlomo/workspace/personal/optimashkanta/src/optimashkanta/data/eyal_data/שלמה_1397896.5_6835_1.8_13.11_מידע_מלא.xlsx'),
    #     sheet_index=1,
    # )
    get_all_tamhils()
