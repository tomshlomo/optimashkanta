from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import NewType

import numpy as np
import numpy.typing as npt
import numpy_financial as npf
import pandas as pd


class Cols:
    MONTH = "month"
    PMT = "pmt"
    IPMT = "ipmt"
    PPMT = "ppmt"
    VAL = "val"
    MONTHLY_RATE = "monthly_rate"
    AGE = "age"
    RBI = "rbi"
    INFLATION = "inflation"
    OGEN_TZMOODA = "ogen_tzmooda"
    OGEN_LO_TZMOODA = "ogen_lo_tzmooda"
    PIRAON_MOODKAM = "piraon_mookdam"


EconomicPrediction = NewType(  # type:ignore[valid-newtype]
    "EconomicPrediction", pd.DataFrame
)


@dataclass
class Loan:
    value: float
    first_month: int
    duration: int

    def simulate(self, economic_prediction: EconomicPrediction) -> pd.DataFrame:
        months = pd.RangeIndex(
            start=self.first_month,
            stop=self.first_month + self.duration,
            name=Cols.MONTH,
        )
        df = pd.DataFrame(
            index=months,
        )
        df[Cols.MONTHLY_RATE] = self.predict_monthly_rate(months, economic_prediction)
        df[Cols.AGE] = np.arange(self.duration)
        cols = [Cols.PMT, Cols.IPMT, Cols.PPMT, Cols.VAL, Cols.PIRAON_MOODKAM]
        is_tzmooda = self.is_tzmooda()
        value = self.value
        for month, row in df.iterrows():
            if is_tzmooda:
                value = self.inflate(value, month, economic_prediction)
            rate = row[Cols.MONTHLY_RATE]
            age = row[Cols.AGE]
            pmt = npf.pmt(rate=rate, nper=self.duration - age, pv=value)
            ipmt = -rate * value
            ppmt = pmt - ipmt
            piraon_mookdam = self.calc_piraon_mookdam(month, pmt, rate)
            df.loc[month, cols] = (pmt, ipmt, ppmt, value, piraon_mookdam)
            value = value + ppmt
        return df

    def calc_piraon_mookdam(
        self,
        month: int,
        pmt: float,
        monthly_rate: float,
    ) -> float:
        if isinstance(self, Prime):
            return 0.0
        avg_monthly_rate_at_piraon = 3 / 100 / 12  # todo: don't hard code
        months_from_piraon_to_change = self.duration - month + self.first_month
        if isinstance(self, Mishtana):
            months_from_piraon_to_change %= self.changes_every
        months_from_change_to_end = (
            self.duration - month + self.first_month - months_from_piraon_to_change
        )
        return (
            piraon_mookdam_before_discount(
                monthly_rate,
                avg_monthly_rate_at_piraon,
                pmt,
                months_from_piraon_to_change,
                months_from_change_to_end,
            )
            * 0.7
        )  # todo: don't hardcode 0.7

    def inflate(
        self, val: float, month: int, economic_prediction: EconomicPrediction
    ) -> float:
        return val * (1 + float(economic_prediction.at[month, Cols.INFLATION]) / 12)

    def predict_monthly_rate(
        self, months: pd.RangeIndex, economic_prediction: EconomicPrediction
    ) -> npt.NDArray[np.float64]:
        return (
            self.predict_yearly_rate(
                months=months, economic_prediction=economic_prediction
            )
            / 12
        )

    @abstractmethod
    def predict_yearly_rate(
        self, months: pd.RangeIndex, economic_prediction: EconomicPrediction
    ) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def is_tzmooda(self) -> bool:
        pass


class Tzmooda(Loan, ABC):
    def is_tzmooda(self) -> bool:
        return True


class LoTzmooda(Loan, ABC):
    def is_tzmooda(self) -> bool:
        return False


def piraon_mookdam_before_discount(
    monthly_rate_at_piraon: float,
    avg_monthly_rate_at_piraon: float,
    pmt_at_piraon: float,
    months_from_piraon_to_change: int,
    months_from_change_to_end: int,
) -> float:
    # todo:vectorize
    total_value_until_change_with_avg_rate = npf.pv(
        avg_monthly_rate_at_piraon, months_from_piraon_to_change, -pmt_at_piraon
    )
    total_value_until_change_with_my_rate = npf.pv(
        monthly_rate_at_piraon, months_from_piraon_to_change, -pmt_at_piraon
    )
    total_value_after_change = npf.pv(
        monthly_rate_at_piraon, months_from_change_to_end, -pmt_at_piraon
    )
    hivoon_coef_avg = (1 + avg_monthly_rate_at_piraon) ** (
        -months_from_piraon_to_change
    )
    hivoon_coef_my = (1 + monthly_rate_at_piraon) ** (-months_from_piraon_to_change)
    return float(
        np.clip(
            total_value_until_change_with_my_rate
            - total_value_until_change_with_avg_rate
            + total_value_after_change * (hivoon_coef_my - hivoon_coef_avg),
            0,
            None,
        )
    )


@dataclass
class Kvooa(Loan, ABC):
    yearly_rate: float

    def predict_yearly_rate(
        self, months: pd.RangeIndex, economic_prediction: EconomicPrediction
    ) -> npt.NDArray[np.float64]:
        return np.zeros(months.shape[0]) + self.yearly_rate

    def predict_early_piraon(
        self,
        month: int,
        economic_prediction,
    ) -> float:
        return (
            npf.pv(3 / 100 / 12, 360 - 84, 990) - npf.pv(4.65 / 100 / 12, 360 - 84, 990)
        ) * 0.7


@dataclass
class Kalatz(Kvooa, LoTzmooda):
    def monthly_rate(self) -> float:
        return self.yearly_rate / 12


@dataclass
class Katz(Kvooa, Tzmooda):
    pass

    def predict_early_piraon(
        self,
        month: int,
        economic_prediction,
    ) -> float:
        pass

    #
    # def predict_yearly_rate(
    #         self, months: pd.RangeIndex, economic_prediction: EconomicPrediction
    # ) -> npt.NDArray[np.float64]:
    #     return np.zeros(months.shape[0]) + self.initial_yearly_rate


@dataclass
class Mishtana(Loan, ABC):
    initial_yearly_rate: float
    changes_every: int

    def spread(self, economic_prediction: EconomicPrediction) -> float:
        return self.initial_yearly_rate - float(
            economic_prediction.at[self.first_month, self.ogen_col()]
        )

    def ogen_col(self) -> str:
        return Cols.OGEN_TZMOODA if self.is_tzmooda() else Cols.OGEN_LO_TZMOODA

    def predict_yearly_rate(
        self, months: pd.RangeIndex, economic_prediction: EconomicPrediction
    ) -> npt.NDArray[np.float64]:
        spread = self.spread(economic_prediction)
        d = months - self.first_month
        ogen_month = (d // self.changes_every) * self.changes_every
        ogen = economic_prediction.loc[ogen_month, self.ogen_col()]
        return ogen.values + spread  # type:ignore[no-any-return]


@dataclass
class Matz(Mishtana, Tzmooda):
    pass


@dataclass
class Malatz(Mishtana, LoTzmooda):
    pass


@dataclass
class Prime(LoTzmooda):
    initial_yearly_rate: float

    def spread(self, economic_prediction: EconomicPrediction) -> float:
        return self.initial_yearly_rate - float(
            economic_prediction.at[self.first_month, Cols.RBI]
        )

    def predict_yearly_rate(
        self, months: pd.RangeIndex, economic_prediction: EconomicPrediction
    ) -> npt.NDArray[np.float64]:
        return economic_prediction.loc[  # type:ignore[no-any-return]
            months, Cols.RBI
        ].values + self.spread(economic_prediction=economic_prediction)


@dataclass
class Tamhil:
    loans: dict[str, Loan]

    def simulate(self, economic_prediction: EconomicPrediction) -> pd.DataFrame:
        dfs = [
            loan.simulate(economic_prediction=economic_prediction)
            for loan in self.loans.values()
        ]
        return pd.concat(dfs, axis=1, keys=self.loans.keys(), names=["loan_name"])
