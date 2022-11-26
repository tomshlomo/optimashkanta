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


EconomicPrediction = NewType(
    "EconomicPrediction", pd.DataFrame
)  # type:ignore[valid-newtype]


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
        cols = [Cols.PMT, Cols.IPMT, Cols.PPMT, Cols.VAL]
        value = self.value
        for month, row in df.iterrows():
            value = self.inflate(value, month, economic_prediction)
            rate = row[Cols.MONTHLY_RATE]
            age = row[Cols.AGE]
            pmt = npf.pmt(rate=rate, nper=self.duration - age, pv=value)
            ipmt = -rate * value
            ppmt = pmt - ipmt
            df.loc[month, cols] = (pmt, ipmt, ppmt, value)
            value = value + ppmt
        return df

    @abstractmethod
    def inflate(
        self, val: float, month: int, economic_prediction: EconomicPrediction
    ) -> float:
        pass

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

    def is_tzmooda(self) -> bool:
        if isinstance(self, Tzmooda):
            return True
        if isinstance(self, LoTzmooda):
            return False
        raise TypeError


class Tzmooda(Loan, ABC):
    def inflate(
        self, val: float, month: int, economic_prediction: EconomicPrediction
    ) -> float:
        return val * (1 + float(economic_prediction.at[month, Cols.INFLATION]) / 12)


class LoTzmooda(Loan, ABC):
    def inflate(
        self, val: float, month: int, economic_prediction: EconomicPrediction
    ) -> float:
        return val


@dataclass
class Kvooa(Loan, ABC):
    yearly_rate: float

    def predict_yearly_rate(
        self, months: pd.RangeIndex, economic_prediction: EconomicPrediction
    ) -> npt.NDArray[np.float64]:
        return np.zeros(months.shape[0]) + self.yearly_rate


@dataclass
class Kalatz(Kvooa, LoTzmooda):
    yearly_rate: float


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
class MishtanaTzmooda(Mishtana, Tzmooda):
    pass


@dataclass
class MishtanaLoTzmooda(Mishtana, Tzmooda):
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
        return economic_prediction.loc[
            months, Cols.RBI
        ].values + self.spread(  # type:ignore[no-any-return]
            economic_prediction=economic_prediction
        )


@dataclass
class Katz(Tzmooda):
    initial_yearly_rate: float

    def predict_yearly_rate(
        self, months: pd.RangeIndex, economic_prediction: EconomicPrediction
    ) -> npt.NDArray[np.float64]:
        return np.zeros(months.shape[0]) + self.initial_yearly_rate


@dataclass
class Tamhil:
    loans: dict[str, Loan]

    def simulate(self, economic_prediction: EconomicPrediction) -> pd.DataFrame:
        dfs = [
            loan.simulate(economic_prediction=economic_prediction)
            for loan in self.loans.values()
        ]
        return pd.concat(dfs, axis=1, keys=self.loans.keys(), names=["loan_name"])
