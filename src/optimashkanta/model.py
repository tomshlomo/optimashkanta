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
    AMLAT_PIRAON_MOODKAM = "amlat_piraon_mookdam"
    PIRAON_MOOKDAM_PRICE = "piraon_mookdam_price"
    DEFLATION_COEF = "deflation_coef"
    DEFLATED_PMT = "deflated_pmt"
    DEFLATED_PIRAON_MOOKDAM_PRICE = "piraon_mookdam_price"
    DEFLATED_AMLAT_PIRAON_MOOKDAM = "deflated_amlat_piraon_mookdam"
    CUM_PMT = "cum_pmt"
    TOTAL_PRICE = "total_price"
    DEFLATED_TOTAL_PRICE = "deflated_total_price"
    RATIO = "ratio"
    DEFLATED_RATIO = "deflated_ratio"


def get_avg_col_from_duration(remaining_months: int, is_tzmooda: bool) -> str:
    remaining_years = remaining_months / 12
    if remaining_years >= 25:
        x = "25_inf"
    elif remaining_years >= 20:
        x = "20_25"
    elif remaining_years >= 15:
        x = "15_20"
    elif remaining_years >= 10:
        x = "10_15"
    elif remaining_years >= 5:
        x = "5_10"
    elif remaining_years >= 1 and not is_tzmooda:
        x = "1_5"
    elif not is_tzmooda:
        x = "0_1"
    else:
        x = "0_5"
    x = "tzamood_" + x
    if not is_tzmooda:
        x = "lo_" + x
    return x


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
        cols = [Cols.PMT, Cols.IPMT, Cols.PPMT, Cols.VAL, Cols.AMLAT_PIRAON_MOODKAM]
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
            piraon_mookdam = self.calc_amlat_piraon_mookdam(
                month, pmt, rate, economic_prediction
            )
            df.loc[month, cols] = (pmt, ipmt, ppmt, value, piraon_mookdam)
            value = value + ppmt
        df[Cols.PIRAON_MOOKDAM_PRICE] = df[Cols.AMLAT_PIRAON_MOODKAM] + df[Cols.VAL]
        df[Cols.CUM_PMT] = np.cumsum(df[Cols.PMT])
        df[Cols.TOTAL_PRICE] = -df[Cols.CUM_PMT] + df[Cols.PIRAON_MOOKDAM_PRICE]
        df[Cols.RATIO] = df[Cols.TOTAL_PRICE] / self.value

        df[Cols.DEFLATION_COEF] = np.insert(
            np.cumprod(
                1
                / (
                    1
                    + economic_prediction.loc[df.index[:-1], Cols.INFLATION].values / 12
                )
            ),
            0,
            1.0,
            axis=0,
        )
        for col in [
            Cols.PIRAON_MOOKDAM_PRICE,
            Cols.AMLAT_PIRAON_MOODKAM,
            Cols.PMT,
            Cols.RATIO,
            Cols.TOTAL_PRICE,
        ]:
            df[f"deflated_{col}"] = df[col] * df[Cols.DEFLATION_COEF]
        return df

    def last_month(self) -> int:
        return self.first_month + self.duration - 1

    def _get_avg_monthly_rate_at_piraon(
        self, month: int, economic_prediction: EconomicPrediction
    ) -> float:
        months_remaining = self.last_month() - month + 1
        col = get_avg_col_from_duration(months_remaining, self.is_tzmooda())
        month = min(month, economic_prediction.index[-1])
        return economic_prediction.at[month, col] / 12

    def calc_amlat_piraon_mookdam(
        self,
        month: int,
        pmt: float,
        monthly_rate: float,
        economic_prediction: EconomicPrediction,
    ) -> float:
        if isinstance(self, Prime):
            return 0.0
        avg_monthly_rate_at_piraon = self._get_avg_monthly_rate_at_piraon(
            month, economic_prediction
        )
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
        month = min(month, economic_prediction.index[-1])
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


@dataclass
class Kalatz(Kvooa, LoTzmooda):
    def monthly_rate(self) -> float:
        return self.yearly_rate / 12


@dataclass
class Katz(Kvooa, Tzmooda):
    pass


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
        months = np.clip(months, 0, economic_prediction.index[-1])
        return economic_prediction.loc[  # type:ignore[no-any-return]
            months, Cols.RBI
        ].values + self.spread(economic_prediction=economic_prediction)


@dataclass
class Tamhil:
    loans: dict[str, Loan]

    def value(self) -> float:
        return sum(loan.value for loan in self.loans.values())

    def simulate(self, economic_prediction: EconomicPrediction) -> pd.DataFrame:
        dfs = [
            loan.simulate(economic_prediction=economic_prediction)
            for loan in self.loans.values()
        ]
        df = pd.concat(dfs, axis=1, keys=self.loans.keys(), names=["loan_name"])
        cols_to_sum = [
            Cols.TOTAL_PRICE,
            Cols.PMT,
            Cols.PIRAON_MOOKDAM_PRICE,
            Cols.AMLAT_PIRAON_MOODKAM,
            Cols.DEFLATED_TOTAL_PRICE,
            Cols.DEFLATED_PIRAON_MOOKDAM_PRICE,
            Cols.DEFLATED_PMT,
            Cols.DEFLATED_AMLAT_PIRAON_MOOKDAM,
        ]
        for col in cols_to_sum:
            cols = [(loan_name, col) for loan_name in self.loans.keys()]
            if col in [Cols.TOTAL_PRICE, Cols.DEFLATED_TOTAL_PRICE]:
                df[col] = df.loc[:, cols].fillna(method="ffill").sum(axis=1)
            else:
                df[col] = df.loc[:, cols].fillna(0).sum(axis=1)
        df[Cols.RATIO] = df[Cols.TOTAL_PRICE] / self.value()
        df[Cols.DEFLATED_RATIO] = df[Cols.DEFLATED_TOTAL_PRICE] / self.value()
        return df
