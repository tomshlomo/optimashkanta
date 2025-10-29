from copy import copy
from typing import Optional

from optimashkanta.model import Kalatz
from optimashkanta.model import Katz
from optimashkanta.model import Loan
from optimashkanta.model import Malatz
from optimashkanta.model import Matz
from optimashkanta.model import Prime
from optimashkanta.model import Tamhil


luz = {
    0: (17.5 - 5) / 75,
    5: (17.5) / 75,
    11: (17.5) / 75,
    17: (17.5) / 75,
    23: (10) / 75,
}


def to_peimot(tamhil: Tamhil) -> Tamhil:
    def key_(loan: Loan) -> (int, int):
        loan_type = type(loan)
        loan_type_to_priority = {
            Kalatz: 1,
            Katz: 2,
            Prime: 3,
            Malatz: 4,
            Matz: 5,
        }
        return (loan_type_to_priority[loan_type], loan.duration)

    sorted_loans = sorted(tamhil.loans.items(), key=lambda x: key_(x[1]))
    loans = {}
    for first_month, fraction in luz.items():
        total_value = tamhil.value() * fraction
        while abs(total_value) > 10:
            loan_to_split_name, loan_to_split = sorted_loans.pop(0)
            new_loan, remaining_loan = split_loan(loan_to_split, total_value)
            new_loan_name = f"{loan_to_split_name}_{first_month}"
            new_loan.first_month = first_month
            loans[new_loan_name] = new_loan
            total_value -= new_loan.value
            if remaining_loan.value > 10:
                sorted_loans.insert(0, (loan_to_split_name, remaining_loan))
    new_tamhil = Tamhil(loans=loans)
    assert abs(new_tamhil.value() - tamhil.value()) <= 100
    return new_tamhil


def split_loan(loan: Loan, new_value: float) -> (Loan, Loan):
    new_loan = copy(loan)
    new_loan.value = min(new_value, loan.value)
    remaining_value = loan.value - new_loan.value
    remaining_loan = copy(loan)
    remaining_loan.value = remaining_value
    return new_loan, remaining_loan
