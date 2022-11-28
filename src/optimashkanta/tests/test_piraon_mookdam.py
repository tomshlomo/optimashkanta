from optimashkanta.model import piraon_mookdam_before_discount


def test_kvooa_piraon_mookdam_before_discount():
    result = piraon_mookdam_before_discount(
        monthly_rate_at_piraon=4.65 / 100 / 12,
        avg_monthly_rate_at_piraon=3 / 100 / 12,
        pmt_at_piraon=-990,
        months_from_piraon_to_change=360 - 84,
        months_from_change_to_end=0,
    )
    assert abs(result - 20708 / 0.7) < 1e1


def test_mishtana_piraon_mookdam_before_discount():
    result = piraon_mookdam_before_discount(
        monthly_rate_at_piraon=4.65 / 100 / 12,
        avg_monthly_rate_at_piraon=3 / 100 / 12,
        pmt_at_piraon=-990,
        months_from_piraon_to_change=36,
        months_from_change_to_end=240,
    )
    assert abs(result - 5345 / 0.7) < 1e1
