"""Task 3's verdict: the band decides consistency, the mean decides the direction.

Run with:  colcon test --packages-select l3_ekf_demo
"""

from l3_ekf_demo.ekf import nis_verdict

DOF, LO, HI = 2, 0.051, 7.378        # config/ekf.yaml: 95% band for m = 2


def test_consistent():
    # 19 of 20 inside the band (95%), mean near 2
    values = [2.0] * 19 + [9.0]
    verdict, mean, inside = nis_verdict(values, DOF, LO, HI)
    assert verdict == "consistent"
    assert inside == 95.0
    assert abs(mean - 2.35) < 1e-9


def test_overconfident_with_mean_below_the_band_top():
    # Mean 4.2: inside the band's top (7.378) but above its expected value (2),
    # and only 60% inside. The old check (mean > 7.378) called this underconfident.
    values = [3.0] * 12 + [8.0] * 6 + [0.01] * 2
    verdict, mean, inside = nis_verdict(values, DOF, LO, HI)
    assert inside == 60.0
    assert DOF < mean < HI
    assert verdict == "overconfident"


def test_underconfident():
    # R far too large: most values below the band, mean far below 2
    values = [0.01] * 15 + [0.5] * 5
    verdict, mean, inside = nis_verdict(values, DOF, LO, HI)
    assert inside == 25.0
    assert mean < DOF
    assert verdict == "underconfident"
