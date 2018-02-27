import Debye_spectrum_0


def test__coefficient_ap():
    ap = Debye_spectrum_0.EqnConstructor._coefficient_ap(1,1)
    assert abs(ap-0.0) < 0.00001

def test__coefficient_bp():
    bp = Debye_spectrum_0.EqnConstructor._coefficient_bp(0,1)
    assert abs(bp-1.0) < 0.00001

def test_coefficient_sigma():
    sigma = Debye_spectrum_0.EqnConstructor._coefficient_sigma(1)
    assert abs(sigma + 0.5) < 0.00001