import pytest

from simengine.queueing import erlang_c, mm1, mmc


def test_mm1_textbook_values():
    q = mm1(0.8, 1.0)
    assert q.rho == pytest.approx(0.8)
    assert q.L == pytest.approx(4.0)
    assert q.W == pytest.approx(5.0)
    assert q.Wq == pytest.approx(4.0)
    assert q.L == pytest.approx(0.8 * q.W)  # Little's Law


def test_erlang_c_known_value():
    # Two servers, offered load 1: P(wait) = 1/3.
    assert erlang_c(2, 1.0) == pytest.approx(1 / 3)


def test_mmc_with_one_server_is_mm1():
    a, b = mmc(0.7, 1.0, 1), mm1(0.7, 1.0)
    for field in ("rho", "p_wait", "Lq", "L", "Wq", "W"):
        assert getattr(a, field) == pytest.approx(getattr(b, field))


def test_mmc_stays_finite_for_many_servers():
    q = mmc(190.0, 1.0, 200)
    assert 0 < q.p_wait < 1


@pytest.mark.parametrize("call", [lambda: mm1(1.0, 1.0), lambda: mmc(3.0, 1.0, 3)])
def test_unstable_systems_are_rejected(call):
    with pytest.raises(ValueError):
        call()
