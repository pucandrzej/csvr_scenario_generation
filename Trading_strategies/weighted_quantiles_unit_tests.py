import numpy as np
import pytest

from Trading_strategies.strategies_utils import (
    weighted_median,
    vanilla_band,
    weighted_band,
    batch_weighted_quantiles,
)


def test_weighted_median():
    v = np.array([0.0, 10.0])

    assert weighted_median(v, np.array([0.5, 0.5])) == pytest.approx(0.0)  # exact
    assert weighted_median(v, np.array([0.25, 0.75])) == pytest.approx(
        10 / 3
    )  # interpolated


def test_weighted_quantiles():
    v, w = np.array([0.0, 10.0]), np.array([0.25, 0.75])
    q = batch_weighted_quantiles(v, w, [0.25, 0.50])

    assert q[0] == pytest.approx(0.0)  # exact
    assert q[1] == pytest.approx(10 / 3)  # interpolated


def test_quantile_median_consistency():
    v = np.array([8.0, 1.0, 5.0, 3.0])
    w = np.array([0.1, 0.2, 0.3, 0.4])

    assert batch_weighted_quantiles(v, w, [0.5])[0] == pytest.approx(
        weighted_median(v, w)
    )


Y = np.array(
    [
        [1.0, 2.0, 4.0, 8.0],
        [2.0, 3.0, 6.0, 9.0],
    ]
)


def test_vanilla_band():
    cases = [
        ("upper", 0.50, [2.0, 3.0]),  # exact
        ("upper", 0.60, [2.8, 4.2]),  # interpolated
        ("lower", 0.50, [4.0, 6.0]),  # exact
        ("lower", 0.60, [3.2, 4.8]),  # interpolated
    ]
    for kind, scp, expected in cases:
        assert vanilla_band(Y, scp, kind) == pytest.approx(expected)


def test_weighted_band():
    w = np.array([0.1, 0.2, 0.3, 0.4])
    cases = [
        ("upper", 0.30, [2.0, 3.0]),  # exact
        ("upper", 0.40, [8 / 3, 4]),  # interpolated
        ("lower", 0.70, [4, 6]),  # exact
        ("lower", 0.80, [3, 4.5]),  # interpolated
    ]
    for kind, scp, expected in cases:
        assert weighted_band(Y, w, scp, kind) == pytest.approx(expected)


def test_uniform_weight_band_consistency():
    w = np.ones(Y.shape[1]) / Y.shape[1]

    for kind in ["upper", "lower"]:
        for scp in [0.25, 0.50, 0.75, 1.0]:
            print(Y, w, scp, kind)
            print(weighted_band(Y, w, scp, kind))
            print(vanilla_band(Y, scp, kind))
            assert weighted_band(Y, w, scp, kind) == pytest.approx(
                vanilla_band(Y, scp, kind)
            )


if __name__ == "__main__":
    tests = [
        test_weighted_median,
        test_weighted_quantiles,
        test_quantile_median_consistency,
        test_vanilla_band,
        test_weighted_band,
        test_uniform_weight_band_consistency,
    ]

    for test in tests:
        test()
        print(f"✓ {test.__name__}")

    print(f"\n✓ ALL {len(tests)} TESTS PASSED")
