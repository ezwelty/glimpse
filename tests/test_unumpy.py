import glimpse.unumpy as unp
import numpy as np
import uncertainties.unumpy


def test_against_uncertainties(tol=1e-6):
    # Build arrays
    n = 3
    mean = np.random.randn(n, n)
    sigma = np.random.randn(n, n)
    Ax = uncertainties.unumpy.uarray(mean, np.abs(sigma))
    Bx = uncertainties.unumpy.uarray(sigma, np.abs(mean))
    Ay = unp.uarray(mean, np.abs(sigma))
    By = unp.uarray(sigma, np.abs(mean))
    # Compute tests
    tests = {
        "A + a": (Ax + 1.5, Ay + 1.5),
        "a + A": (1.5 + Ax, 1.5 + Ay),
        "A - a": (Ax - 1.5, Ay - 1.5),
        "a - A": (1.5 - Ax, 1.5 - Ay),
        "A * a": (Ax * 1.5, Ay * 1.5),
        "a * A": (1.5 * Ax, 1.5 * Ay),
        "A * -a": (Ax * -1.5, Ay * -1.5),
        "-a * A": (-1.5 * Ax, -1.5 * Ay),
        "1 / A": (1 / Ax, 1 / Ay),
        "a / A": (1.5 / Ax, 1.5 / Ay),
        "A^0": (Ax ** 0, Ay ** 0),
        "A^1": (Ax ** 1, Ay ** 1),
        "A^2": (Ax ** 2, Ay ** 2),
        "A * B": (Ax * Bx, Ay * By),
        "A + B": (Ax + Bx, Ay + By),
        "A - B": (Ax - Bx, Ay - By),
        "A / B": (Ax / Bx, Ay / By),
        "sin(A)": (uncertainties.unumpy.sin(Ax), Ay.sin()),
        "cos(A)": (uncertainties.unumpy.cos(Ax), Ay.cos()),
        "abs(A)": (np.abs(Ax), Ay.abs()),
    }
    # Check tests
    for key in tests:
        x, y = tests[key]
        dmean = uncertainties.unumpy.nominal_values(x) - y.mean
        dsigma = uncertainties.unumpy.std_devs(x) - y.sigma
        assert np.all((dmean < tol) & (dsigma < tol)), "Failed " + key
