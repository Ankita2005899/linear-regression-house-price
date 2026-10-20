import numpy as np
from src.from_scratch import LinearRegressionScratch

def test_scratch_recovers_coefficients():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2)); y = 3 + 2 * X[:, 0] - 1 * X[:, 1]
    m = LinearRegressionScratch(method="normal").fit(X, y)
    assert abs(m.intercept_ - 3) < 1e-6 and np.allclose(m.coef_, [2, -1], atol=1e-6)
