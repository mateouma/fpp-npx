import numpy as np
import statsmodels.api as sm

def harmonic_regression(t, y, f, K):
    """
    Remove harmonics
    """
    X = np.column_stack([
        func(2 * np.pi * k * f * t)
        for k in range(1, K+1)
        for func in (np.sin, np.cos)
    ])
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    y_fit = model.fittedvalues
    residuals = y - y_fit

    return y_fit, residuals, model