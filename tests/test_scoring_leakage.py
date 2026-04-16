import numpy as np
import pandas as pd

from plasmid_priority.scoring.core import _linear_residual_series


def test_residual_series_leakage():
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=[1, 2, 3, 4, 5])

    predictors = pd.DataFrame({
        "log1p_member_count": [0.1, 0.2, 0.3, 20.0, 50.0],
    }, index=[1, 2, 3, 4, 5])

    fit_mask = pd.Series([True, True, True, False, False], index=[1, 2, 3, 4, 5])

    # Predictors clearly separate 4,5. If test set is included in regression,
    # coefficients change heavily.
    res_with_mask = _linear_residual_series(values, predictors, fit_mask=fit_mask)

    # Change the test-set feature heavily and see if it impacts the test predictions
    # Actually wait, coefficients only fit on fit_mask
    predictors_mod = predictors.copy()
    predictors_mod.loc[4, "log1p_member_count"] = 1000.0
    res_mod = _linear_residual_series(values, predictors_mod, fit_mask=fit_mask)

    # The mask should apply tightly, so modifying a non-masked row should only affect that row's residual
    # and not change the regression coefficients. So first 3 residuals must be exactly the same.
    np.testing.assert_allclose(res_with_mask.iloc[:3], res_mod.iloc[:3])
