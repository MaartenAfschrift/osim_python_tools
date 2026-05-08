from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Allow importing the local module without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surrogate_muscle_tendon_geom import (  # noqa: E402
    fit_polynomials_for_muscles,
    evaluate_muscle_polynomials,
)


def test_polynomial_evaluator_matches_fit_stats():
    muscle_name = "test_muscle"
    coordinates_fit = ["hip_flexion_r", "knee_angle_r"]

    n = 120
    q1_deg = np.linspace(-40.0, 40.0, n)
    q2_deg = np.linspace(-60.0, 5.0, n)
    q_deg = np.column_stack([q1_deg, q2_deg])
    q = np.deg2rad(q_deg)

    # Synthetic 2D polynomial in radians.
    lmt = (
        0.35
        + 0.08 * q[:, 0]
        - 0.03 * q[:, 1]
        + 0.12 * q[:, 0] ** 2
        - 0.05 * q[:, 0] * q[:, 1]
        + 0.04 * q[:, 1] ** 2
    )

    # dM = -d(lMT)/dq
    dl_dq1 = 0.08 + 0.24 * q[:, 0] - 0.05 * q[:, 1]
    dl_dq2 = -0.03 - 0.05 * q[:, 0] + 0.08 * q[:, 1]
    dm1 = -dl_dq1
    dm2 = -dl_dq2

    ik_data = pd.DataFrame(
        {
            "time": np.linspace(0.0, 1.0, n),
            "hip_flexion_r": q1_deg,
            "knee_angle_r": q2_deg,
        }
    )
    lmt_data = pd.DataFrame({"time": ik_data["time"], muscle_name: lmt})
    dm_data = pd.DataFrame(
        {
            "time": ik_data["time"],
            f"{muscle_name}_hip_flexion_r": dm1,
            f"{muscle_name}_knee_angle_r": dm2,
        }
    )

    muscle_poly_info = fit_polynomials_for_muscles(
        ik_data=ik_data,
        lmt_data=lmt_data,
        dm_data=dm_data,
        coordinates_fit=coordinates_fit,
        muscles_selected=[muscle_name],
        order_bounds=(2, 2),
        threshold_rmse_y=1e-12,
        threshold_rmse_ydx=1e-12,
        threshold_rel_err=1e-9,
    )

    assert len(muscle_poly_info["muscle"]) == 1
    model = muscle_poly_info["muscle"][0]

    lmt_pred, dm_pred = evaluate_muscle_polynomials(
        muscle_poly_info,
        q_data=ik_data,
        q_in_degrees=True,
    )

    # Evaluator output should match the stored fit outputs on training data.
    np.testing.assert_allclose(
        lmt_pred[muscle_name].to_numpy(), model["stats"]["y_fit"], rtol=1e-10, atol=1e-10
    )

    ydx_fit = model["stats"]["ydx_fit"]
    np.testing.assert_allclose(
        dm_pred[f"{muscle_name}_hip_flexion_r"].to_numpy(), -ydx_fit[:, 0], rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        dm_pred[f"{muscle_name}_knee_angle_r"].to_numpy(), -ydx_fit[:, 1], rtol=1e-10, atol=1e-10
    )


if __name__ == "__main__":
    try:
        test_polynomial_evaluator_matches_fit_stats()
        print("test_polynomial_evaluator_matches_fit_stats: PASSED")
    except Exception as exc:
        print("test_polynomial_evaluator_matches_fit_stats: FAILED")
        print(exc)
        raise
