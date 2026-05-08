# methods to approximate muscle-tendon geometry using surrogate models
# --------------------------------------------------------------

import numpy as np
from scipy.stats import qmc
import os
import pandas as pd

# functions to add
# create dummy motion (data to fit model)
# function to fit surrogate model to data
# functions to fit polynomial
from general_utilities import WriteMotionFile

def create_dummy_motion(coordinates, n_datapoints, coordinates_fit,
                        lower_bound, upper_bound,
                        coordinate_defaults= None,
                        mot_outfile = None):
    # coordinates: list of coordinate names (e.g., ['hip_flexion', 'knee_angle'])
    # n_datapoints: number of datapoints
    # lower_bound: lower bound coordinates
    # upper_bound: upper bound coordinates
    # coordinate_defaults: default value for all coordinates in the model
    if coordinate_defaults is None:
        coordinate_defaults = [0.0] * len(coordinates)
    if mot_outfile is None:
        mot_outfile = os.path.join(os.getcwd(),'dummy_lhs_ik.mot')

    # create an inverse kinematics file
    # Start from the default pose for every sample.
    data_matrix = np.tile(coordinate_defaults, (n_datapoints, 1))

    # Latin hypercube sampling for the selected coordinates.
    sampler = qmc.LatinHypercube(d=len(coordinates_fit), seed=1)
    samples = sampler.random(n=n_datapoints)
    samples_scaled = qmc.scale(samples, lower_bound, upper_bound)

    for i, coord_name in enumerate(coordinates_fit):
        coord_idx = coordinates.index(coord_name)
        data_matrix[:, coord_idx] = samples_scaled[:, i]

    # Add a time column and write a .mot file.
    time = np.linspace(0.0, n_datapoints - 1, n_datapoints)
    mot_matrix = np.column_stack([time, data_matrix])
    mot_headers = ["time"] + coordinates
    WriteMotionFile(mot_matrix, mot_headers, mot_outfile)
    return mot_matrix


def _all_powers(n_var, max_order):
    powers = []

    def rec_build(prefix, remaining, idx):
        if idx == n_var - 1:
            powers.append(prefix + [remaining])
            return
        for p in range(remaining + 1):
            rec_build(prefix + [p], remaining - p, idx + 1)

    for total_order in range(max_order + 1):
        rec_build([], total_order, 0)
    return np.asarray(powers, dtype=int)


def _monomial_matrix(x, powers):
    n_points = x.shape[0]
    phi = np.ones((n_points, powers.shape[0]))
    for j in range(powers.shape[0]):
        for i in range(x.shape[1]):
            p = powers[j, i]
            if p > 0:
                phi[:, j] *= x[:, i] ** p
    return phi


def _jacobian_monomial_matrix(x, powers):
    n_points, n_var = x.shape
    n_terms = powers.shape[0]
    jac = np.zeros((n_points * n_var, n_terms))

    for k in range(n_var):
        row_slice = slice(k * n_points, (k + 1) * n_points)
        for j in range(n_terms):
            p_k = powers[j, k]
            if p_k == 0:
                continue
            term = np.ones(n_points) * p_k
            for i in range(n_var):
                p = powers[j, i] - (1 if i == k else 0)
                if p > 0:
                    term *= x[:, i] ** p
            jac[row_slice, j] = term
    return jac


def _rmse(y_true, y_pred, axis=None):
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=axis))


def _fit_mvpoly(x, y, order_bounds=(1, 5), ydx=None,
                threshold_rmse_y=0.003,
                threshold_rmse_ydx=0.003,
                threshold_rel_err=0.05):
    # PredSim-inspired fit: scan polynomial order and include derivative constraints.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if ydx is not None:
        ydx = np.asarray(ydx, dtype=float)

    mu_mean = np.mean(x, axis=0)
    mu_std = np.std(x, axis=0)
    mu_std[mu_std == 0] = 1.0
    x_sc = (x - mu_mean) / mu_std

    best = None
    fit_flag = "max_order"
    chosen_order = int(order_bounds[1])

    for order in range(int(order_bounds[0]), int(order_bounds[1]) + 1):
        powers = _all_powers(x.shape[1], order)
        phi = _monomial_matrix(x_sc, powers)

        if ydx is None:
            x_aug = phi
            y_aug = y
        else:
            jac = _jacobian_monomial_matrix(x_sc, powers)
            # Chain rule to map derivatives from scaled variables to original variables.
            for k in range(x.shape[1]):
                row_slice = slice(k * x.shape[0], (k + 1) * x.shape[0])
                jac[row_slice, :] /= mu_std[k]
            x_aug = np.vstack([phi, jac])
            y_aug = np.concatenate([y, ydx.reshape(-1)])

        coeff = np.linalg.lstsq(x_aug, y_aug, rcond=None)[0]
        y_fit = phi @ coeff
        rmse_y = float(_rmse(y, y_fit))
        rel_y = np.abs(y - y_fit) < threshold_rel_err * np.maximum(np.abs(y), 1e-12)

        if ydx is not None:
            ydx_fit = (_jacobian_monomial_matrix(x_sc, powers) @ coeff).reshape(x.shape[1], x.shape[0]).T
            ydx_fit = ydx_fit / mu_std
            rmse_ydx = _rmse(ydx, ydx_fit, axis=0)
            rel_ydx = np.abs(ydx - ydx_fit) < threshold_rel_err * np.maximum(np.abs(ydx), 1e-12)
        else:
            ydx_fit = None
            rmse_ydx = np.array([])
            rel_ydx = np.array([True])

        best = {
            "coeff": coeff,
            "powers": powers,
            "mu": np.vstack([mu_mean, mu_std]),
            "order": order,
            "rmse_y": rmse_y,
            "rmse_ydx": rmse_ydx,
            "y_fit": y_fit,
            "ydx_fit": ydx_fit,
        }
        chosen_order = order

        if rmse_y < threshold_rmse_y and (rmse_ydx.size == 0 or np.all(rmse_ydx < threshold_rmse_ydx)):
            fit_flag = "rmse"
            break
        if np.all(rel_y) and np.all(rel_ydx):
            fit_flag = "rel_err"
            break

    best["fit_flag"] = fit_flag
    best["order"] = chosen_order
    return best


def fit_polynomials_for_muscles(ik_data, lmt_data, dm_data, coordinates_fit,
                                muscles_selected=None,
                                order_bounds=(1, 5),
                                threshold_rmse_y=0.003,
                                threshold_rmse_ydx=0.003,
                                threshold_rel_err=0.05,
                                min_moment_arm=1e-4):
    # Convert to DataFrame if arrays are provided.
    if not isinstance(ik_data, pd.DataFrame):
        ik_data = pd.DataFrame(ik_data)
    if not isinstance(lmt_data, pd.DataFrame):
        lmt_data = pd.DataFrame(lmt_data)
    if not isinstance(dm_data, pd.DataFrame):
        dm_data = pd.DataFrame(dm_data)

    q_deg = ik_data[coordinates_fit].to_numpy()
    q = np.deg2rad(q_deg)

    if muscles_selected is None:
        muscles_selected = [c for c in lmt_data.columns if c != "time"]

    muscle_models = []
    for muscle_name in muscles_selected:
        if muscle_name not in lmt_data.columns:
            continue

        crossing_coords = []
        dm_cols = []
        for coord_name in coordinates_fit:
            col_name = f"{muscle_name}_{coord_name}"
            if col_name in dm_data.columns:
                if np.max(np.abs(dm_data[col_name].to_numpy())) > min_moment_arm:
                    crossing_coords.append(coord_name)
                    dm_cols.append(col_name)

        if len(crossing_coords) == 0:
            continue

        idx = [coordinates_fit.index(c) for c in crossing_coords]
        x_muscle = q[:, idx]
        y_muscle = lmt_data[muscle_name].to_numpy()
        # dM = -d(lMT)/dq in PredSim convention.
        ydx_muscle = -dm_data[dm_cols].to_numpy()

        fit_res = _fit_mvpoly(
            x_muscle,
            y_muscle,
            order_bounds=order_bounds,
            ydx=ydx_muscle,
            threshold_rmse_y=threshold_rmse_y,
            threshold_rmse_ydx=threshold_rmse_ydx,
            threshold_rel_err=threshold_rel_err,
        )

        muscle_models.append({
            "m_name": muscle_name,
            "DOF": crossing_coords,
            "coeff": fit_res["coeff"],
            "powers": fit_res["powers"],
            "order": fit_res["order"],
            "mu": fit_res["mu"],
            "lMT_error_rms": fit_res["rmse_y"],
            "dm_error_rms": fit_res["rmse_ydx"],
            "fit_flag": fit_res["fit_flag"],
            "stats": {
                "y_fit": fit_res["y_fit"],
                "ydx_fit": fit_res["ydx_fit"],
            },
        })

    return {"coordinates_fit": list(coordinates_fit), "muscle": muscle_models}


def evaluate_single_muscle_polynomial(muscle_model, q_data, coordinates_fit, q_in_degrees=True):
    if isinstance(q_data, pd.DataFrame):
        q_full = q_data[coordinates_fit].to_numpy(dtype=float)
    else:
        q_full = np.asarray(q_data, dtype=float)

    if q_in_degrees:
        q_full = np.deg2rad(q_full)

    idx = [coordinates_fit.index(dof) for dof in muscle_model["DOF"]]
    x = q_full[:, idx]

    mu = np.asarray(muscle_model["mu"], dtype=float)
    x_sc = (x - mu[0, :]) / mu[1, :]

    phi = _monomial_matrix(x_sc, muscle_model["powers"])
    lmt_pred = phi @ muscle_model["coeff"]

    jac_sc = _jacobian_monomial_matrix(x_sc, muscle_model["powers"])
    jac = (jac_sc @ muscle_model["coeff"]).reshape(len(muscle_model["DOF"]), x.shape[0]).T
    jac = jac / mu[1, :]
    dm_pred = -jac

    return lmt_pred, dm_pred


def evaluate_muscle_polynomials(muscle_poly_info, q_data, coordinates_fit=None, q_in_degrees=True):
    if coordinates_fit is None:
        coordinates_fit = muscle_poly_info.get("coordinates_fit")
        if coordinates_fit is None:
            raise ValueError("coordinates_fit not provided and not found in muscle_poly_info")

    if isinstance(q_data, pd.DataFrame):
        n_points = len(q_data)
        time_col = q_data["time"].to_numpy() if "time" in q_data.columns else None
    else:
        q_data = np.asarray(q_data)
        n_points = q_data.shape[0]
        time_col = None

    lmt_pred = pd.DataFrame(index=np.arange(n_points))
    dm_pred = pd.DataFrame(index=np.arange(n_points))

    if time_col is not None:
        lmt_pred["time"] = time_col
        dm_pred["time"] = time_col

    for muscle_model in muscle_poly_info.get("muscle", []):
        lmt_m, dm_m = evaluate_single_muscle_polynomial(
            muscle_model, q_data, coordinates_fit, q_in_degrees=q_in_degrees
        )
        m_name = muscle_model["m_name"]
        lmt_pred[m_name] = lmt_m
        for j, dof in enumerate(muscle_model["DOF"]):
            dm_pred[f"{m_name}_{dof}"] = dm_m[:, j]

    return lmt_pred, dm_pred


def create_casadi_muscle_polynomial_function(muscle_poly_info, function_name="muscle_geom_surrogate"):
    try:
        import casadi as ca
    except ImportError as exc:
        raise ImportError("casadi is required to build the surrogate function") from exc

    coordinates_fit = muscle_poly_info.get("coordinates_fit")
    if coordinates_fit is None:
        raise ValueError("coordinates_fit not found in muscle_poly_info")

    muscle_models = muscle_poly_info.get("muscle", [])
    n_q = len(coordinates_fit)
    n_m = len(muscle_models)

    q = ca.SX.sym("q", n_q)
    lmt = ca.SX.zeros(n_m, 1)
    dm = ca.SX.zeros(n_m, n_q)

    for im, muscle_model in enumerate(muscle_models):
        dof_idx = [coordinates_fit.index(dof) for dof in muscle_model["DOF"]]
        mu = np.asarray(muscle_model["mu"], dtype=float)
        coeff = np.asarray(muscle_model["coeff"], dtype=float).reshape(-1)
        powers = np.asarray(muscle_model["powers"], dtype=int)

        x_sc = []
        for j_local, j_global in enumerate(dof_idx):
            x_sc.append((q[j_global] - float(mu[0, j_local])) / float(mu[1, j_local]))

        expr = 0
        for term_idx in range(powers.shape[0]):
            monomial = 1
            for j_local in range(powers.shape[1]):
                p = int(powers[term_idx, j_local])
                if p > 0:
                    monomial *= x_sc[j_local] ** p
            expr += float(coeff[term_idx]) * monomial

        lmt[im] = expr
        for j_global in dof_idx:
            dm[im, j_global] = -ca.jacobian(expr, q[j_global])

    return ca.Function(function_name, [q], [lmt, dm], ["q"], ["lMT", "dM"])


def create_casadi_muscle_polynomial_named_output_function(muscle_poly_info,
                                                          function_name="muscle_geom_surrogate_named"):
    try:
        import casadi as ca
    except ImportError as exc:
        raise ImportError("casadi is required to build the surrogate function") from exc

    coordinates_fit = muscle_poly_info.get("coordinates_fit")
    if coordinates_fit is None:
        raise ValueError("coordinates_fit not found in muscle_poly_info")

    muscle_models = muscle_poly_info.get("muscle", [])
    n_q = len(coordinates_fit)
    q = ca.SX.sym("q", n_q)

    outputs = []
    output_names = []
    for muscle_model in muscle_models:
        m_name = muscle_model["m_name"]
        dof_idx = [coordinates_fit.index(dof) for dof in muscle_model["DOF"]]
        mu = np.asarray(muscle_model["mu"], dtype=float)
        coeff = np.asarray(muscle_model["coeff"], dtype=float).reshape(-1)
        powers = np.asarray(muscle_model["powers"], dtype=int)

        x_sc = []
        for j_local, j_global in enumerate(dof_idx):
            x_sc.append((q[j_global] - float(mu[0, j_local])) / float(mu[1, j_local]))

        expr = 0
        for term_idx in range(powers.shape[0]):
            monomial = 1
            for j_local in range(powers.shape[1]):
                p = int(powers[term_idx, j_local])
                if p > 0:
                    monomial *= x_sc[j_local] ** p
            expr += float(coeff[term_idx]) * monomial

        dm_row = ca.SX.zeros(n_q, 1)
        for j_global in dof_idx:
            dm_row[j_global] = -ca.jacobian(expr, q[j_global])

        outputs.extend([expr, dm_row])
        output_names.extend([f"lMT_{m_name}", f"dM_{m_name}"])

    return ca.Function(function_name, [q], outputs, ["q"], output_names)
