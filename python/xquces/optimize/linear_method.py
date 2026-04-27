from __future__ import annotations

import math
from typing import Any, Callable, Union

import numpy as np
from pyscf.lib.linalg_helper import safe_eigh
from scipy.optimize import OptimizeResult, minimize
from scipy.sparse.linalg import LinearOperator


def _jacobian_finite_diff(
    params_to_vec: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    dim: int,
    epsilon: float,
) -> np.ndarray:
    jac = np.zeros((dim, len(params)), dtype=complex)
    for i in range(len(params)):
        unit = np.zeros(len(params), dtype=float)
        unit[i] = epsilon
        jac[:, i] = (params_to_vec(params + unit) - params_to_vec(params - unit)) / (
            2 * epsilon
        )
    return jac


def _orthogonalize_columns(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    coeffs = vec.T.conj() @ mat
    return mat - vec.reshape((-1, 1)) * coeffs.reshape((1, -1))


def _compute_adaptive_tikhonov(
    overlap_mat: np.ndarray,
    target_cond: float,
) -> float:
    S_pp = overlap_mat[1:, 1:]
    eigvals = np.linalg.eigvalsh(S_pp)
    eigvals = eigvals[eigvals > 0]
    if len(eigvals) == 0:
        return 1e-3
    lam_max = float(eigvals[-1])
    lam_min = float(eigvals[0])
    if lam_min <= 0 or lam_max / lam_min <= target_cond:
        return 0.0
    mu = (lam_max - target_cond * lam_min) / (target_cond - 1.0)
    return max(mu, 0.0)


def _linear_method_matrices(
    vec: np.ndarray,
    jac: np.ndarray,
    hamiltonian: LinearOperator,
) -> tuple[np.ndarray, np.ndarray]:
    _, n_params = jac.shape
    energy_mat = np.zeros((n_params + 1, n_params + 1), dtype=complex)
    overlap_mat = np.zeros_like(energy_mat)

    energy_mat[0, 0] = np.vdot(vec, hamiltonian @ vec)
    ham_jac = hamiltonian @ jac
    energy_mat[0, 1:] = vec.conj() @ ham_jac
    energy_mat[1:, 0] = energy_mat[0, 1:].conj()
    energy_mat[1:, 1:] = jac.T.conj() @ ham_jac

    overlap_mat[0, 0] = 1.0
    overlap_mat[0, 1:] = vec.conj() @ jac
    overlap_mat[1:, 0] = overlap_mat[0, 1:].conj()
    overlap_mat[1:, 1:] = jac.T.conj() @ jac

    return energy_mat.real, overlap_mat.real


def _solve_linear_method_eigensystem(
    energy_mat: np.ndarray,
    overlap_mat: np.ndarray,
    regularization: float,
    lindep: float,
) -> tuple[float, np.ndarray]:
    n_params = energy_mat.shape[0] - 1
    energy_mat_reg = energy_mat.copy()
    energy_mat_reg[1:, 1:] += regularization * np.eye(n_params)
    eigs, vecs, _ = safe_eigh(energy_mat_reg, overlap_mat, lindep)
    for eig, vec in zip(eigs, vecs.T):
        reference_coeff = vec[0]
        if (
            np.isfinite(eig)
            and np.all(np.isfinite(vec))
            and abs(reference_coeff) > 1e-10
        ):
            scaled = (vec / reference_coeff).real
            if np.all(np.isfinite(scaled)):
                return float(eig), scaled

    no_update = np.zeros(n_params + 1, dtype=np.float64)
    no_update[0] = 1.0
    return float(energy_mat[0, 0]), no_update


def _get_param_update(
    energy_mat: np.ndarray,
    overlap_mat: np.ndarray,
    regularization: float,
    variation: float,
    lindep: float,
) -> np.ndarray:
    _, param_variations = _solve_linear_method_eigensystem(
        energy_mat, overlap_mat, regularization, lindep=lindep
    )
    average_overlap = float(
        np.real(np.dot(param_variations, overlap_mat @ param_variations))
    )
    if not np.isfinite(average_overlap) or average_overlap < -1:
        return np.zeros(energy_mat.shape[0] - 1, dtype=np.float64)
    average_overlap = max(average_overlap, 0.0)
    numerator = (1 - variation) * average_overlap
    denominator = (1 - variation) + variation * math.sqrt(1 + average_overlap)
    update = param_variations[1:] / (1 + numerator / denominator)
    if not np.all(np.isfinite(update)):
        return np.zeros_like(update)
    return update


def minimize_linear_method(
    params_to_vec: Callable[[np.ndarray], np.ndarray],
    hamiltonian: LinearOperator,
    x0: np.ndarray,
    *,
    jac: Callable[[np.ndarray], np.ndarray] | None = None,
    maxiter: int = 1000,
    lindep: float = 1e-8,
    epsilon: float = 1e-8,
    ftol: float = 1e-8,
    gtol: float = 1e-5,
    regularization: float = 1e-4,
    regularization_max: float | None = 1.0,
    variation: float = 0.5,
    tikhonov: Union[float, str, None] = None,
    tikhonov_target_cond: float = 1e6,
    tikhonov_max: float | None = None,
    optimize_regularization: bool = True,
    optimize_variation: bool = True,
    optimize_kwargs: dict | None = None,
    callback: Callable[[OptimizeResult], Any] | None = None,
) -> OptimizeResult:
    """Minimize energy using the linear method.

    A drop-in replacement for ffsim's ``minimize_linear_method`` with two additions:

    * **Analytic Jacobian** (``jac`` parameter): when provided, the Jacobian is
      computed analytically instead of via finite differences.  This eliminates the
      ``O(n_params)`` factor of full state-vector evaluations per iteration.

      ``jac(params)`` must return a real or complex array of shape
      ``(hilbert_space_dim, n_params)`` where column ``k`` is
      ``d|ψ(params)⟩ / d params[k]``.

      For the iGCR diagonal operator ``D(θ_D)`` the Jacobian column for parameter
      ``k`` is simply ``1j * Ô_k * |ψ⟩`` where ``Ô_k`` is the eigenvalue of the
      corresponding number operator in the Fock basis — a cheap element-wise
      multiplication performed once after computing ``|ψ⟩``.

    * **Regularization cap** (``regularization_max``): prevents ``regularization``
      from drifting above this value across iterations when
      ``optimize_regularization=True``.  Defaults to ``1.0``.

    Parameters
    ----------
    params_to_vec:
        Maps parameter vector to state vector ``|ψ(params)⟩``.
    hamiltonian:
        Hamiltonian as a ``LinearOperator``.
    x0:
        Initial parameter vector.
    jac:
        Optional analytic Jacobian.  If ``None``, falls back to central finite
        differences with step ``epsilon``.
    maxiter:
        Maximum number of iterations.
    lindep:
        Linear-dependence threshold passed to ``safe_eigh``.
    epsilon:
        Finite-difference step size (only used when ``jac=None``).
    ftol:
        Convergence tolerance on relative energy change.
    gtol:
        Convergence tolerance on gradient infinity-norm.
    regularization:
        Initial diagonal shift on the energy matrix.
    regularization_max:
        Hard upper cap on ``regularization`` after each inner optimization.
        ``None`` disables the cap.
    variation:
        Initial variation parameter in ``[0, 1]``.
    tikhonov:
        Tikhonov shift on the overlap matrix.  ``"auto"`` for adaptive,
        ``None`` or ``0`` to disable, or a fixed positive float.
    tikhonov_target_cond:
        Target condition number used by ``tikhonov="auto"``.
    tikhonov_max:
        Hard cap on the adaptive Tikhonov shift.  ``None`` disables the cap.
    optimize_regularization:
        Whether to optimize ``regularization`` each iteration.
    optimize_variation:
        Whether to optimize ``variation`` each iteration.
    optimize_kwargs:
        Keyword arguments forwarded to ``scipy.optimize.minimize`` for the inner
        regularization/variation optimization.  Defaults to
        ``{"method": "L-BFGS-B"}``.
    callback:
        Called after each iteration with an ``OptimizeResult`` containing
        diagnostic fields.
    """
    if regularization < 0:
        raise ValueError(f"regularization must be nonnegative. Got {regularization}.")
    if not 0 <= variation <= 1:
        raise ValueError(f"variation must be between 0 and 1. Got {variation}.")
    if maxiter < 1:
        raise ValueError(f"maxiter must be at least 1. Got {maxiter}.")
    if tikhonov is not None and tikhonov != "auto":
        tikhonov = float(tikhonov)
        if tikhonov < 0:
            raise ValueError(f"tikhonov must be nonnegative. Got {tikhonov}.")

    if optimize_kwargs is None:
        optimize_kwargs = dict(method="L-BFGS-B")

    params = x0.copy()
    success = False
    message = "Stop: Total number of iterations reached limit."
    nfev = 0
    njev = 0
    nlinop = 0

    def _call_params_to_vec(p: np.ndarray) -> np.ndarray:
        nonlocal nfev
        nfev += 1
        return params_to_vec(p)

    def _apply_hamiltonian(v: np.ndarray) -> np.ndarray:
        nonlocal nlinop
        if v.ndim == 1:
            nlinop += 1
        else:
            nlinop += v.shape[1]
        return hamiltonian @ v

    grad = np.zeros_like(params)
    _prev_energy: float | None = None

    for i in range(maxiter):
        vec = _call_params_to_vec(params)

        if jac is not None:
            jac_mat = jac(params)
            njev += 1
        else:
            jac_mat = _jacobian_finite_diff(
                _call_params_to_vec, params, len(vec), epsilon
            )
            njev += 1

        jac_mat = _orthogonalize_columns(jac_mat, vec)

        energy_mat, overlap_mat = _linear_method_matrices(vec, jac_mat, hamiltonian)
        nlinop += jac_mat.shape[1] + 1  # ham @ jac columns + ham @ vec

        energy = float(energy_mat[0, 0])
        grad = 2 * energy_mat[0, 1:]

        S_pp = overlap_mat[1:, 1:]
        svals = np.linalg.svd(S_pp, compute_uv=False)
        svals_sorted = np.sort(svals)
        s_max = float(np.max(svals))
        s_min = float(np.min(svals))
        cond_S = float(s_max / s_min) if s_min > 0 else float("inf")
        rank_S = int(np.sum(svals > 1e-10 * s_max))
        n_soft = (
            int(np.sum(svals_sorted < 1e-8 * s_max)) if s_max > 0 else len(svals_sorted)
        )
        svals_small = svals_sorted[: min(10, len(svals_sorted))].copy()

        if tikhonov == "auto":
            tikhonov_mu = _compute_adaptive_tikhonov(
                overlap_mat, target_cond=tikhonov_target_cond
            )
            if tikhonov_max is not None:
                tikhonov_mu = min(tikhonov_mu, tikhonov_max)
        elif tikhonov is not None and float(tikhonov) > 0:
            tikhonov_mu = float(tikhonov)
        else:
            tikhonov_mu = 0.0

        if tikhonov_mu > 0:
            overlap_mat_reg = overlap_mat.copy()
            n_params_lm = overlap_mat.shape[0] - 1
            overlap_mat_reg[1:, 1:] += tikhonov_mu * np.eye(n_params_lm)
            svals_reg = np.linalg.svd(overlap_mat_reg[1:, 1:], compute_uv=False)
            s_max_reg = float(np.max(svals_reg))
            s_min_reg = float(np.min(svals_reg))
            cond_S_reg = float(s_max_reg / s_min_reg) if s_min_reg > 0 else float("inf")
        else:
            overlap_mat_reg = overlap_mat
            cond_S_reg = cond_S

        previous_energy = _prev_energy
        _prev_energy = energy

        njev += 1

        if previous_energy is not None and (
            abs(previous_energy - energy) / max(abs(previous_energy), abs(energy), 1.0)
            <= ftol
        ):
            success = True
            message = "Convergence: Relative reduction of objective function <= ftol."
            break
        if np.max(np.abs(grad)) <= gtol:
            success = True
            message = "Convergence: Norm of gradient <= gtol."
            break

        if optimize_regularization and optimize_variation:

            def _f_both(x: np.ndarray) -> float:
                try:
                    reg_ = x[0] ** 2
                    var_ = 0.5 * (1 + math.tanh(x[1]))
                    p_update = _get_param_update(
                        energy_mat,
                        overlap_mat_reg,
                        reg_,
                        var_,
                        lindep,
                    )
                    if not np.all(np.isfinite(p_update)):
                        return float("inf")
                    v = _call_params_to_vec(params + p_update)
                    value = float(np.vdot(v, _apply_hamiltonian(v)).real)
                    return value if np.isfinite(value) else float("inf")
                except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                    return float("inf")

            reg_param = math.sqrt(regularization)
            var_param = math.atanh(2 * min(1 - 1e-8, max(1e-8, variation)) - 1)
            res = minimize(_f_both, x0=[reg_param, var_param], **optimize_kwargs)
            reg_param, var_param = res.x
            regularization = reg_param**2
            if regularization_max is not None:
                regularization = min(regularization, regularization_max)
            variation = 0.5 * (1 + math.tanh(var_param))

        elif optimize_regularization:

            def _f_reg(x: np.ndarray) -> float:
                try:
                    reg_ = x[0] ** 2
                    p_update = _get_param_update(
                        energy_mat,
                        overlap_mat_reg,
                        reg_,
                        variation,
                        lindep,
                    )
                    if not np.all(np.isfinite(p_update)):
                        return float("inf")
                    v = _call_params_to_vec(params + p_update)
                    value = float(np.vdot(v, _apply_hamiltonian(v)).real)
                    return value if np.isfinite(value) else float("inf")
                except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                    return float("inf")

            reg_param = math.sqrt(regularization)
            res = minimize(_f_reg, x0=[reg_param], **optimize_kwargs)
            (reg_param,) = res.x
            regularization = reg_param**2
            if regularization_max is not None:
                regularization = min(regularization, regularization_max)

        elif optimize_variation:

            def _f_var(x: np.ndarray) -> float:
                try:
                    var_ = 0.5 * (1 + math.tanh(x[0]))
                    p_update = _get_param_update(
                        energy_mat,
                        overlap_mat_reg,
                        regularization,
                        var_,
                        lindep,
                    )
                    if not np.all(np.isfinite(p_update)):
                        return float("inf")
                    v = _call_params_to_vec(params + p_update)
                    value = float(np.vdot(v, _apply_hamiltonian(v)).real)
                    return value if np.isfinite(value) else float("inf")
                except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                    return float("inf")

            var_param = math.atanh(2 * min(1 - 1e-8, max(1e-8, variation)) - 1)
            res = minimize(_f_var, x0=[var_param], **optimize_kwargs)
            (var_param,) = res.x
            variation = 0.5 * (1 + math.tanh(var_param))

        lm_eig, param_variations = _solve_linear_method_eigensystem(
            energy_mat, overlap_mat_reg, regularization, lindep=lindep
        )
        average_overlap = float(
            np.real(np.dot(param_variations, overlap_mat_reg @ param_variations))
        )
        if not np.isfinite(average_overlap) or average_overlap < -1:
            average_overlap = 0.0
            param_update = np.zeros_like(params)
        else:
            average_overlap = max(average_overlap, 0.0)
            numerator = (1 - variation) * average_overlap
            denominator = (1 - variation) + variation * math.sqrt(1 + average_overlap)
            param_update = param_variations[1:] / (1 + numerator / denominator)
            if not np.all(np.isfinite(param_update)):
                param_update = np.zeros_like(params)

        step_2 = float(np.linalg.norm(param_update))
        step_inf = float(np.max(np.abs(param_update))) if len(param_update) else 0.0
        delta_e = (
            float("nan") if previous_energy is None else float(energy - previous_energy)
        )
        rel_delta_e = (
            float("nan")
            if previous_energy is None
            else float(
                abs(previous_energy - energy)
                / max(abs(previous_energy), abs(energy), 1.0)
            )
        )

        intermediate_result = OptimizeResult(
            x=params.copy(),
            fun=energy,
            jac=grad,
            nfev=nfev,
            njev=njev,
            nlinop=nlinop,
            nit=i + 1,
        )
        intermediate_result.energy_mat = energy_mat
        intermediate_result.overlap_mat = overlap_mat
        intermediate_result.regularization = regularization
        intermediate_result.variation = variation
        intermediate_result.tikhonov_mu = tikhonov_mu
        intermediate_result.s_min = s_min
        intermediate_result.s_max = s_max
        intermediate_result.cond_S = cond_S
        intermediate_result.cond_S_reg = cond_S_reg
        intermediate_result.rank_S = rank_S
        intermediate_result.n_soft = n_soft
        intermediate_result.svals_small = svals_small
        intermediate_result.step_2 = step_2
        intermediate_result.step_inf = step_inf
        intermediate_result.delta_e = delta_e
        intermediate_result.rel_delta_e = rel_delta_e
        intermediate_result.lm_pred_eig = float(lm_eig)
        intermediate_result.using_analytic_jac = jac is not None

        if callback is not None:
            callback(intermediate_result)

        params = params + param_update

    vec = _call_params_to_vec(params)
    energy = float(np.vdot(vec, hamiltonian @ vec).real)

    final_result = OptimizeResult(
        x=params,
        success=success,
        message=message,
        fun=energy,
        jac=grad,
        nfev=nfev,
        njev=njev,
        nlinop=nlinop,
        nit=i + 1,
    )
    final_result.regularization = regularization
    final_result.variation = variation
    final_result.tikhonov = tikhonov
    final_result.using_analytic_jac = jac is not None

    if callback is not None:
        callback(final_result)

    return final_result
