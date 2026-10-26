import warnings

import numpy as np

try:

    import cupy as cp

    GPU_AVAILABLE = cp.cuda.is_available()

except (ImportError, RuntimeError):

    cp = np

    GPU_AVAILABLE = False

    print("Warning: CuPy not installed. Falling back to CPU (NumPy).")

xp = cp if GPU_AVAILABLE else np

def get_array_module(arr):

    if GPU_AVAILABLE:

        return cp.get_array_module(arr)

    return np

def prox_0(x, mu):

    xp = get_array_module(x)

    Q = x.copy()

    thresh = 2 * mu

    Q[x ** 2 < thresh] = 0

    return Q

def prox_1(x, mu):

    xp = get_array_module(x)

    return xp.sign(x) * xp.maximum(xp.abs(x) - mu, 0)

def prox_p(D, alpha, p, eps=1e-12):

    xp = get_array_module(D)

    absD = xp.abs(D)

    abs_safe = xp.maximum(absD, eps)

    thresh = alpha * abs_safe ** (p - 1)

    dd = xp.maximum(absD - thresh, 0)

    return dd * xp.sign(D)

def prox_2(x, mu):

    return x / (1 + 2 * mu)

def frobenius_norm(matrix):

    xp = get_array_module(matrix)

    return xp.linalg.norm(matrix, 'fro')

def build_knn_mask(D, K):

    xp = get_array_module(D)

    n = D.shape[0]

    K_eff = min(K, n - 1) 

    D_safe = D.copy()

    D_safe[xp.arange(n), xp.arange(n)] = xp.inf

    if GPU_AVAILABLE and isinstance(D, cp.ndarray):

        idx = xp.argsort(D_safe, axis=1)[:, :K_eff]

    else:

        idx = xp.argpartition(D_safe, K_eff, axis=1)[:, :K_eff]

    M = xp.zeros_like(D, dtype=float)

    rows = xp.repeat(xp.arange(n)[:, None], K_eff, axis=1)

    M[rows, idx] = 1.0

    xp.fill_diagonal(M, 0.0)

    return M

def project_C_structural(C, M_knn=None, zero_diag=True, row_normalize=True, eps=1e-12):

    xp = get_array_module(C)

    if zero_diag:

        xp.fill_diagonal(C, 0.0)

    if M_knn is not None:

        C = C * M_knn

    if row_normalize:

        row_sums = C.sum(axis=1, keepdims=True)

        if M_knn is not None:

            zero_rows = (row_sums.squeeze() == 0)

            if xp.any(zero_rows):

                C[zero_rows] = M_knn[zero_rows]

                row_sums = C.sum(axis=1, keepdims=True)

        C = C / (row_sums + eps)

    C = xp.maximum(C, 0.0)

    return C

def ball_projection(M, M_star, beta, eps=1e-12):
    xp = get_array_module(M)
    diff = M - M_star
    norm_diff = xp.linalg.norm(diff, 'fro')
    if norm_diff <= beta:
        return M.copy()
    scale = beta / (norm_diff + eps)
    return M_star + scale * diff

def weighted_ball_projection(M, M_star, beta, W, eps=1e-12):
    xp = get_array_module(M)
    if beta < 0:
        raise ValueError("beta must be nonnegative")
    diff = M - M_star
    weights_sq = W ** 2
    norm_w = float(xp.sqrt(xp.sum(weights_sq * diff ** 2)))
    if norm_w <= beta:
        return M.copy()
    positive = weights_sq[weights_sq > 0]
    if positive.size == 0:
        return M.copy()
    if beta == 0:
        projected_diff = diff.copy()
        projected_diff[weights_sq > 0] = 0.0
        return M_star + projected_diff

    lower = 0.0
    upper = max(1.0, (norm_w / beta - 1.0) / float(xp.min(positive)))
    beta_sq = beta ** 2
    multiplier = upper
    for _ in range(25):
        projected_diff = diff / (1.0 + multiplier * weights_sq)
        weighted_sq_norm = float(xp.sum(weights_sq * projected_diff ** 2))
        residual = weighted_sq_norm - beta_sq
        if residual > 0:
            lower = multiplier
        else:
            upper = multiplier
        if abs(residual) <= 1e-14 * max(1.0, beta_sq):
            break
        derivative = -2.0 * float(xp.sum(
            weights_sq ** 2 * diff ** 2 / (1.0 + multiplier * weights_sq) ** 3
        ))
        newton = multiplier - residual / derivative if derivative < 0 else np.nan
        if np.isfinite(newton) and lower < newton < upper:
            multiplier = newton
        else:
            multiplier = 0.5 * (lower + upper)
    return M_star + diff / (1.0 + upper * weights_sq)

def scale_normalize_BX(B, B1, X, Y1, eps=1e-12):

    xp = get_array_module(B)

    s = xp.linalg.norm(B, 'fro')

    s = max(s, eps)

    B = B / s

    B1 = B1 / s

    X = X * s

    Y1 = Y1 / s

    return B, B1, X, Y1

def moNMTF(Xa, Xb, Xc, opts):

    use_gpu = opts.get('use_gpu', True) and GPU_AVAILABLE

    if use_gpu:

        print("Using GPU acceleration with CuPy")

        xp = cp

        A = cp.asarray(Xa)

        B_star = cp.asarray(Xb)

        C_star = cp.asarray(Xc)

    else:

        print("Using CPU (NumPy)")

        xp = np

        A = Xa.copy()

        B_star = Xb.copy()

        C_star = Xc.copy()

    rho_star = opts['rho_star']

    eta = opts['eta']

    p = opts['p']

    lambda_B = opts['lambda_B']

    lambda_C = opts['lambda_C']

    if 'beta_B' not in opts or 'beta_C' not in opts:

        raise ValueError("moNMTF: It is necessary to manually provide beta-B and beta-C in opts.")

    beta_B = opts['beta_B']

    beta_C = opts['beta_C']

    lambda_X = opts.get('lambda_X', 1e-2)

    use_weighted_ball = opts.get('use_weighted_ball', True)

    alpha_in = opts.get('alpha_in', 0.8)
    alpha_out = opts.get('alpha_out', 0.4)
    adaptive_step = opts.get('adaptive_step', False)
    require_stationarity = opts.get('require_stationarity', adaptive_step)
    stationarity_tol = float(opts.get('stationarity_tol', 1e-3))
    backtrack_factor = float(opts.get('backtrack_factor', 0.5))
    max_backtracks = int(opts.get('max_backtracks', 20))
    descent_slack = float(opts.get('descent_slack', 1e-12))
    if stationarity_tol <= 0:
        raise ValueError("stationarity_tol must be positive")
    if not 0 < backtrack_factor < 1:
        raise ValueError("backtrack_factor must be in (0, 1)")
    if max_backtracks < 0 or descent_slack < 0:
        raise ValueError("max_backtracks and descent_slack must be nonnegative")

    display = opts.get('display', True)
    maxIter = int(opts.get('maxIter', 200))
    minIter = int(opts.get('minIter', 0))
    if maxIter <= 0:
        raise ValueError("maxIter must be positive")
    if minIter < 0 or minIter > maxIter:
        raise ValueError("minIter must satisfy 0 <= minIter <= maxIter")
    epsilon_abs = opts.get('epsilon_abs', 1e-4)
    epsilon_rel = opts.get('epsilon_rel', 1e-3)
    tolerance_rel = opts.get('tolerance_rel', opts.get('tolerance', 1e-6))
    zero_diag_C = opts.get('zero_diag_C', True)

    row_normalize_C = opts.get('row_normalize_C', False)

    scale_every = opts.get('scale_every', 5)

    K = opts.get('K', 20)

    M_knn = opts.get('M_knn', None)

    D = opts.get('D', None)

    B_mask = opts.get('B_mask', None)

    zero_diag_B = opts.get('zero_diag_B', False)

    seed = opts.get('seed', 42)

    m, n = A.shape

    if M_knn is None and (D is not None):

        if use_gpu and not isinstance(D, cp.ndarray):

            D_gpu = cp.asarray(D)

            M_knn = build_knn_mask(D_gpu, K)

        else:

            M_knn = build_knn_mask(D, K)

    elif M_knn is not None and use_gpu and not isinstance(M_knn, cp.ndarray):
        M_knn = cp.asarray(M_knn)

    mask_conflict_fraction = 0.0
    mask_conflict_action = 'none'
    mask_conflict_policy = opts.get('mask_conflict_policy', 'error')
    if mask_conflict_policy not in {'error', 'disable', 'ignore'}:
        raise ValueError("mask_conflict_policy must be one of: error, disable, ignore")
    if M_knn is not None:
        prior_support = C_star != 0
        prior_edge_count = int(xp.count_nonzero(prior_support))
        conflict_count = int(xp.count_nonzero(prior_support & (M_knn == 0)))
        mask_conflict_fraction = conflict_count / max(1, prior_edge_count)
        conflict_tolerance = float(opts.get('mask_conflict_tolerance', 0.05))
        if mask_conflict_fraction > conflict_tolerance:
            message = (
                f"C_star/KNN conflict: {conflict_count}/{prior_edge_count} prior edges "
                f"({mask_conflict_fraction:.1%}) lie outside M_knn"
            )
            if mask_conflict_policy == 'error':
                raise ValueError(message)
            if mask_conflict_policy == 'disable':
                warnings.warn(message + "; disabling M_knn for this run", RuntimeWarning)
                M_knn = None
                mask_conflict_action = 'disabled'
            else:
                warnings.warn(message + "; retaining the conflicting mask", RuntimeWarning)
                mask_conflict_action = 'ignored'

    MB_prior = (B_star != 0).astype(float)

    MC_prior = (C_star != 0).astype(float)

    W_B = alpha_in * MB_prior + alpha_out * (1.0 - MB_prior)

    W_C = alpha_in * MC_prior + alpha_out * (1.0 - MC_prior)

    rng = np.random.RandomState(seed)

    B_init = rng.uniform(0, 1, size=B_star.shape)

    C_init = rng.uniform(0, 1, size=C_star.shape)

    X_init = rng.uniform(0, 1, size=A.shape)

    if use_gpu:

        B = cp.asarray(B_init)

        C = cp.asarray(C_init)

        X = cp.asarray(X_init)

        if B_mask is not None:

            B_mask = cp.asarray(B_mask)

    else:

        B = B_init

        C = C_init

        X = X_init

    if zero_diag_B and B.ndim == 2 and B.shape[0] == B.shape[1]:

        xp.fill_diagonal(B, 0.0)

    if B_mask is not None:

        if B_mask.shape != B.shape:

            raise ValueError(f"The shape of the B_mask {B_mask.shape} is not consistent with the shape of B {B.shape}")

        B *= B_mask

    C = project_C_structural(C, M_knn, zero_diag=zero_diag_C, row_normalize=row_normalize_C)

    B1 = B.copy()

    C1 = C.copy()

    Y1 = xp.zeros_like(B)

    Y2 = xp.zeros_like(C)

    obj = []

    obj_raw = []

    r_norm_list = []

    s_norm_list = []
    stop_reason = 'max_iter'
    converged = False
    eta_B_values = []
    eta_C_values = []
    eta_X_values = []
    stationarity_values = []
    backtrack_counts = []
    backtrack_exhausted = 0

    def constrain_B(candidate):
        if zero_diag_B and candidate.ndim == 2 and candidate.shape[0] == candidate.shape[1]:
            xp.fill_diagonal(candidate, 0.0)
        if B_mask is not None:
            candidate *= B_mask
        candidate = xp.maximum(candidate, 0.0)
        if use_weighted_ball:
            return weighted_ball_projection(candidate, B_star, beta_B, W_B)
        return ball_projection(candidate, B_star, beta_B)

    def constrain_C(candidate):
        candidate = project_C_structural(candidate, M_knn, zero_diag=zero_diag_C, row_normalize=row_normalize_C)
        if use_weighted_ball:
            candidate = weighted_ball_projection(candidate, C_star, beta_C, W_C)
        else:
            candidate = ball_projection(candidate, C_star, beta_C)
        return project_C_structural(candidate, M_knn, zero_diag=zero_diag_C, row_normalize=row_normalize_C)

    for ii in range(maxIter):

        rho_old = rho_star
        gradB = -(A - B @ X @ C) @ C.T @ X.T + rho_old * (B - B1) + Y1
        if adaptive_step:
            eta_B = eta
            current_B_loss = (
                0.5 * float(xp.linalg.norm(A - B @ X @ C, 'fro') ** 2)
                + 0.5 * rho_old * float(xp.linalg.norm(B - B1, 'fro') ** 2)
                + float(xp.sum(Y1 * (B - B1)))
            )
            accepted_B = False
            for bt_B in range(max_backtracks + 1):
                B_new = constrain_B(B - eta_B * gradB)
                candidate_B_loss = (
                    0.5 * float(xp.linalg.norm(A - B_new @ X @ C, 'fro') ** 2)
                    + 0.5 * rho_old * float(xp.linalg.norm(B_new - B1, 'fro') ** 2)
                    + float(xp.sum(Y1 * (B_new - B1)))
                )
                if candidate_B_loss <= current_B_loss + descent_slack * max(1.0, abs(current_B_loss)):
                    accepted_B = True
                    break
                if bt_B < max_backtracks:
                    eta_B *= backtrack_factor
            if not accepted_B:
                backtrack_exhausted += 1
        else:
            eta_B = eta
            bt_B = 0
            B_new = constrain_B(B - eta_B * gradB)
        eta_B_values.append(float(eta_B))

        b = B_new + Y1 / rho_old

        if p == 1:

            B1_new = prox_1(b, lambda_B / rho_old)

        elif p == 0:

            B1_new = prox_0(b, lambda_B / rho_old)

        elif p == 2:

            B1_new = prox_2(b, lambda_B / rho_old)

        else:

            B1_new = prox_p(b, lambda_B / rho_old, p)

        if zero_diag_B and B1_new.ndim == 2 and B1_new.shape[0] == B1_new.shape[1]:

            xp.fill_diagonal(B1_new, 0.0)

        if B_mask is not None:

            B1_new *= B_mask

        B1_new = xp.maximum(B1_new, 0.0)

        gradC = -X.T @ B_new.T @ (A - B_new @ X @ C) + rho_old * (C - C1) + Y2
        if adaptive_step:
            eta_C = eta
            current_C_loss = (
                0.5 * float(xp.linalg.norm(A - B_new @ X @ C, 'fro') ** 2)
                + 0.5 * rho_old * float(xp.linalg.norm(C - C1, 'fro') ** 2)
                + float(xp.sum(Y2 * (C - C1)))
            )
            accepted_C = False
            for bt_C in range(max_backtracks + 1):
                C_new = constrain_C(C - eta_C * gradC)
                candidate_C_loss = (
                    0.5 * float(xp.linalg.norm(A - B_new @ X @ C_new, 'fro') ** 2)
                    + 0.5 * rho_old * float(xp.linalg.norm(C_new - C1, 'fro') ** 2)
                    + float(xp.sum(Y2 * (C_new - C1)))
                )
                if candidate_C_loss <= current_C_loss + descent_slack * max(1.0, abs(current_C_loss)):
                    accepted_C = True
                    break
                if bt_C < max_backtracks:
                    eta_C *= backtrack_factor
            if not accepted_C:
                backtrack_exhausted += 1
        else:
            eta_C = eta
            bt_C = 0
            C_new = constrain_C(C - eta_C * gradC)
        eta_C_values.append(float(eta_C))

        c = C_new + Y2 / rho_old

        if p == 1:

            C1_new = prox_1(c, lambda_C / rho_old)

        elif p == 0:

            C1_new = prox_0(c, lambda_C / rho_old)

        elif p == 2:

            C1_new = prox_2(c, lambda_C / rho_old)

        else:

            C1_new = prox_p(c, lambda_C / rho_old, p)

        C1_new = project_C_structural(C1_new, M_knn, zero_diag=zero_diag_C, row_normalize=row_normalize_C)

        C1_new = xp.maximum(C1_new, 0.0)

        gradX = - B_new.T @ (A - B_new @ X @ C_new) @ C_new.T + lambda_X * X
        if adaptive_step:
            eta_X = eta
            current_X_loss = (
                0.5 * float(xp.linalg.norm(A - B_new @ X @ C_new, 'fro') ** 2)
                + 0.5 * lambda_X * float(xp.sum(X ** 2))
            )
            accepted_X = False
            for bt_X in range(max_backtracks + 1):
                X_new = xp.maximum(X - eta_X * gradX, 0.0)
                candidate_X_loss = (
                    0.5 * float(xp.linalg.norm(A - B_new @ X_new @ C_new, 'fro') ** 2)
                    + 0.5 * lambda_X * float(xp.sum(X_new ** 2))
                )
                if candidate_X_loss <= current_X_loss + descent_slack * max(1.0, abs(current_X_loss)):
                    accepted_X = True
                    break
                if bt_X < max_backtracks:
                    eta_X *= backtrack_factor
            if not accepted_X:
                backtrack_exhausted += 1
        else:
            eta_X = eta
            bt_X = 0
            X_new = xp.maximum(X - eta_X * gradX, 0.0)
        eta_X_values.append(float(eta_X))
        backtrack_counts.append(int(bt_B + bt_C + bt_X))

        gradient_mapping_norm = xp.sqrt(
            (xp.linalg.norm(B_new - B, 'fro') / eta_B) ** 2
            + (xp.linalg.norm(C_new - C, 'fro') / eta_C) ** 2
            + (xp.linalg.norm(X_new - X, 'fro') / eta_X) ** 2
        )
        raw_gradient_norm = xp.sqrt(
            xp.linalg.norm(gradB, 'fro') ** 2
            + xp.linalg.norm(gradC, 'fro') ** 2
            + xp.linalg.norm(gradX, 'fro') ** 2
        )
        stationarity_ratio = float(gradient_mapping_norm / max(1.0, float(raw_gradient_norm)))
        stationarity_values.append(stationarity_ratio)

        if scale_every and (ii % scale_every == 0) and (ii > 0):

            B_norm = float(xp.linalg.norm(B_new, 'fro'))

            if B_norm > 1e6 or B_norm < 1e-6: 

                B_new, B1_new, X_new, Y1 = scale_normalize_BX(B_new, B1_new, X_new, Y1)

                if display:

                    print(f"Iter {ii}: Rescaled, B_norm was {B_norm:.2e}")

        loss_recon = 0.5 * float(xp.linalg.norm(A - B_new @ X_new @ C_new, 'fro') ** 2)

        if p == 0:

            reg_B = lambda_B * float(xp.count_nonzero(B1_new))

            reg_C = lambda_C * float(xp.count_nonzero(C1_new))

        elif p == 1:

            reg_B = lambda_B * float(xp.sum(xp.abs(B1_new)))

            reg_C = lambda_C * float(xp.sum(xp.abs(C1_new)))

        elif p == 2:

            reg_B = lambda_B * float(xp.sum(B1_new ** 2))

            reg_C = lambda_C * float(xp.sum(C1_new ** 2))

        else:

            reg_B = lambda_B * float(xp.sum(xp.abs(B1_new) ** p))

            reg_C = lambda_C * float(xp.sum(xp.abs(C1_new) ** p))

        reg_X = 0.5 * lambda_X * float(xp.sum(X_new ** 2))
        obj_val = loss_recon + reg_B + reg_C + reg_X
        obj_raw.append(obj_val)

        aug_penalty = rho_old / 2 * (

                float(xp.linalg.norm(B_new - B1_new, 'fro') ** 2) +

                float(xp.linalg.norm(C_new - C1_new, 'fro') ** 2)

        )

        obj.append(obj_val + aug_penalty)

        nvar = B_new.size + C_new.size

        epsilon_pri = np.sqrt(nvar) * epsilon_abs + epsilon_rel * max(

            float(xp.linalg.norm(B_new, 'fro')), float(xp.linalg.norm(B1_new, 'fro')),

            float(xp.linalg.norm(C_new, 'fro')), float(xp.linalg.norm(C1_new, 'fro')))

        dual_norm = rho_old * np.sqrt(float(xp.linalg.norm(Y1, 'fro') ** 2 + xp.linalg.norm(Y2, 'fro') ** 2))

        epsilon_dual = np.sqrt(nvar) * epsilon_abs + epsilon_rel * dual_norm

        r_norm = float(xp.linalg.norm(B_new - B1_new, 'fro') + xp.linalg.norm(C_new - C1_new, 'fro'))

        s_norm = rho_old * float(xp.linalg.norm(B1_new - B1, 'fro') + xp.linalg.norm(C1_new - C1, 'fro'))

        r_norm_list.append(r_norm)

        s_norm_list.append(s_norm)

        completed_iterations = ii + 1
        can_stop = completed_iterations >= minIter

        stationarity_ok = (not require_stationarity) or stationarity_ratio <= stationarity_tol
        if can_stop and stationarity_ok and r_norm < epsilon_pri and s_norm < epsilon_dual:
            if display:

                print(f"Iter {ii}, obj={obj[-1]:.4f}, raw={obj_raw[-1]:.4f}, r={r_norm:.4f}, s={s_norm:.4f}")

                print("Converged by residuals.")

            B, B1, C, C1, X, Y1, Y2 = (B_new, B1_new, C_new, C1_new, X_new, Y1, Y2)
            stop_reason = 'residuals'
            converged = True
            break
        relative_objective_change = np.inf
        if ii > 0:
            relative_objective_change = abs(obj[-1] - obj[-2]) / max(1.0, abs(obj[-2]))
        if can_stop and stationarity_ok and ii > 0 and relative_objective_change <= tolerance_rel:
            if display:

                print(f"Iter {ii}, obj={obj[-1]:.4f}, raw={obj_raw[-1]:.4f}, r={r_norm:.4f}, s={s_norm:.4f}")

                print("Converged by objective stagnation.")

            B, B1, C, C1, X, Y1, Y2 = (B_new, B1_new, C_new, C1_new, X_new, Y1, Y2)
            stop_reason = 'relative_objective_stagnation'
            converged = True
            break

        if completed_iterations >= max(minIter, 50) and ii > 0 and obj_raw[-1] - obj_raw[-2] > 1e3:
            if display:

                print(f"Iter {ii}, obj={obj[-1]:.4f}, raw={obj_raw[-1]:.4f}, r={r_norm:.4f}, s={s_norm:.4f}")

                print("Raw objective diverged, early stop.")

            B, B1, C, C1, X, Y1, Y2 = (B_new, B1_new, C_new, C1_new, X_new, Y1, Y2)
            stop_reason = 'diverged'
            break

        if display and ii % 50 == 0:

            print(f"Iter {ii}, obj={obj[-1]:.4f}, raw={obj_raw[-1]:.4f}, r={r_norm:.4f}, s={s_norm:.4f}")

        Y1_new = Y1 + rho_old * (B_new - B1_new)

        Y2_new = Y2 + rho_old * (C_new - C1_new)

        B, B1, C, C1, X, Y1, Y2 = (B_new, B1_new, C_new, C1_new, X_new, Y1_new, Y2_new)

        rho_min = opts.get('rho_min', 1e-6)

        rho_max = opts.get('rho_max', 1e6)

        if r_norm > 10 * s_norm and r_norm > 1e-6:

            rho_star = min(rho_old * 2, rho_max)

        elif s_norm > 10 * r_norm and s_norm > 1e-6:

            rho_star = max(rho_old / 2, rho_min)

        else:

            rho_star = rho_old * 1.0

    if use_gpu:

        B = cp.asnumpy(B)

        X = cp.asnumpy(X)

        C = cp.asnumpy(C)

    run_info = {
        'iterations': len(obj_raw),
        'converged': converged,
        'stop_reason': stop_reason,
        'minIter': minIter,
        'maxIter': maxIter,
        'adaptive_step': adaptive_step,
        'require_stationarity': require_stationarity,
        'stationarity_tol': stationarity_tol,
        'final_stationarity_ratio': stationarity_values[-1],
        'min_stationarity_ratio': min(stationarity_values),
        'total_backtracks': sum(backtrack_counts),
        'max_backtracks_per_iteration': max(backtrack_counts),
        'backtrack_exhausted_blocks': backtrack_exhausted,
        'mask_conflict_fraction': mask_conflict_fraction,
        'mask_conflict_policy': mask_conflict_policy,
        'mask_conflict_action': mask_conflict_action,
        'eta_B_range': [min(eta_B_values), max(eta_B_values)],
        'eta_C_range': [min(eta_C_values), max(eta_C_values)],
        'eta_X_range': [min(eta_X_values), max(eta_X_values)],
    }
    result = (B, X, C, obj, obj_raw, r_norm_list, s_norm_list)
    if opts.get('return_info', False):
        return result + (run_info,)
    return result
