
import numpy as np


def _standardize_columns(Phi):
    # Phi: [B, F]
    mu = Phi.mean(axis=0, keepdims=True)
    Phi0 = Phi - mu
    # RMS per column (complex): sqrt(mean(|.|^2))
    scale = np.sqrt(np.mean(np.abs(Phi0)**2, axis=0, keepdims=True)) + 1e-12
    Phi_std = Phi0 / scale
    return Phi_std, mu, scale

def _standardize_vector(y):
    mu = y.mean()
    y0 = y - mu
    scale = np.sqrt(np.mean(np.abs(y0)**2)) + 1e-12
    y_std = y0 / scale
    return y_std, mu, scale

def _solve_ridge_complex_scaled(Phi, y, lam):
    """
    Standardize Phi (per-column) and y, solve ridge on the standardized system,
    then return (w_unscaled, pack) where pack holds scaling to apply at test-time.
    """
    Phi_std, mu_phi, sc_phi = _standardize_columns(Phi)     # [B,F]
    y_std,  mu_y,  sc_y  = _standardize_vector(y)           # [B]

    F = Phi_std.shape[1]
    A = Phi_std.conj().T @ Phi_std
    b = Phi_std.conj().T @ y_std
    A_reg = A + lam * np.eye(F, dtype=np.complex128)
    try:
        w_std = np.linalg.solve(A_reg, b)
    except np.linalg.LinAlgError:
        w_std, *_ = np.linalg.lstsq(A_reg, b, rcond=None)

    # Unscale weights so you can use them on raw Phi at inference:
    # y_std ≈ ( (Phi - mu_phi)/sc_phi ) @ w_std  ⇒
    # y_hat ≈ mu_y + sc_y * y_std = mu_y + sc_y * ((Phi - mu_phi)/sc_phi) @ w_std
    # Define w_unscaled and bias:
    w_unscaled = (sc_y * w_std) / sc_phi.flatten()
    bias = mu_y - (mu_phi @ w_unscaled).item()

    pack = dict(mu_phi=mu_phi, sc_phi=sc_phi, mu_y=mu_y, sc_y=sc_y, bias=bias)
    return w_unscaled, bias, pack

def _predict_with_pack(Phi, w_unscaled, bias):
    # Phi: raw features [B,F]
    return Phi @ w_unscaled + bias


def complex_power_std(x: np.ndarray):
    """
    Return RMS magnitude std for a complex array: sqrt(mean(|x|^2)).
    If x is real, this is the usual RMS.
    """
    return np.sqrt(np.mean(np.abs(x)**2))

def standardize_complex(x: np.ndarray):
    """
    Standardize a complex array to zero-mean and unit RMS magnitude.
    Returns standardized array, original mean, and original RMS.
    """
    mu = np.mean(x)
    x0 = x - mu
    s = complex_power_std(x0)
    if s == 0:
        return x0, mu, 1.0
    return x0 / s, mu, s

def impose_crosslink_correlation(H1: np.ndarray, H2: np.ndarray, rho_target: float, rng=None):
    """
    Given two complex arrays H1 and H2 (same shape), construct H2' such that
    corr(H1, H2') ≈ rho_target in the complex correlation sense:
        corr = E[ H1 * conj(H2') ] / sqrt( E|H1|^2 * E|H2'|^2 )
    We leave H1 unchanged and modify H2.
    """
    if rng is None:
        rng = np.random.default_rng(1234)

    assert H1.shape == H2.shape, "H1 and H2 must have the same shape"
    shape = H1.shape

    # Flatten for operations, then reshape back at the end
    x = H1.reshape(-1).astype(np.complex128, copy=False)
    y = H2.reshape(-1).astype(np.complex128, copy=False)

    # Standardize both to zero mean, unit RMS (over *all* TF/antenna samples)
    xs, mu_x, s_x = standardize_complex(x)
    ys, mu_y, s_y = standardize_complex(y)

    # Ensure ys is not perfectly collinear with xs (in rare edge cases)
    # We'll optionally mix in some fresh complex noise for numerical stability.
    if np.isclose(complex_power_std(ys - xs), 0.0):
        noise = (rng.standard_normal(xs.size) + 1j * rng.standard_normal(xs.size)) / np.sqrt(2.0)
        ys = noise.astype(np.complex128)

    # Construct new standardized y':  y' = rho * x + sqrt(1 - rho^2) * e,
    # where e is ys whitened to be uncorrelated with x (approximately).
    # For large samples, taking e=ys is sufficient; optionally orthogonalize.
    rho = np.clip(rho_target, -0.999, 0.999)

    # Gram-Schmidt: make e ≈ orthogonal to x (in the complex inner-product sense)
    # <x,e> = mean( x * conj(e) ).
    inner_xy = np.mean(xs * np.conj(ys))
    e = ys - inner_xy * xs  # remove x-component
    # Normalize e to unit RMS
    e_std = complex_power_std(e)
    if e_std < 1e-8:
        # regenerate random orthogonal-ish noise if needed
        noise = (rng.standard_normal(xs.size) + 1j * rng.standard_normal(xs.size)) / np.sqrt(2.0)
        e = noise - np.mean(noise)
        e = e / complex_power_std(e)
    else:
        e = e / e_std

    y_new_std = rho * xs + np.sqrt(max(0.0, 1.0 - rho**2)) * e

    # Rescale to original H2 power and re-add original mean (to preserve marginal stats of H2)
    y_new = y_new_std * s_y + mu_y

    # Reshape back
    H2_new = y_new.reshape(shape)

    return H2_new

def complex_corr_coeff(x: np.ndarray, y: np.ndarray):
    """
    Empirical complex correlation coefficient:
      corr = E[ x * conj(y) ] / sqrt( E|x|^2 * E|y|^2 )
    """
    x = x.reshape(-1).astype(np.complex128, copy=False)
    y = y.reshape(-1).astype(np.complex128, copy=False)
    num = np.mean(x * np.conj(y))
    den = np.sqrt(np.mean(np.abs(x)**2) * np.mean(np.abs(y)**2))
    if den == 0:
        return 0.0 + 0.0j
    return num / den

def _nmse(y_hat, y_true, eps=1e-12):
    mse = np.mean(np.abs(y_hat - y_true)**2)
    pwr = np.mean(np.abs(y_true)**2)
    return mse / (pwr + eps), mse, pwr

def wiener_predict_TEST_last_subframe_from_both_links(h_freq_csi_history_list_corr, lam=1e-4, target_link=1):
    H0, H1 = h_freq_csi_history_list_corr
    assert H0.shape == H1.shape
    S, N_r, N_t, Nsym, Nsc = H0.shape
    assert S >= 3

    B = N_r * N_t * Nsym * Nsc
    H0r = H0.reshape(S, B).astype(np.complex128, copy=False)
    H1r = H1.reshape(S, B).astype(np.complex128, copy=False)

    # Choose which link to predict
    Hr_tar = H0r if target_link == 0 else H1r

    # Build train features (both links, times 0..S-3)
    F = 2 * (S - 2)
    Phi_train = np.empty((B, F), dtype=np.complex128)
    col = 0
    for s in range(0, S-2):
        Phi_train[:, col] = H0r[s, :]; col += 1
        Phi_train[:, col] = H1r[s, :]; col += 1
    y_train = Hr_tar[S-2, :]

    w, bias, _ = _solve_ridge_complex_scaled(Phi_train, y_train, lam)
    y_train_hat = _predict_with_pack(Phi_train, w, bias)
    nmse_train, mse_train, pwr_train = _nmse(y_train_hat, y_train)

    # Test features (both links, times 1..S-2)
    Phi_test = np.empty((B, F), dtype=np.complex128)
    col = 0
    for s in range(1, S-1):
        Phi_test[:, col] = H0r[s, :]; col += 1
        Phi_test[:, col] = H1r[s, :]; col += 1
    y_test = Hr_tar[S-1, :]

    y_test_hat = _predict_with_pack(Phi_test, w, bias)
    nmse_test, mse_test, pwr_test = _nmse(y_test_hat, y_test)

    y_test_hat_tensor = y_test_hat.reshape(N_r, N_t, Nsym, Nsc)
    return (
        y_test_hat_tensor,
        w,
        dict(
            mode="both_links",
            target_link=target_link,
            F=F, B=B, S=S, shape_last=(N_r, N_t, Nsym, Nsc),
            nmse_test=nmse_test, mse_test=mse_test, pwr_test=pwr_test,
            nmse_train=nmse_train, mse_train=mse_train, pwr_train=pwr_train,
            lam=lam
        )
    )

def wiener_predict_TEST_last_subframe_single_link(h_freq_csi_history_list_corr, lam=1e-4, link_index=0):
    """
    Train on times {1..S-2} --> target {S-1} for the chosen link,
    then apply learned w to {2..S-1} to predict {S}. Returns TEST NMSE.
    """
    H = h_freq_csi_history_list_corr[link_index]
    S, N_r, N_t, Nsym, Nsc = H.shape
    assert S >= 3, "Need S >= 3 for a non-empty train set and a test step."

    B = N_r * N_t * Nsym * Nsc
    Hr = H.reshape(S, B).astype(np.complex128, copy=False)

    # -------- Train: features {0..S-3} -> target {S-2}
    F = (S - 2)
    Phi_train = Hr[0:S-2, :].T          # [B, S-2]
    y_train = Hr[S-2, :]                # target at time S-1 (1-based)

    w, bias, _ = _solve_ridge_complex_scaled(Phi_train, y_train, lam)
    y_train_hat = _predict_with_pack(Phi_train, w, bias)
    nmse_train, mse_train, pwr_train = _nmse(y_train_hat, y_train)

    # -------- Test: features {1..S-2} -> target {S-1}
    Phi_test = Hr[1:S-1, :].T           # [B, S-2]
    y_test = Hr[S-1, :]

    y_test_hat = _predict_with_pack(Phi_test, w, bias)
    nmse_test, mse_test, pwr_test = _nmse(y_test_hat, y_test)

    y_test_hat_tensor = y_test_hat.reshape(N_r, N_t, Nsym, Nsc)

    return (
        y_test_hat_tensor,
        w,
        dict(
            mode="single_link",
            link_index=link_index,
            F=F, B=B, S=S, shape_last=(N_r, N_t, Nsym, Nsc),
            nmse_test=nmse_test, mse_test=mse_test, pwr_test=pwr_test,
            nmse_train=nmse_train, mse_train=mse_train, pwr_train=pwr_train,
            lam=lam
        )
    )

def pretty_print_wiener_info(info, title="Result"):
    mode = info.get("mode", "N/A")
    link = info.get("link_index", "-")
    nmse_t, nmse_tr = info["nmse_test"], info["nmse_train"]
    S, F, lam = info["S"], info["F"], info["lam"]
    N_r, N_t, Nsym, Nsc = info["shape_last"]

    print(f"\n[{title}]")
    print(f"  Mode: {mode} (link={link}) | λ={lam:.1e} | S={S}, F={F}")
    print(f"  Test NMSE={nmse_t:.3e}, Train NMSE={nmse_tr:.3e}")
    print(f"  Output shape: ({N_r}×{N_t}×{Nsym}×{Nsc})")


def pairwise_complex_correlations(h_freq_csi_history_list, by_abs=True):
    """
    Compute empirical complex correlation coefficients for all pairs (i,j)
    in h_freq_csi_history_list that have the *same shape*.
    
    corr(x,y) = E[x * conj(y)] / sqrt(E|x|^2 * E|y|^2), computed over all entries.
    
    Args:
        h_freq_csi_history_list: list of np.ndarray (complex or real).
        by_abs: if True, sort by |corr| descending; else by real(corr) descending.
    
    Returns:
        results: list of dicts sorted from highest to lowest with keys:
            {
              "i": int,           # index of first element
              "j": int,           # index of second element
              "corr": complex,    # complex correlation coefficient
              "abs_corr": float,  # magnitude of corr
              "shape": tuple      # common shape used
            }
    """
    def _corr(x, y, eps=1e-12):
        x = x.reshape(-1).astype(np.complex128, copy=False)
        y = y.reshape(-1).astype(np.complex128, copy=False)
        num = np.mean(x * np.conj(y))
        den = np.sqrt(np.mean(np.abs(x)**2) * np.mean(np.abs(y)**2)) + eps
        return num / den

    n = len(h_freq_csi_history_list)
    results = []
    for i in range(n):
        Xi = h_freq_csi_history_list[i]
        if Xi is None:
            continue
        for j in range(i+1, n):
            Xj = h_freq_csi_history_list[j]
            if Xj is None:
                continue
            if Xi.shape != Xj.shape:
                continue  # only compare arrays with identical shape
            c = _corr(Xi, Xj)
            results.append({
                "i": i,
                "j": j,
                "corr": c,
                "abs_corr": float(np.abs(c)),
                "shape": Xi.shape
            })
    # Sort
    if by_abs:
        results.sort(key=lambda d: d["abs_corr"], reverse=True)
    else:
        results.sort(key=lambda d: d["corr"].real, reverse=True)
    return results


def pairwise_lagged_correlations(h_freq_csi_history_list, max_lag=1, by_abs=True):
    """
    Compute lagged cross-correlation between all pairs of CSI histories
    (with the same shape), averaged over antennas/subcarriers.
    
    Args:
        h_freq_csi_history_list: list of CSI tensors, each [S, ...] complex
        max_lag: maximum time lag to test (default: 1)
        by_abs: if True, sort by absolute correlation magnitude

    Returns:
        List of dicts sorted by descending |corr| (or real corr if by_abs=False):
          [
            {
              'i': int,               # index of first link
              'j': int,               # index of second link
              'lag': int,             # lag (positive: i leads j)
              'corr': complex,        # complex correlation at that lag
              'abs_corr': float,      # |corr|
              'shape': tuple          # shape of compared tensors
            },
            ...
          ]
    """
    n = len(h_freq_csi_history_list)
    results = []
    for i in range(n):
        Hi = h_freq_csi_history_list[i]
        if Hi is None: continue
        for j in range(i+1, n):
            Hj = h_freq_csi_history_list[j]
            if Hj is None or Hi.shape != Hj.shape:
                continue

            S = Hi.shape[0]
            if S <= max_lag:
                continue

            # Flatten spatial/freq dims
            Xi = Hi.reshape(S, -1).astype(np.complex128)
            Xj = Hj.reshape(S, -1).astype(np.complex128)

            for lag in range(1, max_lag + 1):
                # i leads j: correlate Xi[t] with Xj[t+lag]
                A = Xi[:-lag, :]
                B = Xj[lag:, :]

                num = np.mean(A * np.conj(B))
                den = np.sqrt(np.mean(np.abs(A)**2) * np.mean(np.abs(B)**2)) + 1e-12
                corr = num / den

                results.append(dict(
                    i=i, j=j, lag=lag,
                    corr=corr,
                    abs_corr=np.abs(corr),
                    shape=Hi.shape
                ))

    # Sort results
    key = (lambda r: r['abs_corr']) if by_abs else (lambda r: np.real(r['corr']))
    results.sort(key=key, reverse=True)
    return results

def evaluate_pairwise_wiener_candidates(h_freq_csi_history_list, lam=0.0, max_pairs=None):
    """
    Rank link pairs for the 'both-links' Wiener predictor using the *same* protocol as
    wiener_predict_TEST_last_subframe_from_both_links:
      - Train:  features from times {0..S-3}  -> target {S-2}
      - Test:   features from times {1..S-2}  -> target {S-1}
    For each pair (i,j) with identical shapes, evaluates *both* directions:
      - target_link = i (use i & j as features to predict i)
      - target_link = j (use i & j as features to predict j)

    Args:
        h_freq_csi_history_list: list of CSI tensors, each shaped [S, N_r, N_t, Nsym, Nsc] (complex).
        lam: ridge regularization (use 0.0 for OLS).
        max_pairs: optional int to stop after this many best-scoring results (post-sort).

    Returns:
        results: list of dicts sorted by ascending test NMSE. Each dict has:
            {
              "i": int, "j": int,                 # pair indices (i<j)
              "target_link": int,                 # which link is predicted: i or j
              "nmse_test": float, "nmse_train": float,
              "mse_test": float,  "mse_train": float,
              "pwr_test": float,  "pwr_train": float,
              "S": int, "F": int, "B": int,       # meta info
              "shape": tuple, "lam": float
            }
    """
    results = []
    n = len(h_freq_csi_history_list)

    for i in range(n):
        Hi = h_freq_csi_history_list[i]
        if Hi is None:
            continue
        for j in range(i+1, n):
            Hj = h_freq_csi_history_list[j]
            if Hj is None or Hi.shape != Hj.shape:
                continue

            # Shapes and basic checks
            S, N_r, N_t, Nsym, Nsc = Hi.shape
            if S < 3:
                continue
            B = N_r * N_t * Nsym * Nsc
            F = 2 * (S - 2)

            # Flatten time vs all-other dims
            Hir = Hi.reshape(S, B).astype(np.complex128, copy=False)
            Hjr = Hj.reshape(S, B).astype(np.complex128, copy=False)

            # Build training features (both links, times 0..S-3)
            Phi_train = np.empty((B, F), dtype=np.complex128)
            col = 0
            for s in range(0, S-2):
                Phi_train[:, col] = Hir[s, :]; col += 1
                Phi_train[:, col] = Hjr[s, :]; col += 1

            # Build test features (both links, times 1..S-2)
            Phi_test = np.empty((B, F), dtype=np.complex128)
            col = 0
            for s in range(1, S-1):
                Phi_test[:, col] = Hir[s, :]; col += 1
                Phi_test[:, col] = Hjr[s, :]; col += 1

            # Evaluate both target directions: predict Hi and predict Hj
            for target_link, Hr in ((i, Hir), (j, Hjr)):
                y_train = Hr[S-2, :]
                y_test  = Hr[S-1, :]

                # Solve on TRAIN using your standardized ridge
                w, bias, _ = _solve_ridge_complex_scaled(Phi_train, y_train, lam)

                # Predictions
                y_hat_train = _predict_with_pack(Phi_train, w, bias)
                y_hat_test  = _predict_with_pack(Phi_test,  w, bias)

                # Metrics (same NMSE you use)
                nmse_train, mse_train, pwr_train = _nmse(y_hat_train, y_train)
                nmse_test,  mse_test,  pwr_test  = _nmse(y_hat_test,  y_test)

                results.append(dict(
                    i=i, j=j, target_link=target_link,
                    nmse_test=float(nmse_test),  mse_test=float(mse_test),  pwr_test=float(pwr_test),
                    nmse_train=float(nmse_train), mse_train=float(mse_train), pwr_train=float(pwr_train),
                    S=S, F=F, B=B, shape=Hi.shape, lam=float(lam)
                ))

    # Rank by *lowest* test NMSE
    results.sort(key=lambda d: d["nmse_test"])
    if max_pairs is not None:
        results = results[:max_pairs]
    return results
