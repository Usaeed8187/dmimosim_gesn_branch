
import numpy as np

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


def wiener_predict_last_subframe_from_both_links(h_freq_csi_history_list_corr, lam=1e-4):
    H0, H1 = h_freq_csi_history_list_corr
    assert H0.shape == H1.shape, "H0 and H1 must have identical shapes"
    S, N_r, N_t, Nsym, Nsc = H0.shape

    assert S >= 2, "Need at least 2 subframes to leave one out as target."

    # Number of features: 2 links * (S-1) past subframes
    F = 2*(S-1)

    # Reshape to [S, B] where B = N_r*N_t*Nsym*Nsc
    B = N_r * N_t * Nsym * Nsc
    H0_reshaped = H0.reshape(S, B).astype(np.complex128, copy=False)
    H1_reshaped = H1.reshape(S, B).astype(np.complex128, copy=False)

    # Target: last subframe of link 2 (index 1)
    y = H1_reshaped[-1, :]  # [B]

    # Features: [H0[0], H1[0], H0[1], H1[1], ..., H0[S-2], H1[S-2]]
    Phi = np.empty((B, F), dtype=np.complex128)
    col = 0
    for s in range(S-1):
        Phi[:, col]   = H0_reshaped[s, :]; col += 1
        Phi[:, col]   = H1_reshaped[s, :]; col += 1

    # Solve complex ridge regression: w = (Phi^H Phi + lam*I)^-1 Phi^H y
    A = Phi.conj().T @ Phi  # [F, F]
    b = Phi.conj().T @ y    # [F]
    A_reg = A + lam * np.eye(F, dtype=np.complex128)
    w = np.linalg.solve(A_reg, b)  # [F]

    y_hat = Phi @ w  # [B]

    y_hat_tensor = y_hat.reshape(N_r, N_t, Nsym, Nsc)
    mse = np.mean(np.abs(y_hat - y)**2)
    nmse = mse / (np.mean(np.abs(y)**2) + 1e-12)
    return y_hat_tensor, w, dict(mse=mse, nmse=nmse, F=F, B=B, S=S, shape_last=(N_r, N_t, Nsym, Nsc))


def wiener_predict_last_subframe_single_link(h_freq_csi_history_list_corr, lam=1e-4, link_index=0):
    """
    Predict the last subframe of the specified link using ONLY its own prior subframes.
    
    Args:
        h_freq_csi_history_list_corr: list [H0, H1], each of shape 
            [S, N_r, N_t, Nsym, Nsc], complex.
        lam: ridge regularization scalar (>=0).
        link_index: which link to predict (0 or 1). Default: 0.
        
    Returns:
        y_hat_tensor: predicted last subframe tensor, shape [N_r, N_t, Nsym, Nsc]
        w: complex filter of length (S-1)
        info: dict with diagnostics (mse, nmse, S, B, etc.)
    """
    H = h_freq_csi_history_list_corr[link_index]
    S, N_r, N_t, Nsym, Nsc = H.shape
    assert S >= 2, "Need at least 2 subframes to leave the last one as target."

    # Flatten per subframe to [B], with B = N_r*N_t*Nsym*Nsc
    B = N_r * N_t * Nsym * Nsc
    H_flat = H.reshape(S, B).astype(np.complex128, copy=False)

    # Target: last subframe of the chosen link
    y = H_flat[-1, :]  # [B]

    # Features: all *previous* subframes of the same link
    F = (S - 1)
    Phi = H_flat[:-1, :].T  # [B, F]

    # Solve complex ridge regression
    A = Phi.conj().T @ Phi  # [F, F]
    b = Phi.conj().T @ y    # [F]
    A_reg = A + lam * np.eye(F, dtype=np.complex128)
    w = np.linalg.solve(A_reg, b)  # [F]

    # Predict and reshape
    y_hat = Phi @ w  # [B]
    y_hat_tensor = y_hat.reshape(N_r, N_t, Nsym, Nsc)

    mse = np.mean(np.abs(y_hat - y)**2)
    nmse = mse / (np.mean(np.abs(y)**2) + 1e-12)

    return y_hat_tensor, w, dict(mse=mse, nmse=nmse, F=F, B=B, S=S, shape_last=(N_r, N_t, Nsym, Nsc))