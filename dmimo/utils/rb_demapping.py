import tensorflow as tf
import numpy as np

# simple cache so we don't rebuild A and the factorization every call
_RB_LS_CACHE = {}

def _get_rb_ls_factors(Nsc, NRB, rb_size=12, lam=1e-3, dtype=tf.complex64):
    """
    Build and cache:
      A : [NRB, Nsc] RB-averaging operator (last RB gets the remainder)
      L : lower Cholesky of M = A^H A + lam D^H D  (D = 2nd-diff smoothness)
    """
    key = (int(Nsc), int(NRB), int(rb_size), float(lam), tf.as_dtype(dtype).name)
    if key in _RB_LS_CACHE:
        return _RB_LS_CACHE[key]

    # --- Build A (dense) ---
    sizes = [rb_size]*(NRB-1) + [Nsc - rb_size*(NRB-1)]  # e.g., [12,12,...,8]
    A = np.zeros((NRB, Nsc), dtype=np.float32)
    s = 0
    for i, k in enumerate(sizes):
        A[i, s:s+k] = 1.0/float(k)
        s += k
    A = tf.constant(A, dtype=tf.as_dtype(dtype).real_dtype)  # real weights
    A = tf.cast(A, dtype)  # promote to complex for A^H A

    # --- Build D (second-difference) ---
    D = np.zeros((Nsc-2, Nsc), dtype=np.float32)
    for i in range(Nsc-2):
        D[i, i]   = 1.0
        D[i, i+1] = -2.0
        D[i, i+2] = 1.0
    D = tf.constant(D, dtype=tf.as_dtype(dtype).real_dtype)
    D = tf.cast(D, dtype)

    # --- Normal equations matrix and its Cholesky ---
    AhA = tf.matmul(tf.linalg.adjoint(A), A)            # [Nsc,Nsc]
    Ls  = tf.matmul(tf.linalg.adjoint(D), D)            # [Nsc,Nsc]
    M   = AhA + tf.cast(lam, dtype) * Ls                # Hermitian PD
    L   = tf.linalg.cholesky(M)                         # cache this

    _RB_LS_CACHE[key] = (A, L)
    return A, L

def rb_to_sc_ls(r, Nsc=512, rb_size=12, lam=1e-3, dtype=None):
    """
    Least-squares reconstruction:
        min_h ||A h - r||^2 + lam ||D h||^2
    where A averages 12 (last 8) tones per RB.
    r: [..., NRB] complex (RB values); output: [..., Nsc] complex.
    """
    x = tf.convert_to_tensor(r)
    if dtype is None:
        dtype = x.dtype
    x = tf.cast(x, dtype)

    # infer NRB from the last dim (must be known at runtime)
    NRB = int(x.shape[-1])
    A, L = _get_rb_ls_factors(Nsc=Nsc, NRB=NRB, rb_size=rb_size, lam=lam, dtype=dtype)

    # reshape to batch of vectors and solve
    lead = tf.shape(x)[:-1]
    R = tf.reshape(x, [-1, NRB, 1])                      # [B, NRB, 1]
    rhs = tf.matmul(tf.linalg.adjoint(A), R)             # [B, Nsc, 1]
    H   = tf.linalg.cholesky_solve(L, rhs)               # [B, Nsc, 1]
    H   = tf.reshape(tf.squeeze(H, -1), tf.concat([lead, [Nsc]], axis=0))
    return H