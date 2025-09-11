"""Utilities and multi-mode ESN reservoir for RB×Tx×Rx channel tensors."""

import tensorflow as tf
import math

def _stat_dict(x: tf.Tensor, name: str):
    x = tf.cast(x, tf.complex64) if x.dtype.is_complex else tf.cast(x, tf.float32)
    r = tf.math.real(x)
    i = tf.math.imag(x) if x.dtype.is_complex else tf.zeros_like(r)
    absx = tf.abs(x)
    return {
        "name": name,
        "shape": tuple(x.shape),
        "mean_r": float(tf.reduce_mean(r).numpy()),
        "mean_i": float(tf.reduce_mean(i).numpy()),
        "std_abs": float(tf.math.reduce_std(absx).numpy()),
        "rms": float(tf.sqrt(tf.reduce_mean(absx**2)).numpy()),
        "max_abs": float(tf.reduce_max(absx).numpy())
    }

def _print_stats(*pairs, prefix="DBG"):
    # pairs: (tensor, "label")
    rows = []
    for t, label in pairs:
        try:
            s = _stat_dict(t, label)
            rows.append(
                f"{prefix} [{s['name']}]{s['shape']}: mean(r)={s['mean_r']:.3e} "
                f"mean(i)={s['mean_i']:.3e} std|·|={s['std_abs']:.3e} "
                f"rms={s['rms']:.3e} max|·|={s['max_abs']:.3e}"
            )
        except Exception as e:
            rows.append(f"{prefix} [{label}]: <stat error: {e}>")
    print("\n".join(rows))

def _assert_finite(x, tag):
    tf.debugging.assert_all_finite(tf.math.real(x), f"{tag}: real part non-finite")
    if x.dtype.is_complex:
        tf.debugging.assert_all_finite(tf.math.imag(x), f"{tag}: imag part non-finite")

def _is_hermitian(M, tol=1e-5):
    Mh = tf.transpose(M, conjugate=True)
    diff = tf.linalg.norm(M - Mh) / (tf.linalg.norm(M) + 1e-30)
    return float(diff.numpy()) < tol

def _safe_symmetrize(M):
    return 0.5 * (M + tf.transpose(M, conjugate=True))

def _cond_number_gram(M):
    """
    Robust condition number for a Gram matrix (Hermitian PSD).
    - Symmetrize to kill tiny asymmetry
    - Add tiny jitter based on average diagonal to avoid zero/neg eigenvalues
    - Compute on CPU to dodge GPU heevd quirks
    - Fallback to SVD if eigvalsh still fails
    """
    M = tf.cast(M, tf.complex64)

    # Ensure finiteness before anything else
    _assert_finite(tf.math.real(M), "cond(A): real part non-finite")
    _assert_finite(tf.math.imag(M), "cond(A): imag part non-finite")

    # Force Hermitian numerically
    M = _safe_symmetrize(M)

    # Jitter scaled to average diagonal magnitude
    diag = tf.math.real(tf.linalg.diag_part(M))
    mean_diag = tf.reduce_mean(diag)
    # ensure at least O(1e-7) even if mean_diag is tiny
    jitter = tf.cast(tf.maximum(1e-7, 1e-7 * tf.maximum(mean_diag, 1.0)), tf.float32)
    I = tf.eye(tf.shape(M)[0], dtype=tf.complex64)
    M = M + tf.cast(jitter, tf.complex64) * I

    # Try CPU eig first (GPU sometimes fails to converge for complex64)
    try:
        with tf.device("/CPU:0"):
            ev = tf.linalg.eigvalsh(M)
            ev = tf.math.real(ev)
    except Exception as e:
        print(f"DBG cond(): eigvalsh CPU path failed ({e}); falling back to SVD")
        # SVD fallback: for PSD, singular values = eigenvalues
        # SVD works for non-Hermitian too, so it’s a safe fallback.
        with tf.device("/CPU:0"):
            s = tf.linalg.svd(M, compute_uv=False)
            ev = tf.cast(s, tf.float32)

    # Guard against non-positive tiny values
    ev = tf.where(ev <= 0.0, tf.constant(1e-30, ev.dtype), ev)
    c = float((tf.reduce_max(ev) / tf.reduce_min(ev)).numpy())

    # Optional debug prints (uncomment if needed)
    print(f"DBG cond(): hermitian?={_is_hermitian(M)}, min(ev)={float(tf.reduce_min(ev).numpy()):.3e}, max(ev)={float(tf.reduce_max(ev).numpy()):.3e}, cond≈{c:.3e}, jitter={float(jitter.numpy()):.1e}")

    return c


def _tanh_saturation_fraction(z):
    # crude: fraction of elements with |real|>2 or |imag|>2 before tanh
    r = tf.math.real(z)
    i = tf.math.imag(z) if z.dtype.is_complex else tf.zeros_like(r)
    n = tf.cast(tf.size(r), tf.float32)
    sat = tf.reduce_mean(tf.cast((tf.abs(r) > 2.0) | (tf.abs(i) > 2.0), tf.float32))
    return float((sat).numpy())


def _rank(tensor: tf.Tensor) -> int:
    """Return the (possibly dynamic) rank of ``tensor`` as an ``int``."""
    rank = tensor.shape.rank
    if rank is None:
        rank = int(tf.rank(tensor))
    return rank


def mode_n_product(tensor, matrix, mode):
    """Return the mode-``mode`` product of a tensor and matrix.

    Parameters
    ----------
    tensor : tf.Tensor
        Input tensor of arbitrary shape.
    matrix : tf.Tensor
        Matrix to multiply with shape ``[new_dim, old_dim]`` where ``old_dim``
        equals the size of ``tensor`` along ``mode``.
    mode : int
        Zero-based mode index along which multiplication is applied.

    Returns
    -------
    tf.Tensor
        Tensor with the ``mode`` dimension replaced by ``new_dim``.
    """
    tensor = tf.convert_to_tensor(tensor)
    matrix = tf.convert_to_tensor(matrix)
    res = tf.tensordot(tensor, matrix, axes=[[mode], [1]])
    res_rank = _rank(res)
    perm = list(range(res_rank))
    perm = perm[:mode] + [res_rank - 1] + perm[mode:-1]
    return tf.transpose(res, perm)


def unfold(tensor, mode):
    """Return the mode-``mode`` matricization of ``tensor``.

    Parameters
    ----------
    tensor : tf.Tensor
        Input tensor of shape ``[d1, d2, ..., dn]``.
    mode : int
        Zero-based mode index to place as the row dimension.

    Returns
    -------
    tf.Tensor
        Matrix of shape ``[d_mode, prod(other_dims)]``.
    """
    tensor = tf.convert_to_tensor(tensor)
    rank = _rank(tensor)
    perm = [mode] + [i for i in range(rank) if i != mode]
    trans = tf.transpose(tensor, perm)
    shape = tf.shape(trans)
    return tf.reshape(trans, [shape[0], -1])


def kron(a, b):
    """Kronecker product of two matrices ``a`` and ``b``.

    Both inputs must have shape ``[m, n]`` and ``[p, q]`` respectively.
    The output has shape ``[m*p, n*q]``.
    """
    a = tf.convert_to_tensor(a)
    b = tf.convert_to_tensor(b)
    a_shape = tf.shape(a)
    b_shape = tf.shape(b)
    return tf.reshape(tf.tensordot(a, b, axes=0),
                      [a_shape[0] * b_shape[0], a_shape[1] * b_shape[1]])


class multimode_esn_pred:
    """Reservoir computer operating on 3‑mode channel tensors.

    The reservoir state ``S`` has dimensions ``[d_f, d_t, d_r]`` corresponding to
    reduced representations of resource blocks (RBs), transmit antennas, and
    receive antennas.  Inputs ``Y`` are tensors with dimensions
    ``[N_f, N_t, N_r]``.
    """

    def __init__(self,
                 N_f,  # number of RBs / subcarriers
                 N_t,  # number of transmit antennas
                 N_r,  # number of receive antennas
                 d_f,  # RB state dimension
                 d_t,  # Tx state dimension
                 d_r,  # Rx state dimension
                 window_len=1,  # optional window length along RB mode
                 sigma=None,  # elementwise non-linearity
                 alpha=1.0,  # leaky integration factor
                 dtype=tf.complex64,
                 target_rho=0.8,
                 debug=False, safe_solve=False):
        """Initialize random reservoir and input coupling matrices.

        Parameters correspond to the tensor sizes described above.  Weight
        matrices ``A_*`` (recurrent) and ``U_*`` (input) follow the dimensions:

        - ``A_f`` : ``[d_f, d_f]``
        - ``A_t`` : ``[d_t, d_t]``
        - ``A_r`` : ``[d_r, d_r]``
        - ``U_f`` : ``[d_f, L*N_f]``
        - ``U_t`` : ``[d_t, N_t]``
        - ``U_r`` : ``[d_r, N_r]``
        """

        if sigma is None:
            def sigma(x):
                return tf.complex(tf.tanh(tf.math.real(x)), tf.tanh(tf.math.imag(x)))

        # Store tensor dimensions
        self.N_f = N_f
        self.N_t = N_t
        self.N_r = N_r
        # Store state dimensions
        self.d_f = d_f
        self.d_t = d_t
        self.d_r = d_r
        # Input window length and activation
        self.window_len = window_len
        self.sigma = sigma
        self.alpha = alpha
        self.dtype = dtype
        self.target_rho = target_rho

        # --- Feature-queue dimensions (state + raw/windowed input) ---
        # Y_tilde has shape [L*N_f, N_t, N_r]; S has [d_f, d_t, d_r]
        self.df_feat = d_f + window_len * N_f
        self.dt_feat = d_t + N_t
        self.dr_feat = d_r + N_r

        # Recurrent transformation matrices along each mode, scaled below unity
        self.A_f = self._init_matrix(d_f, d_f)
        self.A_t = self._init_matrix(d_t, d_t)
        self.A_r = self._init_matrix(d_r, d_r)

        # Input coupling matrices. ``U_f`` handles optional RB windowing.
        self.U_f = self._rand_matrix(d_f, N_f * window_len)
        self.U_t = self._rand_matrix(d_t, N_t)
        self.U_r = self._rand_matrix(d_r, N_r)

        # Optional: normalize input gains to keep injection energy modest
        # Comment this block out if you don't want any change in behavior.
        uf_scale = 1.0 / math.sqrt(max(1, N_f * window_len))
        ut_scale = 1.0 / math.sqrt(max(1, N_t))
        ur_scale = 1.0 / math.sqrt(max(1, N_r))
        self.U_f = tf.cast(self.U_f, self.dtype) * tf.cast(uf_scale, self.dtype)
        self.U_t = tf.cast(self.U_t, self.dtype) * tf.cast(ut_scale, self.dtype)
        self.U_r = tf.cast(self.U_r, self.dtype) * tf.cast(ur_scale, self.dtype)

        self.debug = debug
        self.safe_solve = safe_solve  # switch to Cholesky+jitter path if True

        if self.debug:
            # Report "spectral radius" actually achieved for A_* and norms of U_*
            for name, A in [("A_f", self.A_f), ("A_t", self.A_t), ("A_r", self.A_r)]:
                vals = tf.linalg.eigvals(A)
                rho = float(tf.reduce_max(tf.abs(vals)).numpy())
                print(f"DBG init {name}: spectral radius ~ {rho:.4f}")
            for name, U in [("U_f", self.U_f), ("U_t", self.U_t), ("U_r", self.U_r)]:
                fro = float(tf.linalg.norm(U).numpy())
                print(f"DBG init {name}: ‖·‖_F = {fro:.3e} shape={U.shape}")


        self.reset_state()

    def build_feature_queue(self, S, Y_tilde):
        """
        Construct feature tensor concatenating state and input per mode.

        Instead of a block-diagonal layout, the state ``S`` and the windowed
        input ``Y_tilde`` are concatenated along each mode individually.  The
        resulting tensor has dimensions ``[d_f + L*N_f, d_t + N_t, d_r + N_r]``
        and contains overlap regions ensuring state–input cross terms exist.
        """
        S = tf.convert_to_tensor(S, dtype=self.dtype)               # [d_f, d_t, d_r]
        Y_tilde = tf.convert_to_tensor(Y_tilde, dtype=self.dtype)   # [L*N_f, N_t, N_r]

        # Start with state padded to the feature dimensions
        G = tf.pad(S,
                   paddings=[[0, self.window_len * self.N_f],  # mode-0
                             [0, self.N_t],                    # mode-1
                             [0, self.N_r]])                   # mode-2

        # Input placed along each mode individually.  Inclusion–exclusion
        # removes double counting from overlapping regions.
        Y_f = tf.pad(Y_tilde,
                     paddings=[[self.d_f, 0],
                               [0, self.d_t],
                               [0, self.d_r]])
        Y_t = tf.pad(Y_tilde,
                     paddings=[[0, self.d_f],
                               [self.d_t, 0],
                               [0, self.d_r]])
        Y_r = tf.pad(Y_tilde,
                     paddings=[[0, self.d_f],
                               [0, self.d_t],
                               [self.d_r, 0]])

        Y_ft = tf.pad(Y_tilde,
                      paddings=[[self.d_f, 0],
                                [self.d_t, 0],
                                [0, self.d_r]])
        Y_fr = tf.pad(Y_tilde,
                      paddings=[[self.d_f, 0],
                                [0, self.d_t],
                                [self.d_r, 0]])
        Y_tr = tf.pad(Y_tilde,
                      paddings=[[0, self.d_f],
                                [self.d_t, 0],
                                [self.d_r, 0]])
        Y_ftr = tf.pad(Y_tilde,
                       paddings=[[self.d_f, 0],
                                 [self.d_t, 0],
                                 [self.d_r, 0]])

        G = G + Y_f + Y_t + Y_r - Y_ft - Y_fr - Y_tr + Y_ftr

        scale = tf.cast(1.0 / tf.sqrt(tf.cast(self.df_feat * self.dt_feat * self.dr_feat, tf.float32)), self.dtype)
        G = G * scale

        if self.debug:
            _print_stats((S, "S"), (Y_tilde, "Y_tilde"), (G, "G(feature)"))
            _assert_finite(G, "G(feature)")

        return G  # [df_feat, dt_feat, dr_feat]
    
    def collect_features(self, Y_seq, washout=0):
        """
        Replay sequence and collect (feature_queue, target) pairs.
        Y_seq: [K, N_f, N_t, N_r]
        Returns: lists of G_k and Y_{k+1}.
        """
        self.reset_state()
        Y_seq = tf.convert_to_tensor(Y_seq, dtype=self.dtype)
        seq = tf.unstack(Y_seq)
        feats = []
        targets = []
        for k in range(len(seq) - 1):
            _ = self.step(seq[k])                  # updates self.S and self._last_Y_tilde
            if k >= washout:
                G_k = self.build_feature_queue(self.S, self._last_Y_tilde)
                feats.append(G_k)
                targets.append(seq[k + 1])

        return feats, targets

    def fit_readout_features(self, features, targets, lambdas=(0.0, 0.0, 0.0), iters=3):
        """
        Train mode-wise readout from the FEATURE QUEUE via ridge-ALS.

        features: list of G tensors, each [df_feat, dt_feat, dr_feat]
        targets : list of Y tensors, each [N_f, N_t, N_r]
        """
        # Ensure tensors have correct dtype
        features = [tf.convert_to_tensor(g, dtype=self.dtype) for g in features]
        targets  = [tf.convert_to_tensor(y, dtype=self.dtype) for y in targets]

        # Initialize readout matrices C_f, C_t, C_r
        self.C_f = self._rand_matrix(self.N_f, self.df_feat)
        self.C_t = self._rand_matrix(self.N_t, self.dt_feat)
        self.C_r = self._rand_matrix(self.N_r, self.dr_feat)

        lam_f, lam_t, lam_r = lambdas
        eye_f = tf.eye(self.df_feat, dtype=self.dtype)
        eye_t = tf.eye(self.dt_feat, dtype=self.dtype)
        eye_r = tf.eye(self.dr_feat, dtype=self.dtype)

        # Precompute unfoldings for efficiency on FEATURES (not states)
        F1_b = self.stack_unfoldings_batch(features, 0)  # [K, df_feat, dt_feat*dr_feat]
        Y1_b = self.stack_unfoldings_batch(targets,  0)  # [K, N_f,    N_t*N_r]
        F2_b = self.stack_unfoldings_batch(features, 1)  # [K, dt_feat, dr_feat*df_feat]
        Y2_b = self.stack_unfoldings_batch(targets,  1)  # [K, N_t,    N_f*N_r]
        F3_b = self.stack_unfoldings_batch(features, 2)  # [K, dr_feat, dt_feat*df_feat]
        Y3_b = self.stack_unfoldings_batch(targets,  2)  # [K, N_r,    N_f*N_t]

        if self.debug:
            print(f"DBG fit_readout_features: K={len(features)}, lambdas={lambdas}, iters={iters}")
            # quick global stats
            _print_stats((features[0], "G[0]"), (targets[0], "Y[0]"))

        for sweep in range(iters):
            # === Update C_f ===
            kron_tr = kron(self.C_t, self.C_r)          # [N_t*N_r, dt_feat*dr_feat]
            kron_tr_T = tf.transpose(kron_tr)           # [dt_feat*dr_feat, N_t*N_r]
            Z1_b = tf.matmul(F1_b, kron_tr_T)           # [K, df_feat, N_t*N_r]
            Z1 = tf.reshape(tf.transpose(Z1_b, [1,0,2]), [self.df_feat, -1])
            Y1 = tf.reshape(tf.transpose(Y1_b, [1,0,2]), [self.N_f,   -1])
            A1 = tf.matmul(Z1, Z1, adjoint_b=True) + lam_f * eye_f
            B1 = tf.matmul(Y1, Z1, adjoint_b=True)
            if self.debug:
                _print_stats((Z1, f"Z1(sweep={sweep})"))
                _assert_finite(tf.math.real(A1), "A1 real non-finite before cond()")
                _assert_finite(tf.math.imag(A1), "A1 imag non-finite before cond()")
                condA1 = _cond_number_gram(A1)
                print(f"DBG C_f sweep={sweep}: cond(A1)≈{condA1:.3e}, λ_f={lam_f:.1e}")
            self.C_f = (tf.transpose(
                tf.linalg.cholesky_solve(tf.linalg.cholesky(tf.cast(A1, tf.complex64)), 
                                         tf.transpose(tf.cast(B1, tf.complex64)))
            ) if self.safe_solve else
                # tf.transpose(tf.linalg.solve(A1, tf.transpose(B1)))
                tf.transpose(tf.linalg.lstsq(A1, tf.transpose(B1), l2_regularizer=0.0))
            )
            if self.debug:
                print(f"DBG C_f sweep={sweep}: ‖C_f‖_F={float(tf.linalg.norm(self.C_f).numpy()):.3e}")

            # === Update C_t ===
            kron_fr = kron(self.C_f, self.C_r)          # [N_f*N_r, df_feat*dr_feat]
            kron_fr_T = tf.transpose(kron_fr)           # [df_feat*dr_feat, N_f*N_r]
            Z2_b = tf.matmul(F2_b, kron_fr_T)           # [K, dt_feat, N_f*N_r]
            Z2 = tf.reshape(tf.transpose(Z2_b, [1,0,2]), [self.dt_feat, -1])
            Y2 = tf.reshape(tf.transpose(Y2_b, [1,0,2]), [self.N_t,     -1])
            if self.debug:
                _print_stats((F2_b, "F2_b"), (Y2_b, "Y2_b"),
                            (Z2_b, "Z2_b"), (Z2, "Z2(flat)"),
                            (Y2, "Y2(flat)"))
            A2 = tf.matmul(Z2, Z2, adjoint_b=True) + lam_t * eye_t
            B2 = tf.matmul(Y2, Z2, adjoint_b=True)
            if self.debug:
                _print_stats((Z2, f"Z2(sweep={sweep})"))
                _assert_finite(tf.math.real(A2), "A2 real non-finite before cond()")
                _assert_finite(tf.math.imag(A2), "A2 imag non-finite before cond()")
                condA2 = _cond_number_gram(A2)
                print(f"DBG C_t sweep={sweep}: cond(A2)≈{condA2:.3e}, λ_t={lam_t:.1e}")

            self.C_t = (tf.transpose(
                tf.linalg.cholesky_solve(tf.linalg.cholesky(tf.cast(A2, tf.complex64)), 
                                         tf.transpose(tf.cast(B2, tf.complex64)))
            ) if self.safe_solve else
                # tf.transpose(tf.linalg.solve(A2, tf.transpose(B2)))
                tf.transpose(tf.linalg.lstsq(A2, tf.transpose(B2), l2_regularizer=0.0))
            )
            if self.debug:
                print(f"DBG C_t sweep={sweep}: ‖C_t‖_F={float(tf.linalg.norm(self.C_t).numpy()):.3e}")

            # === Update C_r ===
            kron_ft = kron(self.C_f, self.C_t)          # [N_f*N_t, df_feat*dt_feat]
            kron_ft_T = tf.transpose(kron_ft)           # [df_feat*dt_feat, N_f*N_t]
            Z3_b = tf.matmul(F3_b, kron_ft_T)           # [K, dr_feat, N_f*N_t]
            Z3 = tf.reshape(tf.transpose(Z3_b, [1,0,2]), [self.dr_feat, -1])
            Y3 = tf.reshape(tf.transpose(Y3_b, [1,0,2]), [self.N_r,     -1])
            A3 = tf.matmul(Z3, Z3, adjoint_b=True) + lam_r * eye_r
            B3 = tf.matmul(Y3, Z3, adjoint_b=True)
            if self.debug:
                _print_stats((Z3, f"Z3(sweep={sweep})"))
                _assert_finite(tf.math.real(A3), "A3 real non-finite before cond()")
                _assert_finite(tf.math.imag(A3), "A3 imag non-finite before cond()")
                condA3 = _cond_number_gram(A3)
                print(f"DBG C_r sweep={sweep}: cond(A3)≈{condA3:.3e}, λ_r={lam_r:.1e}")

            self.C_r = (tf.transpose(
                tf.linalg.cholesky_solve(tf.linalg.cholesky(tf.cast(A3, tf.complex64)), 
                                         tf.transpose(tf.cast(B3, tf.complex64)))
            ) if self.safe_solve else
                # tf.transpose(tf.linalg.solve(A3, tf.transpose(B3)))
                tf.transpose(tf.linalg.lstsq(A3, tf.transpose(B3), l2_regularizer=0.0))
            )
            if self.debug:
                print(f"DBG C_r sweep={sweep}: ‖C_r‖_F={float(tf.linalg.norm(self.C_r).numpy()):.3e}")


    def _rand_matrix(self, rows, cols):
        """Return a random matrix with entries ~ N(0,1)."""
        if self.dtype.is_complex:
            real = tf.random.normal([rows, cols], dtype=self.dtype.real_dtype)
            imag = tf.random.normal([rows, cols], dtype=self.dtype.real_dtype)
            return tf.complex(real, imag)
        return tf.random.normal([rows, cols], dtype=self.dtype)

    def _init_matrix(self, rows, cols):
        """Draw a random matrix scaled to spectral radius ``target_rho``."""
        mat = self._rand_matrix(rows, cols)
        vals = tf.linalg.eigvals(mat)
        rho = tf.reduce_max(tf.abs(vals))
        mat = mat / tf.cast(rho, self.dtype)
        mat = mat * tf.cast(self.target_rho, self.dtype)
        return mat

    def reset_state(self):
        """Clear the reservoir state and input window."""
        # State tensor S ∈ ℝ^{d_f×d_t×d_r}
        self.S = tf.zeros([self.d_f, self.d_t, self.d_r], dtype=self.dtype)
        # Circular buffer storing the last L input tensors
        zero = tf.zeros([self.N_f, self.N_t, self.N_r], dtype=self.dtype)
        self.window = [zero for _ in range(self.window_len)]

    def step(self, Y_k):
        """Advance the reservoir by one time step and cache windowed input."""
        Y_k = tf.convert_to_tensor(Y_k, dtype=self.dtype)

        # Update input history buffer
        self.window = [Y_k] + self.window[:-1]
        if self.window_len > 1:
            Y_tilde = tf.concat(self.window, axis=0)  # [L*N_f, N_t, N_r]
        else:
            Y_tilde = Y_k                              # [N_f, N_t, N_r]

        # Cache the windowed input for feature-queue construction
        self._last_Y_tilde = Y_tilde

        # Recurrent state propagation: S ×_1 A_f ×_2 A_t ×_3 A_r
        pre = mode_n_product(self.S, self.A_f, 0)
        pre = mode_n_product(pre, self.A_t, 1)
        pre = mode_n_product(pre, self.A_r, 2)

        # Input injection: Y_tilde ×_1 U_f ×_2 U_t ×_3 U_r
        inp = mode_n_product(Y_tilde, self.U_f, 0)
        inp = mode_n_product(inp, self.U_t, 1)
        inp = mode_n_product(inp, self.U_r, 2)

        if self.debug:
            _print_stats((Y_k, "Y_k"), (Y_tilde, "Y_tilde"),
                         (self.S, "S_prev"), (pre, "pre(A*S)"), (inp, "inp(U*Y)"))
            sat = _tanh_saturation_fraction(pre + inp)
            print(f"DBG step: tanh saturation frac ≈ {sat:.3f}")
            _assert_finite(pre, "pre(A*S)")
            _assert_finite(inp, "inp(U*Y)")

        # Leaky state update
        new_state = self.sigma(pre + inp)
        S_next = (1 - self.alpha) * self.S + self.alpha * new_state
        if self.debug:
            _print_stats((new_state, "new_state=tanh(pre+inp)"),
                         (S_next, "S_next"))
            _assert_finite(S_next, "S_next")
        self.S = S_next

        return self.S

    def _readout(self, G=None, S=None, Y_tilde=None):
        """
        Map a feature tensor ``G`` (queue) to the channel domain using readout matrices.
        If ``G`` is None, it is built from ``S`` and ``Y_tilde`` (one of them may be None,
        in which case the cached self.S / self._last_Y_tilde are used).
        """
        if not all(hasattr(self, attr) for attr in ("C_f", "C_t", "C_r")):
            raise RuntimeError("Readout matrices not fitted. Call fit_readout_features() first.")

        if G is None:
            if S is None:
                S = self.S
            if Y_tilde is None:
                if not hasattr(self, "_last_Y_tilde"):
                    raise RuntimeError("No cached Y_tilde. Run step(...) at least once.")
                Y_tilde = self._last_Y_tilde
            G = self.build_feature_queue(S, Y_tilde)  # [df_feat, dt_feat, dr_feat]

        out = mode_n_product(G, self.C_f, 0)
        out = mode_n_product(out, self.C_t, 1)
        out = mode_n_product(out, self.C_r, 2)
        if self.debug:
            _print_stats((G, "G@readout"), (out, "readout_out"))
            _assert_finite(out, "readout_out")

        return out


    def predict(self, history, washout=0, lambdas=(1e-3, 1e-3, 1e-3), iters=2):
        """
        Train readout from `history` and return the next-step prediction.
        Returns a tensor shaped like one predicted subframe in your 8-D layout.
        """
        if self.debug:
            print(f"DBG predict(): history dtype={history.dtype} shape={history.shape} washout={washout}")

        # --- reshape to [M, S, N_f, N_t, N_r] (M = N_syms, S = N_subframes) ---
        x = tf.convert_to_tensor(history, dtype=self.dtype)
        x = tf.squeeze(x)                                # [S, N_r, N_t, M, N_f]
        x = tf.transpose(x, perm=[3, 0, 4, 2, 1])        # [M, S, N_f, N_t, N_r]

        # Inputs and labels across *subframes* for each symbol m
        inputs  = x[:, :-1, ...]   # [M, S-1, N_f, N_t, N_r]
        labels  = x[:,  1:, ...]   # [M, S-1, N_f, N_t, N_r]

        # --- Build (feature_queue, target) over subframes, per symbol, then union ---
        M = inputs.shape[0] or tf.shape(inputs)[0]
        all_feats, all_targets = [], []
        m_range = range(M) if isinstance(M, int) else tf.range(M)
        for m in m_range:
            seq_in  = inputs[m]   # [S-1, N_f, N_t, N_r]
            seq_out = labels[m]   # [S-1, N_f, N_t, N_r]
            self.reset_state()
            seq_in_list  = tf.unstack(seq_in)
            seq_out_list = tf.unstack(seq_out)
            for k in range(len(seq_in_list)):
                _ = self.step(seq_in_list[k])                         # updates self.S and self._last_Y_tilde
                if k >= washout:
                    G_k = self.build_feature_queue(self.S, self._last_Y_tilde)
                    all_feats.append(G_k)
                    all_targets.append(seq_out_list[k])
        
        all_feats_stack = tf.stack(all_feats)
        g_rms = tf.cast(tf.sqrt(tf.reduce_mean(tf.abs(all_feats_stack)**2)), dtype=self.dtype)
        all_feats = [g / (g_rms + 1e-8) for g in all_feats]

        y_rms = tf.cast(tf.sqrt(tf.reduce_mean(tf.abs(tf.stack(all_targets))**2)), dtype=self.dtype)
        all_targets = [y / (y_rms + 1e-8) for y in all_targets]

        if not all_feats:
            raise ValueError("After washout, no training pairs remain. Lower `washout`.")

        # --- Fit readout on feature queue ---
        self.fit_readout_features(all_feats, all_targets, lambdas=lambdas, iters=iters)

        if self.debug:
            _print_stats((tf.stack(all_targets,0), "train_targets"),
                         (tf.stack(train_preds,0), "train_preds"))
            print(f"DBG train recon NMSE: {train_nmse:.6f}")

        # --- DEBUG: training reconstruction NMSE ---
        def nmse(a, b, eps=1e-12):
            num = tf.reduce_sum(tf.abs(a - b) ** 2)
            den = tf.reduce_sum((tf.abs(a) + tf.abs(b)) ** 2) + eps
            return (num / den).numpy()
        
        train_preds = [self._readout(G=g) for g in all_feats]
        train_nmse = nmse(tf.stack(all_targets, 0), tf.stack(train_preds, 0))
        print(f"[DEBUG] train recon NMSE: {train_nmse:.6f}")


        # --- Predict the next subframe (per symbol index), then pack back to 8-D ---
        next_subframe_syms = []
        for m in m_range:
            seq_full = x[m]  # [S, N_f, N_t, N_r]
            self.reset_state()
            for Y in tf.unstack(seq_full):
                self.step(Y)  # after last Y, self.S and self._last_Y_tilde correspond to s = S-1
            G_last = self.build_feature_queue(self.S, self._last_Y_tilde)
            next_subframe_syms.append(self._readout(G=G_last))  # [N_f, N_t, N_r]

        Y_next = tf.stack(next_subframe_syms, axis=0)             # [M, N_f, N_t, N_r]
        if self.debug:
            _print_stats((Y_next, "Y_next_stack"))

        # pack to [1, 1, 1, N_r, 1, N_t, M, N_f]
        pred = Y_next[tf.newaxis, tf.newaxis, ...]
        pred = tf.transpose(pred, perm=[0, 5, 1, 4, 2, 3])
        if self.debug:
            _print_stats((pred, "pred(final)"))

        return pred

    

    def stack_unfoldings(self, tensors, mode):
        """Concatenate mode-``mode`` unfoldings of a list of tensors."""
        mats = [unfold(t, mode) for t in tensors]
        return tf.concat(mats, axis=1)

    def stack_unfoldings_batch(self, tensors, mode):
        """Stack mode-``mode`` unfoldings into a batch dimension."""
        mats = [unfold(t, mode) for t in tensors]
        return tf.stack(mats, axis=0)

    def collect_states(self, Y_seq, washout=0):
        """Replay sequence and collect (state, target) pairs.

        Parameters
        ----------
        Y_seq : sequence of tensors
            List or array of length ``K`` containing channel tensors of shape
            ``[N_f, N_t, N_r]``.
        washout : int, optional
            Number of initial steps to skip when collecting training samples.

        Returns
        -------
        tuple(list, list)
            Lists of states ``S_{k+1}`` and targets ``Y_{k+2}``.
        """
        self.reset_state()
        Y_seq = tf.convert_to_tensor(Y_seq, dtype=self.dtype)
        seq = tf.unstack(Y_seq)
        states = []
        targets = []
        for k in range(len(seq) - 1):
            S = self.step(seq[k])
            if k >= washout:
                states.append(S)
                targets.append(seq[k + 1])
        return states, targets

    def fit_readout(self, states, targets, lambdas=(0.0, 0.0, 0.0), iters=2):
        """Train mode-wise readout matrices via ridge-ALS.

        Parameters
        ----------
        states : list of tf.Tensor
            Reservoir states ``S`` with shape ``[d_f, d_t, d_r]``.
        targets : list of tf.Tensor
            Desired next channels ``Y`` with shape ``[N_f, N_t, N_r]``.
        lambdas : tuple(float, float, float)
            Ridge regularization for ``C_f``, ``C_t``, and ``C_r`` respectively.
        iters : int
            Number of alternating minimization sweeps.
        """

        # Ensure tensors have correct dtype
        states = [tf.convert_to_tensor(s, dtype=self.dtype) for s in states]
        targets = [tf.convert_to_tensor(y, dtype=self.dtype) for y in targets]

        # Initialize readout matrices C_f, C_t, C_r
        self.C_f = self._rand_matrix(self.N_f, self.d_f)
        self.C_t = self._rand_matrix(self.N_t, self.d_t)
        self.C_r = self._rand_matrix(self.N_r, self.d_r)

        lam_f, lam_t, lam_r = lambdas
        eye_f = tf.eye(self.d_f, dtype=self.dtype)
        eye_t = tf.eye(self.d_t, dtype=self.dtype)
        eye_r = tf.eye(self.d_r, dtype=self.dtype)

        # Precompute unfoldings for efficiency
        S1_b = self.stack_unfoldings_batch(states, 0)  # [K, d_f, d_t*d_r]
        Y1_b = self.stack_unfoldings_batch(targets, 0)  # [K, N_f, N_t*N_r]
        S2_b = self.stack_unfoldings_batch(states, 1)  # [K, d_t, d_f*d_r]
        Y2_b = self.stack_unfoldings_batch(targets, 1)  # [K, N_t, N_f*N_r]
        S3_b = self.stack_unfoldings_batch(states, 2)  # [K, d_r, d_f*d_t]
        Y3_b = self.stack_unfoldings_batch(targets, 2)  # [K, N_r, N_f*N_t]

        for _ in range(iters):
            # === Update C_f ===
            # Columns in unfold(S,0) and unfold(Y,0) order the other two modes
            # as (t, r); hence use kron(C_t, C_r).
            kron_tr = kron(self.C_t, self.C_r)  # [N_t*N_r, d_t*d_r]
            kron_tr_T = tf.transpose(kron_tr)  # [d_t*d_r, N_t*N_r]
            Z1_b = tf.matmul(S1_b, kron_tr_T)  # [K, d_f, N_t*N_r]
            Z1 = tf.reshape(tf.transpose(Z1_b, [1, 0, 2]), [self.d_f, -1])
            Y1 = tf.reshape(tf.transpose(Y1_b, [1, 0, 2]), [self.N_f, -1])
            A1 = tf.matmul(Z1, Z1, adjoint_b=True) + lam_f * eye_f
            B1 = tf.matmul(Y1, Z1, adjoint_b=True)
            # self.C_f = tf.transpose(tf.linalg.solve(A1, tf.transpose(B1)))
            self.C_f = tf.transpose(tf.linalg.lstsq(A1, tf.transpose(B1), l2_regularizer=0.0))

            # === Update C_t ===
            # Mode-1 unfolding orders columns as (f, r); use kron(C_f, C_r).
            kron_fr = kron(self.C_f, self.C_r)  # [N_f*N_r, d_f*d_r]
            kron_fr_T = tf.transpose(kron_fr)  # [d_f*d_r, N_f*N_r]
            Z2_b = tf.matmul(S2_b, kron_fr_T)  # [K, d_t, N_f*N_r]
            Z2 = tf.reshape(tf.transpose(Z2_b, [1, 0, 2]), [self.d_t, -1])
            Y2 = tf.reshape(tf.transpose(Y2_b, [1, 0, 2]), [self.N_t, -1])
            A2 = tf.matmul(Z2, Z2, adjoint_b=True) + lam_t * eye_t
            B2 = tf.matmul(Y2, Z2, adjoint_b=True)
            # self.C_t = tf.transpose(tf.linalg.solve(A2, tf.transpose(B2)))
            self.C_t = tf.transpose(tf.linalg.lstsq(A2, tf.transpose(B2), l2_regularizer=0.0))

            # === Update C_r ===
            # Mode-2 unfolding orders columns as (f, t); use kron(C_f, C_t).
            kron_ft = kron(self.C_f, self.C_t)  # [N_f*N_t, d_f*d_t]
            kron_ft_T = tf.transpose(kron_ft)  # [d_f*d_t, N_f*N_t]
            Z3_b = tf.matmul(S3_b, kron_ft_T)  # [K, d_r, N_f*N_t]
            Z3 = tf.reshape(tf.transpose(Z3_b, [1, 0, 2]), [self.d_r, -1])
            Y3 = tf.reshape(tf.transpose(Y3_b, [1, 0, 2]), [self.N_r, -1])
            A3 = tf.matmul(Z3, Z3, adjoint_b=True) + lam_r * eye_r
            B3 = tf.matmul(Y3, Z3, adjoint_b=True)
            # self.C_r = tf.transpose(tf.linalg.solve(A3, tf.transpose(B3)))
            self.C_r = tf.transpose(tf.linalg.lstsq(A3, tf.transpose(B3), l2_regularizer=0.0))

    def forecast(self, Y_init, steps):
        """Autoregressively predict ``steps`` future channel tensors.

        Parameters
        ----------
        Y_init : tf.Tensor
            Starting channel tensor ``[N_f, N_t, N_r]``.
        steps : int
            Number of future time steps to generate.

        Returns
        -------
        list of tf.Tensor
            Predicted sequence, each of shape ``[N_f, N_t, N_r]``.
        """
        preds = []
        Y_k = tf.convert_to_tensor(Y_init, dtype=self.dtype)
        for _ in range(steps):
            S = self.step(Y_k)
            Y_k = self._readout(S)
            preds.append(Y_k)
        return preds