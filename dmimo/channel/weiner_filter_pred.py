import numpy as np


class weiner_filter_pred:

    def __init__(self, method="using_one_link", type=np.complex64):
        
        self.dtype = type

        self.method = method # "using_one_link", "using_one_link_windowed", "using_one_link_MIMO", "using_all_links_MIMO"

    def predict(self, h_freq_csi_history, K=None):
        

        # Helper to get antenna index ranges
        def ant_range(node_idx, bs_count, ue_count):
            if node_idx == 0:
                return np.arange(0, bs_count)
            start = bs_count + (node_idx - 1) * ue_count
            return np.arange(start, start + ue_count)
            
        if "using_one_link" in self.method:
            
            T, _, _, RxAnt, _, TxAnt, num_syms, RB = h_freq_csi_history.shape
            num_bs_ant = 4
            num_ue_ant = 2
            num_rx_nodes = int((RxAnt - num_bs_ant) / num_ue_ant)+1
            num_tx_nodes = int((TxAnt - num_bs_ant) / num_ue_ant)+1

            h_freq_csi_predicted = np.zeros(h_freq_csi_history[0,...].shape, dtype=h_freq_csi_history.dtype)
            
            for tx_node_idx in range(num_tx_nodes):
                tx_ant_idx = ant_range(tx_node_idx, num_bs_ant, num_ue_ant)
                for rx_node_idx in range(num_rx_nodes):
                    rx_ant_idx = ant_range(rx_node_idx, num_bs_ant, num_ue_ant)

                    curr_h_freq_csi_history = h_freq_csi_history[:,:,:,rx_ant_idx,:,...]
                    curr_h_freq_csi_history = curr_h_freq_csi_history[:,:,:,:,:,tx_ant_idx,...]
                    
                    if self.method == "using_one_link":
                        tmp = self.predict_using_single_link_vectorized(np.squeeze(curr_h_freq_csi_history))
                    elif self.method == "using_one_link_windowed":
                        if K is None:
                            K = curr_h_freq_csi_history.shape[0]-1
                        tmp = self.predict_next_using_single_link_rolling(np.squeeze(curr_h_freq_csi_history), K=K)
                    elif self.method == "using_one_link_MIMO":
                        tmp = self.predict_using_single_link_MIMO_ARp(np.squeeze(curr_h_freq_csi_history), K=K)
                    tmp = tmp[np.newaxis, np.newaxis, :, np.newaxis, ...]
                    
                    rx_idx, tx_idx = np.ix_(rx_ant_idx, tx_ant_idx)
                    h_freq_csi_predicted[:, :, rx_idx, :, tx_idx, :, :] = tmp.transpose(2, 4, 0, 1, 3, 5, 6)
        
        elif self.method == "using_all_links_MIMO":
            # expects a dict {(r,t): H_rt} with heterogeneous sizes.

            link_hist_dict = {}

            T, _, _, RxAnt, _, TxAnt, num_syms, RB = h_freq_csi_history.shape
            num_bs_ant = 4
            num_ue_ant = 2
            num_rx_nodes = int((RxAnt - num_bs_ant) / num_ue_ant)+1
            num_tx_nodes = int((TxAnt - num_bs_ant) / num_ue_ant)+1

            for r_node in range(num_rx_nodes):
                rx_idx = ant_range(r_node, num_bs_ant, num_ue_ant)
                Nr = len(rx_idx)

                for t_node in range(num_tx_nodes):
                    tx_idx = ant_range(t_node, num_bs_ant, num_ue_ant)
                    Nt = len(tx_idx)

                    curr_h_freq_csi_history = h_freq_csi_history[:,:,:,rx_idx,:,...]
                    curr_h_freq_csi_history = curr_h_freq_csi_history[:,:,:,:,:,tx_idx,...]
                    curr_h_freq_csi_history = curr_h_freq_csi_history[:, 0, 0, :, 0, :, :, :]

                    link_hist_dict[(r_node, t_node)] = curr_h_freq_csi_history.astype(np.complex128, copy=False)

            h_freq_csi_predicted_dict = self.predict_using_all_links_MIMO_ARp(link_hist_dict, K=K, lam=3e-1,
                                                           standardize=True, add_bias=False, solve="auto")
            
            h_freq_csi_predicted = np.zeros(h_freq_csi_history[0,...].shape, dtype=h_freq_csi_history.dtype)
            
            for (r_node, t_node), H_pred_rt in h_freq_csi_predicted_dict.items():
                rx_ant_idx = ant_range(r_node, num_bs_ant, num_ue_ant)
                tx_ant_idx = ant_range(t_node, num_bs_ant, num_ue_ant)

                tmp = H_pred_rt[np.newaxis, np.newaxis, :, np.newaxis, :, :, :]

                rx_idx, tx_idx = np.ix_(rx_ant_idx, tx_ant_idx)
                h_freq_csi_predicted[:, :, rx_idx, :, tx_idx, :, :] = tmp.transpose(2, 4, 0, 1, 3, 5, 6)

        else:
            raise Exception("Incompatible method")
        
        return h_freq_csi_predicted


    def predict_using_single_link_vectorized(self, h_freq_csi_history, lam=1e-4):
        """
        Predict the last subframe of a single link using ONLY its own prior subframes.

        Args:
            h_freq_csi_history: np.ndarray, shape [S, N_r, N_t, Nsym, Nsc], complex.
            lam: float, ridge regularization (>=0).

        Returns:
            h_freq_csi_predicted: np.ndarray, shape [N_r, N_t, Nsym, Nsc], complex,
                the prediction for the last subframe from all previous subframes.
        """
        H = h_freq_csi_history
        S, N_r, N_t, Nsym, Nsc = H.shape
        assert S >= 2, "Need at least 2 subframes to leave the last one as target."

        # Flatten per subframe to [B], with B = N_r*N_t*Nsym*Nsc
        B = N_r * N_t * Nsym * Nsc
        # Solve in complex128 for numerical stability, cast back to self.dtype at the end
        H_flat = H.reshape(S, B).astype(np.complex128, copy=False)

        # Target: last subframe of the link
        y = H_flat[-1, :]  # [B]

        # Features: all previous subframes of the same link
        F = S - 1
        Phi = H_flat[:-1, :].T  # [B, F]

        # Complex ridge Wiener solve: w = (Phi^H Phi + lam I)^-1 Phi^H y
        A = Phi.conj().T @ Phi            # [F, F]
        b = Phi.conj().T @ y              # [F]
        A_reg = A + lam * np.eye(F, dtype=np.complex128)
        w = np.linalg.solve(A_reg, b)     # [F]

        # Predict and reshape back to tensor
        Phi = H_flat[1:, :].T  # [B, F]
        y_hat = Phi @ w                   # [B]
        h_freq_csi_predicted = y_hat.reshape(N_r, N_t, Nsym, Nsc).astype(self.dtype, copy=False)

        return h_freq_csi_predicted
    
    def predict_next_using_single_link_rolling(self, H, K=4, lam=1e-2):
        """
        One-step-ahead linear predictor using K lags, trained in the *dual*.
        H: [S, N_r, N_t, Nsym, Nsc] complex
        Returns: prediction for frame S (next after S-1)
        """
        H = H.astype(np.complex128, copy=False)
        S, N_r, N_t, Nsym, Nsc = H.shape
        assert S >= K + 1, "Need at least K+1 frames"
        B = N_r * N_t * Nsym * Nsc
        X = H.reshape(S, B)  # [S, B]

        # Build rolling windows: for t=K..S-2, input = [X[t],...,X[t-K+1]], target = X[t+1]
        idxs = list(range(K, S-1))
        Ttr  = len(idxs)

        # Build Phi (Ttr x K*B) and Y (Ttr x B)
        # (This is ~ Ttr*K*B elements; typically << (K*B)^2.)
        Phi = np.empty((Ttr, K*B), dtype=np.complex128)
        Y   = np.empty((Ttr, B),    dtype=np.complex128)
        for i, t in enumerate(idxs):
            Phi[i, :] = np.concatenate([X[t - k, :] for k in range(K)], axis=0)
            Y[i, :]   = X[t + 1, :]

        # Dual matrices
        # A = Phi Phi^H  (Ttr x Ttr), Hermitian PSD
        A = Phi @ Phi.conj().T
        A += lam * np.eye(Ttr, dtype=np.complex128)  # ridge

        # Solve A * Alpha = Y  (multiple RHS). Use Cholesky for stability.
        try:
            L = np.linalg.cholesky(A)                      # A = L L^H
            # Solve L Z = Y
            Z = np.linalg.solve(L, Y)
            # Solve L^H Alpha = Z
            Alpha = np.linalg.solve(L.conj().T, Z)         # (Ttr x B)
        except np.linalg.LinAlgError:
            # Fallback to generic solve if Cholesky fails (heavy regularization should prevent this)
            Alpha = np.linalg.solve(A, Y)

        # Form last K-lag feature (phi_last) and its dual "kernel" k = Phi * phi_last^H
        phi_last = np.concatenate([X[S - 1 - k, :] for k in range(K)], axis=0)  # (K*B,)
        # k_j = <Phi_j, phi_last> for j=1..Ttr
        k = Phi @ phi_last.conj()  # (Ttr,)

        # Prediction: y_hat = k^H * Alpha   (B,)
        y_hat = k.conj().T @ Alpha
        return y_hat.reshape(N_r, N_t, Nsym, Nsc).astype(self.dtype, copy=False)


    def predict_using_single_link_MIMO(self, h_freq_csi_history, lam=1e-2, standardize=False):
        """
        Vectorized MIMO AR(1):  vec(H_t) = W vec(H_{t-1}) + e_t
        Fit W with ridge using ALL (sym, sc) as independent samples.

        Args:
            h_freq_csi_history: complex array [S, N_r, N_t, Nsym, Nsc]
            lam: ridge regularization
            standardize: if True, z-score features across samples (and unscale outputs)

        Returns:
            pred: complex array [N_r, N_t, Nsym, Nsc] = prediction for frame S from frame S-1
        """
        H = h_freq_csi_history.astype(np.complex128, copy=False)
        S, Nr, Nt, Nsym, Nsc = H.shape
        assert S >= 2, "Need at least 2 frames."

        D = Nr * Nt  # feature/output dims per sample

        # Build training matrices Phi (N x D), Y (N x D)
        # One sample per (t in 1..S-1, sym, sc)
        N = (S - 1) * Nsym * Nsc
        Phi = np.empty((N, D), dtype=np.complex128)
        Y   = np.empty((N, D), dtype=np.complex128)

        idx = 0
        for t in range(1, S):
            Xprev = H[t-1]  # (Nr, Nt, Nsym, Nsc)
            Xnow  = H[t]
            for s in range(Nsym):
                for f in range(Nsc):
                    Phi[idx, :] = Xprev[:, :, s, f].reshape(-1)
                    Y[idx, :]   = Xnow[:, :, s, f].reshape(-1)
                    idx += 1

        # Optional standardization (per-feature across samples)
        if standardize:
            mu_phi = Phi.mean(axis=0, keepdims=True)
            sd_phi = Phi.std(axis=0, keepdims=True) + 1e-12
            Phi_n = (Phi - mu_phi) / sd_phi
            # Center Y as well (helps if drift exists)
            mu_y = Y.mean(axis=0, keepdims=True)
            Y_n = Y - mu_y
        else:
            Phi_n, Y_n = Phi, Y

        # Solve ridge in the *primal* (D is small: ≤ 576)
        # W = (Phi^H Phi + lam I)^-1 Phi^H Y   (each RHS column is handled at once)
        G = Phi_n.conj().T @ Phi_n
        R = Phi_n.conj().T @ Y_n
        A = G + lam * np.eye(D, dtype=np.complex128)
        try:
            # Cholesky is faster & stabler
            L = np.linalg.cholesky(A)
            Z = np.linalg.solve(L, R)
            W = np.linalg.solve(L.conj().T, Z)   # (D x D)
        except np.linalg.LinAlgError:
            W = np.linalg.solve(A, R)

        # Predict next frame S from frame S-1 for each (sym, sc)
        pred = np.empty((Nr, Nt, Nsym, Nsc), dtype=np.complex128)
        for s in range(Nsym):
            for f in range(Nsc):
                x_last = H[S-1, :, :, s, f].reshape(1, -1)  # (1 x D)
                if standardize:
                    xz = (x_last - mu_phi) / sd_phi
                    y_hat = xz @ W + mu_y
                else:
                    y_hat = x_last @ W
                pred[:, :, s, f] = y_hat.reshape(Nr, Nt)

        return pred.astype(self.dtype, copy=False)

    def predict_using_single_link_MIMO_ARp(self, h_freq_csi_history, K=3, lam=1e-2,
                                       standardize=False, add_bias=False):
        """
        Vectorized MIMO AR(p): vec(H_t) = [W1 ... WK] [vec(H_{t-1}); ... ; vec(H_{t-K})] + e_t
        Fit with ridge using ALL (sym, sc) as independent samples, then predict frame S.

        Args:
            h_freq_csi_history: complex array [S, Nr, Nt, Nsym, Nsc]
            K: number of lags (p)
            lam: ridge regularization
            standardize: z-score features across samples (and center outputs)
            add_bias: append a bias column

        Returns:
            pred: [Nr, Nt, Nsym, Nsc] complex
        """
        H = h_freq_csi_history.astype(np.complex128, copy=False)
        S, Nr, Nt, Nsym, Nsc = H.shape
        assert S >= K + 1, "Need at least K+1 frames."
        D = Nr * Nt
        N = (S - K) * Nsym * Nsc  # samples

        # Build design Phi: (N x (K*D [+1 if bias])), and targets Y: (N x D)
        Pdim = K * D + (1 if add_bias else 0)
        Phi = np.empty((N, Pdim), dtype=np.complex128)
        Y   = np.empty((N, D),    dtype=np.complex128)

        idx = 0
        for t in range(K, S):
            Hstack = []
            for k in range(1, K+1):
                Hstack.append(H[t-k].reshape(D, Nsym*Nsc))   # (D, Nsym*Nsc)
            XprevK = np.concatenate(Hstack, axis=0)          # (K*D, Nsym*Nsc)
            Xnow   = H[t].reshape(D, Nsym*Nsc)               # (D, Nsym*Nsc)
            # Fill rows for all (sym, sc)
            for s in range(Nsym*Nsc):
                row = XprevK[:, s]
                if add_bias:
                    Phi[idx, :-1] = row
                    Phi[idx,  -1] = 1.0
                else:
                    Phi[idx, :]   = row
                Y[idx, :] = Xnow[:, s]
                idx += 1

        # Optional standardization (across samples)
        if standardize:
            mu_phi = Phi.mean(axis=0, keepdims=True)
            sd_phi = Phi.std(axis=0, keepdims=True) + 1e-12
            Phi_n = (Phi - mu_phi) / sd_phi
            mu_y  = Y.mean(axis=0)
            Y_n   = Y - mu_y
        else:
            Phi_n, Y_n = Phi, Y

        # Primal ridge: Wbig has shape ((K*D [+1]), D)
        G = Phi_n.conj().T @ Phi_n
        R = Phi_n.conj().T @ Y_n
        A = G + lam * np.eye(Pdim, dtype=np.complex128)
        try:
            L = np.linalg.cholesky(A)
            Z = np.linalg.solve(L, R)
            Wbig = np.linalg.solve(L.conj().T, Z)
        except np.linalg.LinAlgError:
            Wbig = np.linalg.solve(A, R)

        # Predict frame S from last K frames for each (sym, sc)
        pred = np.empty((Nr, Nt, Nsym, Nsc), dtype=np.complex128)
        # Build last K stack once
        Hstack_last = []
        for k in range(1, K+1):
            Hstack_last.append(H[S-k].reshape(D, Nsym*Nsc))
        XlastK = np.concatenate(Hstack_last, axis=0)  # (K*D, Nsym*Nsc)

        for s in range(Nsym*Nsc):
            x = XlastK[:, s]
            if add_bias:
                x = np.concatenate([x, np.array([1.0+0j])])
            if standardize:
                xz = (x - mu_phi.ravel()) / sd_phi.ravel()
                y_hat = xz @ Wbig + mu_y
            else:
                y_hat = x @ Wbig
            y_mat = y_hat.reshape(Nr, Nt)
            # map s back to (sym, sc)
            sym = s // Nsc
            sc  = s %  Nsc
            pred[:, :, sym, sc] = y_mat

        return pred.astype(self.dtype, copy=False)
    

    def predict_using_all_links_MIMO_ARp(self, link_hist_dict, K=3, lam=1e-2,
                                            standardize=False, add_bias=False, solve="auto"):
        """
        Joint AR(p) predictor over *all* links with heterogeneous antenna sizes.

        Args:
            link_hist_dict: dict mapping (r_idx, t_idx) -> H_rt
                H_rt shape: [S, N_r(r_idx), N_t(t_idx), Nsym, Nsc], complex
                S, Nsym, Nsc must be identical across all links.
            K: number of lags (p)
            lam: ridge regularization (>=0)
            standardize: if True, z-score features across samples and center outputs
            add_bias: include a bias column
            solve: "auto" | "primal" | "dual"
                - primal solves (P x P) with P = K*D_tot[+1]
                - dual   solves (N x N) with N = #samples

        Returns:
            pred_dict: dict {(r_idx, t_idx): pred_rt}, where
                pred_rt has shape [N_r(r_idx), N_t(t_idx), Nsym, Nsc], complex
                (prediction for frame S from frames S-1..S-K)
        """
        # ----- Sanity, shapes, common dims -----
        links = sorted(link_hist_dict.keys())  # fixed stacking order
        assert len(links) > 0, "Empty link set."
        # Check shared S, Nsym, Nsc
        S = None; Nsym = None; Nsc = None
        for key in links:
            H = np.asarray(link_hist_dict[key])
            assert H.ndim == 5, f"Link {key} must be [S, Nr, Nt, Nsym, Nsc]"
            if S is None:
                S, _, _, Nsym, Nsc = H.shape
            else:
                s2, _, _, ns2, nf2 = H.shape
                assert s2 == S and ns2 == Nsym and nf2 == Nsc, "All links must share S, Nsym, Nsc"
        assert S >= K + 1, "Need at least K+1 frames."

        def _offset_slice(sl: slice, offset: int) -> slice:
            return slice(sl.start + offset, sl.stop + offset, sl.step)


        # ----- Build per-link sizes and global offsets -----
        RtNt = {key: (link_hist_dict[key].shape[1], link_hist_dict[key].shape[2]) for key in links}
        D_rt = {key: RtNt[key][0] * RtNt[key][1] for key in links}
        cum = 0
        slices = {}
        for key in links:
            d = D_rt[key]
            slices[key] = slice(cum, cum + d)
            cum += d
        Dtot = cum

        # ----- Prepare training matrices -----
        # Each (frame t in K..S-1) and each (sym, sc) yields one sample.
        N_samples = (S - K) * Nsym * Nsc
        Pdim = K * Dtot + (1 if add_bias else 0)

        Phi = np.empty((N_samples, Pdim), dtype=np.complex128)
        Y   = np.empty((N_samples, Dtot), dtype=np.complex128)

        # Pre-reshape frames to (D_rt, Nsym*Nsc) for efficiency
        def frame_to_vecmat(link_key, t):  # returns (D_rt, Nsym*Nsc)
            H = link_hist_dict[link_key][t].astype(np.complex128, copy=False)  # (Nr, Nt, Nsym, Nsc)
            Nr, Nt = RtNt[link_key]
            return H.reshape(Nr*Nt, Nsym*Nsc)

        idx = 0
        for t in range(K, S):
            # XprevK: (K*Dtot, Nsym*Nsc), Xnow: (Dtot, Nsym*Nsc)
            XprevK = np.empty((K * Dtot, Nsym * Nsc), dtype=np.complex128)
            Xnow   = np.empty((Dtot,      Nsym * Nsc), dtype=np.complex128)
            # fill link by link into the stacked buffers
            for key in links:
                sl = slices[key]
                Xnow[sl, :] = frame_to_vecmat(key, t)
            for k in range(1, K+1):
                row_base = (k - 1) * Dtot
                for key in links:
                    sl = slices[key]
                    XprevK[_offset_slice(sl, row_base), :] = frame_to_vecmat(key, t - k)

            # write samples over (sym, sc)
            for s in range(Nsym * Nsc):
                row = XprevK[:, s]
                if add_bias:
                    Phi[idx, :-1] = row
                    Phi[idx,  -1] = 1.0
                else:
                    Phi[idx, :]   = row
                Y[idx, :] = Xnow[:, s]
                idx += 1
        assert idx == N_samples

        # ----- Optional standardization -----
        if standardize:
            mu_phi = Phi.mean(axis=0, keepdims=True)
            sd_phi = Phi.std(axis=0, keepdims=True) + 1e-12
            Phi_n = (Phi - mu_phi) / sd_phi
            mu_y  = Y.mean(axis=0)
            Y_n   = Y - mu_y
        else:
            Phi_n, Y_n = Phi, Y

        # ----- Ridge solve (primal vs dual) -----
        if solve == "auto":
            solve = "primal" if Pdim <= N_samples else "dual"

        if solve == "primal":
            # Wbig: (Pdim x Dtot)
            G = Phi_n.conj().T @ Phi_n
            R = Phi_n.conj().T @ Y_n
            A = G + lam * np.eye(Pdim, dtype=np.complex128)
            try:
                L = np.linalg.cholesky(A)
                Z = np.linalg.solve(L, R)
                Wbig = np.linalg.solve(L.conj().T, Z)
            except np.linalg.LinAlgError:
                Wbig = np.linalg.solve(A, R)
            solve_info = ("primal", Pdim, N_samples)

        else:  # "dual"
            # Alpha solves (Phi Phi^H + lam I) Alpha = Y
            A = Phi_n @ Phi_n.conj().T
            A += lam * np.eye(N_samples, dtype=np.complex128)
            try:
                L = np.linalg.cholesky(A)
                Z = np.linalg.solve(L, Y_n)
                Alpha = np.linalg.solve(L.conj().T, Z)
            except np.linalg.LinAlgError:
                Alpha = np.linalg.solve(A, Y_n)
            # Wbig = Phi^H Alpha  (maps Pdim -> Dtot)
            Wbig = Phi_n.conj().T @ Alpha
            solve_info = ("dual", Pdim, N_samples)

        # ----- Predict frame S from last K frames -----
        # Build last K stack once: XlastK (K*Dtot, Nsym*Nsc)
        XlastK = np.empty((K * Dtot, Nsym * Nsc), dtype=np.complex128)
        for k in range(1, K+1):
            row_base = (k - 1) * Dtot
            for key in links:
                sl = slices[key]
                XlastK[_offset_slice(sl, row_base), :] = frame_to_vecmat(key, S - k)

        pred_dict = {}
        for key in links:
            Nr, Nt = RtNt[key]
            d = D_rt[key]
            sl_out = slices[key]
            assert (sl_out.stop - sl_out.start) == (Nr * Nt), \
                f"Slice width {(sl_out.stop - sl_out.start)} != Nr*Nt {Nr*Nt} for key {key}"

            pred_mat = np.empty((Nr, Nt, Nsym, Nsc), dtype=np.complex128)
            for s in range(Nsym * Nsc):
                x = XlastK[:, s]
                if add_bias:
                    x = np.concatenate([x, np.array([1.0 + 0j])])
                if standardize:
                    y_hat = ((x - mu_phi.ravel()) / sd_phi.ravel()) @ Wbig + mu_y
                else:
                    y_hat = x @ Wbig
                # extract this link's segment and reshape
                assert y_hat.ndim == 1 and y_hat.shape[0] == Dtot, f"y_hat has shape {y_hat.shape}, expected ({Dtot},)"
                y_seg = y_hat[sl_out]
                pred_mat[:, :, s // Nsc, s % Nsc] = y_seg.reshape(Nr, Nt)
            pred_dict[key] = pred_mat.astype(self.dtype, copy=False)

        return pred_dict  # {(r,t): [Nr,Nt,Nsym,Nsc]}
