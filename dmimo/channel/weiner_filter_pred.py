import numpy as np


class weiner_filter_pred:

    def __init__(self, method="using_one_link", type=np.complex64):
        
        self.dtype = type

        self.method = method # "using_one_link", "using_all_links"

    def predict(self, h_freq_csi_history):
        
        if self.method == "using_one_link":
            
            T, _, _, RxAnt, _, TxAnt, num_syms, RB = h_freq_csi_history.shape
            num_bs_ant = 4
            num_ue_ant = 2
            num_rx_nodes = int((RxAnt - num_bs_ant) / num_ue_ant)
            num_tx_nodes = int((TxAnt - num_bs_ant) / num_ue_ant)

            h_freq_csi_predicted = np.zeros(h_freq_csi_history[0,...].shape, dtype=h_freq_csi_history.dtype)

            # Helper to get antenna index ranges
            def ant_range(node_idx, bs_count, ue_count):
                if node_idx == 0:
                    return np.arange(0, bs_count)
                start = bs_count + (node_idx - 1) * ue_count
                return np.arange(start, start + ue_count)
            
            for tx_node_idx in range(num_tx_nodes):
                tx_ant_idx = ant_range(tx_node_idx, num_bs_ant, num_ue_ant)
                for rx_node_idx in range(num_rx_nodes):
                    rx_ant_idx = ant_range(rx_node_idx, num_bs_ant, num_ue_ant)

                    curr_h_freq_csi_history = h_freq_csi_history[:,:,:,rx_ant_idx,:,...]
                    curr_h_freq_csi_history = curr_h_freq_csi_history[:,:,:,:,:,tx_ant_idx,...]
                    
                    tmp = self.predict_next_using_single_link_K(np.squeeze(curr_h_freq_csi_history))
                    tmp = tmp[np.newaxis, np.newaxis, :, np.newaxis, ...]
                    
                    rx_idx, tx_idx = np.ix_(rx_ant_idx, tx_ant_idx)
                    h_freq_csi_predicted[:, :, rx_idx, :, tx_idx, :, :] = tmp.transpose(2, 4, 0, 1, 3, 5, 6)
        
        else:
            raise Exception("Multiple Link Predictor not implemented here yet")
        
        return h_freq_csi_predicted


    def predict_using_single_link(self, h_freq_csi_history, lam=1e-4):
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
    
    def predict_next_using_single_link_K(self, h_freq_csi_history, K=4, lam=1e-2):
        
        """
        Train a one-step-ahead Wiener predictor on rolling pairs and predict the *next* subframe (S).
        h_freq_csi_history: [S, N_r, N_t, Nsym, Nsc]
        """
        H = h_freq_csi_history
        S, N_r, N_t, Nsym, Nsc = H.shape
        assert S >= K + 1, "Need at least K+1 subframes to train and predict next."

        B = N_r * N_t * Nsym * Nsc
        X = H.reshape(S, B).astype(np.complex128, copy=False)

        # Build training pairs: for t=K..S-2, predict X[t+1] from [X[t],...,X[t-K+1]]
        idxs = list(range(K, S-1))
        Phi = np.stack([np.stack([X[t-k, :] for k in range(K)], axis=1) for t in idxs], axis=0)  # [Ttr, B, K]
        y   = np.stack([X[t+1, :] for t in idxs], axis=0)                                        # [Ttr, B]
        Phi = Phi.reshape(-1, K)  # [(Ttr*B), K]
        y   = y.reshape(-1)       # [(Ttr*B)]

        A = Phi.conj().T @ Phi + lam * np.eye(K, dtype=np.complex128)
        b = Phi.conj().T @ y
        w = np.linalg.solve(A, b)  # [K]

        # Predict NEXT subframe S from [X[S-1],...,X[S-K]]
        Phi_next = np.stack([X[S-1-k, :] for k in range(K)], axis=1)  # [B, K]
        y_next_hat = Phi_next @ w
        return y_next_hat.reshape(N_r, N_t, Nsym, Nsc).astype(self.dtype, copy=False)
