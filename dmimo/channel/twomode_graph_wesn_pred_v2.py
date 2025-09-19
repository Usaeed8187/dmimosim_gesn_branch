import copy
import numpy as np
import tensorflow as tf

from dmimo.config import Ns3Config, RCConfig
from dmimo.channel import lmmse_channel_estimation

class twomode_graph_wesn_pred_v2:

    def __init__(self, rc_config, num_freq_re, num_rx_ant, num_tx_ant, type=np.complex64):
        
        self.rc_config = rc_config
        self.ns3_config = Ns3Config()

        self.dtype = type

        self.num_freq_re = num_freq_re
        self.N_r = num_rx_ant
        self.N_t = num_tx_ant

        self.sparsity = rc_config.W_tran_sparsity
        self.spectral_radius = rc_config.W_tran_radius
        self.input_scale = rc_config.input_scale
        self.window_length = rc_config.window_length
        self.reg = rc_config.regularization
        self.enable_window = rc_config.enable_window
        self.history_len = rc_config.history_len

        seed = 10
        self.RS = np.random.RandomState(seed)

        # one tx/rx pair is one vertex
        self.num_tx_nodes = int((self.N_t - self.ns3_config.num_bs_ant)  / self.ns3_config.num_ue_ant) + 1
        self.num_rx_nodes = int((self.N_r - self.ns3_config.num_bs_ant)  / self.ns3_config.num_ue_ant) + 1
        self.N_v = self.num_rx_nodes * self.num_tx_nodes # number of vertices in the graph
        self.N_e = int((self.N_v*(self.N_v-1))/2) # number of edges in the graph (at most. some of them will be zeroed out)

        self.N_in_left = self.N_r
        if self.enable_window:
            self.N_in_right = self.N_t * self.window_length # TODO: only windowing on the transmit antenna axis for now. evaluate windowing on the receive antenna axis later
        else:
            self.N_in_right = self.N_t

        self.d_left = self.ns3_config.num_bs_ant # TODO: currently just basing on the size of the input. try other configurations
        self.d_right = self.ns3_config.num_bs_ant * self.window_length

        self.init_weights()

    def init_weights(self):

        matrices_left = []
        matrices_right = []
        for _ in range(self.N_v):
            result = self.sparse_mat(self.d_left)
            matrices_left.append(result)
            result = self.sparse_mat(self.d_right)
            matrices_right.append(result)
        
        self.W_N_left = np.concatenate(matrices_left, axis=1)
        self.W_N_right = np.concatenate(matrices_right, axis=0)
        
        self.W_res_left = self.sparse_mat(self.d_left)
        self.W_res_right = self.sparse_mat(self.d_right)

        self.W_in_left = 2 * (self.RS.rand(self.d_left, self.N_in_left) - 0.5) # TODO: check if I should make this complex later
        self.W_in_right = 2 * (self.RS.rand(self.N_in_right, self.d_right) - 0.5) # TODO: check if I should make this complex later

        self.S_0 = np.zeros([self.N_v, self.d_left, self.d_right], dtype=self.dtype)
    
    def predict(self, h_freq_csi_history):

        h_freq_csi_predicted = self.pred_v2(h_freq_csi_history)

        return h_freq_csi_predicted
    

    def pred_v2(self, h_freq_csi_history):
        
        
        if tf.rank(h_freq_csi_history).numpy() == 8:
            h_freq_csi_history = np.asarray(h_freq_csi_history).transpose([0,1,2,3,4,5,7,6])
            num_batches = h_freq_csi_history.shape[1]
            num_rx_nodes = h_freq_csi_history.shape[2]
            num_rx_antennas = h_freq_csi_history.shape[3]
            num_tx_nodes = h_freq_csi_history.shape[4]
            num_tx_antennas = h_freq_csi_history.shape[5]
            num_freq_res = h_freq_csi_history.shape[6]
            num_ofdm_syms = h_freq_csi_history.shape[7]
        else:
            raise ValueError("\n The dimensions of h_freq_csi_history are not correct")

        channel_train_input = h_freq_csi_history[:-1, ...]
        channel_train_gt    = h_freq_csi_history[1:,  ...]
        
        if not self.enable_window:
            window_weights = None

        chan_pred = np.zeros(h_freq_csi_history[0,...].shape, dtype=self.dtype)

        S_list, Y_list = [], []

        # --------- (A) FEATURE BUILD PHASE: stack all RBs (and OFDM syms) ----------
        for freq_re in range(num_freq_res):
            for ofdm_sym in range(num_ofdm_syms):

                # Form lists of inputs and labels. 
                # Each element in the list corresponds to a tx-rx node pair. 
                # The lists loop over tx nodes first and rx nodes second
                channel_train_input_list, meta_input = self.extract_tx_rx_node_pairs_numpy(channel_train_input[..., freq_re, ofdm_sym])
                channel_train_gt_list, meta_gt = self.extract_tx_rx_node_pairs_numpy(channel_train_gt[..., freq_re, ofdm_sym])
                num_node_pairs = len(channel_train_gt_list)

                # Optional: do NOT reset S_0 here if you want cross-RB continuity
                self.S_0 = np.zeros([self.N_v, self.d_left, self.d_right], dtype=self.dtype)

                S_f_list, Y_f_list = self.build_S_Y(channel_train_input_list, channel_train_gt_list, meta_input, curr_window_weights=None)
                S_list.append(S_f_list); Y_list.append(Y_f_list)
        
        # S_list: list over RB/OFDM; each item is S_f_list (length N_v) where S_f_list[v] is (F_v, T_chunk). 
        # Y_list: same structure; Y_f_list[v] is (n_rx_v*n_tx_v, T_chunk)
        # Note: F_v = n_rx_v * L*n_tx_v + self.d_left*self.d_right

        # 1) Aggregate per-vertex across all RB/OFDM chunks
        S_vertex = [[] for _ in range(self.N_v)]
        Y_vertex = [[] for _ in range(self.N_v)]

        for S_f_list, Y_f_list in zip(S_list, Y_list):
            for v in range(self.N_v):
                S_vertex[v].append(S_f_list[v])  # (F_v, T_chunk)
                Y_vertex[v].append(Y_f_list[v])  # (n_rx_v*n_tx_v, T_chunk)

        # 2) Concatenate along time axis for each vertex
        S_all_per_v = [np.concatenate(S_vertex[v], axis=1) for v in range(self.N_v)]   # (F_v, sum_T_v)
        Y_all_per_v = [np.concatenate(Y_vertex[v], axis=1) for v in range(self.N_v)]   # (n_rx_v*n_tx_v, sum_T_v)

        # 3) Solve ridge per vertex
        self.W_out_list = []
        for v in range(self.N_v):
            S_v = S_all_per_v[v]                     # (F_v, T_v)
            Y_v = Y_all_per_v[v]                     # (n_rx_v*n_tx_v, T_v)
            G_v = self.reg_p_inv(S_v)                # (T_v, F_v) := S_v^H (S_v S_v^H + λI)^(-1)
            W_out_v = Y_v @ G_v                      # (n_rx_v*n_tx_v, F_v)
            self.W_out_list.append(W_out_v.astype(self.dtype, copy=False))


        # --------- (C) PREDICTION PHASE with per-vertex W_out ----------
        chan_pred = np.squeeze(np.zeros(h_freq_csi_history[0, ...].shape, dtype=self.dtype))

        for freq_re in range(num_freq_res):
            for ofdm_sym in range(num_ofdm_syms):
                # 1) Build per-vertex test inputs (use the last known sequence to predict next step)
                channel_test_input_list, meta_test = self.extract_tx_rx_node_pairs_numpy(
                    channel_train_gt[..., freq_re, ofdm_sym]
                )  # list length N_v; each (T, n_rx_v, n_tx_v)

                # Optional: continuity across RBs (comment these two lines to carry state)
                self.S_0 = np.zeros([self.N_v, self.d_left, self.d_right], dtype=self.dtype)

                # 2) Window & scale (same as training)
                if self.enable_window:
                    Y_3D_win_list = self.form_window_input_signal_list(channel_test_input_list, window_weights=None)
                else:
                    Y_3D_win_list = [arr.copy() for arr in channel_test_input_list]
                Y_3D_win_list = [arr * self.input_scale for arr in Y_3D_win_list]

                # 3) Transit states → (T, N_v, d_left*d_right)
                S_3D_transit = self.state_transit(Y_3D_win_list, meta_test)
                T_test = S_3D_transit.shape[0]
                t_last = T_test - 1  # use the last column for next-step prediction

                # 4) For each vertex: build feature S_v (F_v, T_test), predict, take last col
                #    and stitch into a full [N_r, N_t] matrix
                H_hat_full = np.zeros((self.N_r, self.N_t), dtype=self.dtype)

                for v in range(self.N_v):
                    # features
                    S_transit_v = S_3D_transit[:, v, :]           # (T_test, 2*dL*dR)
                    Yin_v = Y_3D_win_list[v]                      # (T_test, n_rx_v, L*n_tx_v)
                    Yin_v_flat = Yin_v.reshape(T_test, -1)        # (T_test, n_rx_v * L*n_tx_v)
                    S_v_time = np.concatenate([S_transit_v, Yin_v_flat], axis=-1)  # (T_test, F_v)
                    S_v = S_v_time.T                               # (F_v, T_test)

                    # predict per vertex
                    W_out_v = self.W_out_list[v]                  # (n_rx_v*n_tx_v, F_v)
                    Y_hat_v = W_out_v @ S_v                       # (n_rx_v*n_tx_v, T_test)

                    # take next-step prediction from the last time column
                    y_last = Y_hat_v[:, t_last]                   # (n_rx_v*n_tx_v,)
                    entry = meta_test[v]
                    rx_idx = entry["rx_ant_idx"]                  # list of rx antenna indices
                    tx_idx = entry["tx_ant_idx"]                  # list of tx antenna indices
                    n_rx_v = len(rx_idx)
                    n_tx_v = len(tx_idx)
                    H_v = y_last.reshape(n_rx_v, n_tx_v, order='C')

                    # stitch block into full matrix
                    H_hat_full[np.ix_(rx_idx, tx_idx)] = H_v

                # 5) Write into your output tensor at the right slots
                chan_pred[:, :, freq_re, ofdm_sym] = H_hat_full

        # transpose back if you transposed earlier
        chan_pred = chan_pred[np.newaxis, np.newaxis, :, np.newaxis, :, :, :]
        chan_pred = chan_pred.transpose([0,1,2,3,4,6,5])
        chan_pred = tf.convert_to_tensor(chan_pred)
        
        return chan_pred

    def _node_slices(self, total_ant, first_node_ants=4, rest_node_ants=2):
        """Return a list of index arrays, one per node, covering `total_ant` antennas."""
        if total_ant < first_node_ants:
            raise ValueError("total_ant < first_node_ants")
        sizes = [first_node_ants]
        remaining = total_ant - first_node_ants
        if remaining % rest_node_ants != 0:
            raise ValueError("Remaining antennas not divisible by rest_node_ants")
        sizes.extend([rest_node_ants] * (remaining // rest_node_ants))

        nodes = []
        start = 0
        for sz in sizes:
            nodes.append(np.arange(start, start + sz))
            start += sz
        # sanity check
        assert sum(len(x) for x in nodes) == total_ant
        return nodes  # list of 1D index arrays

    def extract_tx_rx_node_pairs_numpy(self, h_freq_csi_history):
        """
        h: (B, R, T)
        returns:
        pairs: list of arrays with shape (B, R_sel, T_sel)
        meta: list of dicts describing (rx_node, tx_node, their antenna indices)
        """
        if h_freq_csi_history.ndim != 6:
            raise ValueError("Expected shape (B,1,1,R,1,T)")

        B, _, _, R, _, T = h_freq_csi_history.shape

        rx_nodes = self._node_slices(R, first_node_ants=4, rest_node_ants=2)
        tx_nodes = self._node_slices(T, first_node_ants=4, rest_node_ants=2)

        pairs = []
        meta  = []

        # tx-major ordering of the list; change loop order if you prefer tx-major
        for r_i, r_idx in enumerate(rx_nodes):
            for t_i, t_idx in enumerate(tx_nodes):
                # Slice: (B, r, t)
                block = h_freq_csi_history[:, 0, 0, r_idx, 0, ...]
                block = block[:, :, t_idx]
                pairs.append(block)
                meta.append({
                    "rx_node": r_i,
                    "rx_ant_idx": r_idx.tolist(),
                    "tx_node": t_i,
                    "tx_ant_idx": t_idx.tolist(),
                    "block_shape": tuple(block.shape)
                })

        return pairs, meta


    def build_S_Y(self, channel_input, channel_output, meta_input, curr_window_weights):
        # channel_input, channel_output: [T, N_r, N_t]
        Y_3D_list = channel_input
        Y_target_3D_list = channel_output

        # Compute adjacency matrices once per RE and per ofdm sym.
        self.adjacency = self.compute_eigenmode_adjacency_cov(Y_3D_list)

        if self.enable_window:
            Y_3D_win_list = self.form_window_input_signal_list(Y_3D_list, curr_window_weights)
        else:
            # TODO: not adapted to twomode input yet
            forget = getattr(self, "forget_length", 0)
            Y_3D_win = np.concatenate([Y_3D, np.zeros([Y_3D.shape[0], forget, Y_3D.shape[2]], dtype=self.dtype)], axis=1)

        Y_3D_win_list = [arr * self.input_scale for arr in Y_3D_win_list]
        S_3D_transit = self.state_transit(Y_3D_win_list, meta_input)
        T = S_3D_transit.shape[0]
        
        S_list, Y_list = [], []

        for v in range(self.N_v):

            # ---- features for vertex v ----
            # S_3D_transit per-vertex slice: (T, d_left*d_right)
            S_transit_v = S_3D_transit[:, v, ...]  # time-major

            # Flatten windowed input for skip (varies with vertex dims)
            Yin_v = Y_3D_win_list[v]                 # (T, n_rx_v, L*n_tx_v)
            Yin_v_flat = Yin_v.reshape(T, -1)        # (T, n_rx_v * L*n_tx_v)

            # Concatenate [state || skip] per time, then transpose to (F_v, T)
            S_v_time_major = np.concatenate([S_transit_v, Yin_v_flat], axis=-1)  # (T, F_v)
            S_v = S_v_time_major.T  # (F_v, T)
            S_list.append(S_v.astype(self.dtype, copy=False))

            # ---- targets for vertex v ----
            # Use the GT block. Flatten per time, then transpose to (n_rx_v*n_tx_v, T)
            Yv = Y_target_3D_list[v]                          # (T, n_rx_v, n_tx_v)
            Yv_flat = Yv.reshape(T, -1, order='C')            # (T, n_rx_v*n_tx_v)
            Y_v = Yv_flat.T                                   # (n_rx_v*n_tx_v, T)
            Y_list.append(Y_v.astype(self.dtype, copy=False))

        return S_list, Y_list


    def calculate_window_weights(self, h_freq_csi_history):

        if self.window_weighting_method == 'autocorrelation':
            def autocorrelation(x):
                """Compute the autocorrelation of a 1D signal."""
                n = len(x)
                x_mean = np.mean(x)
                x_var = np.var(x)
                acf = np.correlate(x - x_mean, x - x_mean, mode='full') / (n * x_var)
                return acf[n-1:]  # Keep only non-negative lags

            h_reshaped = np.moveaxis(h_freq_csi_history, -1, 0)
            acf_result = np.apply_along_axis(autocorrelation, 0, h_reshaped)
            acf_result = np.squeeze(np.mean(acf_result, axis=-1))

            window_weights = np.abs(acf_result)
        elif self.window_weighting_method == 'same_weights':
            window_weights = 1
        elif self.window_weighting_method == 'exponential_decay':
            # x = np.linspace(0, self.window_length-1, self.history_len*self.num_ofdm_sym)
            x = np.linspace(0, self.window_length-1, h_freq_csi_history.shape[1])
            window_weights = np.exp(-x/2)
        elif self.window_weighting_method == 'none':
            window_weights = np.ones(h_freq_csi_history.shape[1])
        else:
            raise ValueError("\n The window_weighting_method specified is not implemented")
        
        return window_weights

    def sparse_mat(self, m):
        
        W = 2*(self.RS.rand(m, m) - 0.5) + 2j*(self.RS.rand(m, m) - 0.5)
        W[self.RS.rand(*W.shape) < self.sparsity] = 0+1j*0
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        W = W * (self.spectral_radius / radius)
        
        return W

    def cal_nmse(self, H, H_hat):
        H_hat = tf.cast(H_hat, dtype=self.dtype)
        mse = np.sum(np.abs(H - H_hat) ** 2)
        normalization_factor = np.sum((np.abs(H) + np.abs(H_hat)) ** 2)
        nmse = mse / normalization_factor
        return nmse

    def reg_p_inv(self, X):
        # X: (F, T)
        F = X.shape[0]
        G = X @ X.conj().T + self.reg * np.eye(F, dtype=self.dtype)  # (F,F)
        G = X.conj().T @ np.linalg.pinv(G)                 # (T,F)

        return G

    def form_window_input_signal_list(self,
        pairs_list,
        window_weights=None,   # optional: shape (L,) or broadcastable to (L, 1, 1)
    ):
        """
        Args
        ----
        pairs_list : list of np.ndarray
            Each element has shape (B, T_sel, R_sel), where:
            - B is num_time_steps
            - T_sel is #Tx antennas for this node-pair
            - R_sel is #Rx antennas for this node-pair
        window_length : int
            Causal window length L.
        dtype : np.dtype or None
            Output dtype; defaults to dtype of each input element.
        window_weights : array-like or None
            Optional weights applied per lag ell (0..L-1). If provided,
            blocks[ell] *= window_weights[ell]. Must be broadcastable to (T_sel, R_sel).

        Returns
        -------
        out_list : list of np.ndarray
            Each element has shape (B, L*T_sel, R_sel).
        """
        L = int(self.window_length)
        if L <= 0:
            raise ValueError("window_length must be >= 1")

        #TODO: not adding window weight functionality yet
        # if window_weights is not None:
        #     ww = np.asarray(window_weights)
        #     if ww.ndim == 1:
        #         # (L,) → scale entire (T_sel, R_sel) slice by ww[ell]
        #         if ww.shape[0] != L:
        #             raise ValueError("window_weights length must equal window_length")
        #     else:
        #         # e.g., (L,1,1) or (L,T_sel,1) etc., will broadcast per block
        #         if ww.shape[0] != L:
        #             raise ValueError("window_weights first dim must equal window_length")
        # else:
        #     ww = None

        out_list = []

        for Y in pairs_list:
            if Y.ndim != 3:
                raise ValueError("Each list element must be [B, R_sel, T_sel]")
            B, R_sel, T_sel = Y.shape

            Y_win = np.zeros((B, R_sel, L * T_sel), dtype=self.dtype)

            # Prebuild a zero block for causal padding
            zero_block = np.zeros((R_sel, T_sel), dtype=self.dtype)

            for k in range(B):
                blocks = []
                for ell in range(L):
                    t = k - ell
                    if t >= 0:
                        block = Y[t]  # (R_sel, T_sel)
                    else:
                        block = zero_block

                    # if ww is not None:
                    #     # multiply by weight for this lag; rely on broadcasting
                    #     block = block * ww[ell]

                    blocks.append(block)

                # Concatenate along Tx axis → (R_sel, L*T_sel)
                Y_win[k] = np.concatenate(blocks, axis=1)

            out_list.append(Y_win)

        return out_list


    def state_transit(self, Y_3D_list, meta):

        T = Y_3D_list[0].shape[0] # number of samples

        S_3D = copy.deepcopy(self.S_0)

        S_3D_list = []
        S_4D = []
        
        for t in range(T):

            step_features = []
            neighbor_states = [None] * self.N_v

            for u in range(self.N_v):
                W_N_left_idx = np.arange((u*self.d_left),(u+1)*self.d_left)
                W_N_right_idx = np.arange((u*self.d_right),(u+1)*self.d_right)
                neighbor_states[u] = self.W_N_left[:, W_N_left_idx] @ S_3D[u, ...] @ self.W_N_right[W_N_right_idx, :]   # (d_left, d_right)

            for v in range(self.N_v):
                
                entry = meta[v]
                rx_idx = entry["rx_ant_idx"]
                tx_idx = entry["tx_ant_idx"][-1]
                prev_tx_idx = entry["tx_ant_idx"][0] * self.window_length
                tx_idx = (tx_idx+1) * self.window_length
                tx_idx = np.arange(prev_tx_idx, tx_idx)
                prev_tx_idx = tx_idx[-1]+1

                input_contrib = self.W_in_left[:, rx_idx] @ Y_3D_list[v][t, ...] @ self.W_in_right[tx_idx, :]

                neighborhood_contrib = np.zeros((self.d_left, self.d_right), dtype=self.dtype)
                row = self.adjacency[v]
                for u in range(self.N_v):
                    w = row[u]
                    neighborhood_contrib += w * neighbor_states[u]

                S_3D[v,...] = self.complex_tanh(input_contrib + neighborhood_contrib)

                step_features.append(S_3D[v,...].reshape(-1))
            
            step_features = np.stack(step_features, axis=0)  # (N_v, d_left*d_right)
            S_3D_list.append(step_features)

        self.S_0 = S_3D.copy()

        S_3D = np.stack(S_3D_list, axis=0)  # (T, N_v, d_left*d_right)

        return S_3D
    
    # ------------------------------------------------------------------------------------
    # Adjacency calculation
    # ------------------------------------------------------------------------------------
    def compute_eigenmode_adjacency_cov(
        self,
        channel_input,      # list of arrays, each (B, R_sel, T_sel)
        subspace_rank_tx=2, # q_t
        subspace_rank_rx=2, # q_r
        shrinkage=0.0,
        transpose_if_needed=False
    ):
        """
        Covariance-based Tx/Rx subspace adjacency.

        Uses top-q eigenvectors of time-averaged covariances.
        If two subspaces have different row counts, the smaller is
        zero-padded so both live in the same ambient dimension.
        """

        N_v = len(channel_input)
        Tx_bases, Rx_bases = [], []

        def pad_rows(Q, N_target):
            """Zero-pad rows of Q (n x q) to (N_target x q)."""
            n, q = Q.shape
            if n == N_target:
                return Q
            Qp = np.zeros((N_target, q), dtype=Q.dtype)
            Qp[:n, :] = Q
            return Qp

        for v in range(N_v):
            Yv = channel_input[v]
            if Yv.ndim != 3:
                raise ValueError("Each list element must be 3D (B, R_sel, T_sel)")
            if transpose_if_needed and Yv.shape[1] < Yv.shape[2]:
                Yv = np.transpose(Yv, (0, 2, 1))

            B, R_sel, T_sel = Yv.shape

            # Empirical covariances
            R_tx = np.zeros((T_sel, T_sel), dtype=np.complex64)
            R_rx = np.zeros((R_sel, R_sel), dtype=np.complex64)
            for b in range(B):
                Hb = Yv[b]
                R_tx += Hb.conj().T @ Hb
                R_rx += Hb @ Hb.conj().T
            R_tx /= max(B, 1)
            R_rx /= max(B, 1)

            if shrinkage > 0.0:
                tr_tx = np.real(np.trace(R_tx)) / max(T_sel, 1)
                tr_rx = np.real(np.trace(R_rx)) / max(R_sel, 1)
                R_tx = (1.0 - shrinkage) * R_tx + shrinkage * tr_tx * np.eye(T_sel, dtype=R_tx.dtype)
                R_rx = (1.0 - shrinkage) * R_rx + shrinkage * tr_rx * np.eye(R_sel, dtype=R_rx.dtype)

            # Eigen-decomp, take top-q
            eval_tx, evec_tx = np.linalg.eigh(R_tx)
            eval_rx, evec_rx = np.linalg.eigh(R_rx)
            idx_tx = np.argsort(eval_tx)[::-1]
            idx_rx = np.argsort(eval_rx)[::-1]
            q_t = min(subspace_rank_tx, T_sel)
            q_r = min(subspace_rank_rx, R_sel)
            Ut = evec_tx[:, idx_tx[:q_t]]
            Ur = evec_rx[:, idx_rx[:q_r]]

            # QR for safety
            Ut, _ = np.linalg.qr(Ut)
            Ur, _ = np.linalg.qr(Ur)

            Tx_bases.append(Ut)
            Rx_bases.append(Ur)

        def subspace_affinity(Ua, Ub):
            # Zero-pad to common row count if needed
            n_a, q_a = Ua.shape
            n_b, q_b = Ub.shape
            n_max = max(n_a, n_b)
            Ua_p = pad_rows(Ua, n_max)
            Ub_p = pad_rows(Ub, n_max)
            q = min(q_a, q_b)
            if q == 0:
                return 0.0
            S = Ua_p.conj().T @ Ub_p
            return float(np.minimum(1.0, np.linalg.norm(S, 'fro')**2 / q))

        adj_tx = np.zeros((N_v, N_v), dtype=float)
        adj_rx = np.zeros((N_v, N_v), dtype=float)
        for v in range(N_v):
            for u in range(N_v):
                adj_tx[v, u] = subspace_affinity(Tx_bases[v], Tx_bases[u])
                adj_rx[v, u] = subspace_affinity(Rx_bases[v], Rx_bases[u])

        np.fill_diagonal(adj_tx, 1.0)
        np.fill_diagonal(adj_rx, 1.0)
        adj_tx = 0.5 * (adj_tx + adj_tx.T)
        adj_rx = 0.5 * (adj_rx + adj_rx.T)

        adj = (adj_rx + adj_tx) / 2
        return adj


    def complex_tanh(self, Y):
        return np.tanh(np.real(Y)) + 1j * np.tanh(np.imag(Y))