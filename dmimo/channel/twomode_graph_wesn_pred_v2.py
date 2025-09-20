import copy
import numpy as np
import tensorflow as tf

from dmimo.config import Ns3Config, RCConfig
from dmimo.channel import lmmse_channel_estimation

class twomode_graph_wesn_pred_v2:

    def __init__(self, rc_config, num_freq_re, num_rx_ant, num_tx_ant, adjacency_method=None, type=np.complex64):
        
        self.rc_config = rc_config
        self.ns3_config = Ns3Config()

        self.dtype = type
        self.tf_dtype = tf.as_dtype(self.dtype)

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

        self.num_epochs = getattr(rc_config, "num_epochs", 0)
        self.learning_rate = getattr(rc_config, "lr", 0.1)
        self.edge_weight_update_method = getattr(rc_config, "edge_weight_update_method", "grad_descent")

        self.adjacency_method = adjacency_method

        seed = 10
        self.RS = np.random.RandomState(seed)

        # one tx/rx pair is one vertex
        self.num_tx_nodes = int((self.N_t - self.ns3_config.num_bs_ant)  / self.ns3_config.num_ue_ant) + 1
        self.num_rx_nodes = int((self.N_r - self.ns3_config.num_bs_ant)  / self.ns3_config.num_ue_ant) + 1
        self.N_v = self.num_rx_nodes * self.num_tx_nodes # number of vertices in the graph
        self.N_e = int((self.N_v*(self.N_v-1))/2) # number of edges in the graph (at most. some of them will be zeroed out)

        lower_tri = np.tril_indices(self.N_v, k=-1)
        self.lower_tri_indices = np.stack(lower_tri, axis=1).astype(np.int32)
        self.num_adjacency_edges = lower_tri[0].size
        self.lower_tri_indices_tf = tf.constant(self.lower_tri_indices, dtype=tf.int32)

        self.N_in_left = self.N_r
        if self.enable_window:
            self.N_in_right = self.N_t * self.window_length # TODO: only windowing on the transmit antenna axis for now. evaluate windowing on the receive antenna axis later
        else:
            self.N_in_right = self.N_t

        self.d_left = self.ns3_config.num_bs_ant # TODO: currently just basing on the size of the input. try other configurations
        self.d_right = self.ns3_config.num_bs_ant * self.window_length

        self.init_weights()

    def init_weights(self):

        # TODO: try matching different vertex W_N's and W_ins to the different vertex input dimensions (Nr, Nt)
        matrices_left = []
        matrices_right = []
        for _ in range(self.N_v):
            result = self.sparse_mat(self.d_left)
            matrices_left.append(result)
            result = self.sparse_mat(self.d_right)
            matrices_right.append(result)
        
        self.W_N_left = np.concatenate(matrices_left, axis=1)
        self.W_N_right = np.concatenate(matrices_right, axis=0)
        
        self.W_in_left = 2 * (self.RS.rand(self.d_left, self.N_in_left) - 0.5) # TODO: check if I should make this complex later
        self.W_in_right = 2 * (self.RS.rand(self.N_in_right, self.d_right) - 0.5) # TODO: check if I should make this complex later

        self.S_0 = np.zeros([self.N_v, self.d_left, self.d_right], dtype=self.dtype)

        W_left_blocks = [self.W_N_left[:, i*self.d_left:(i+1)*self.d_left] for i in range(self.N_v)]
        W_right_blocks = [self.W_N_right[i*self.d_right:(i+1)*self.d_right, :] for i in range(self.N_v)]
        self.W_N_left_blocks_tf = tf.constant(np.stack(W_left_blocks, axis=0).astype(self.dtype), dtype=self.tf_dtype)
        self.W_N_right_blocks_tf = tf.constant(np.stack(W_right_blocks, axis=0).astype(self.dtype), dtype=self.tf_dtype)

        self.W_in_left_tf = tf.cast(tf.constant(self.W_in_left, dtype=tf.float32), self.tf_dtype)
        self.W_in_right_tf = tf.cast(tf.constant(self.W_in_right, dtype=tf.float32), self.tf_dtype)
    
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
                # self.S_0 = np.zeros([self.N_v, self.d_left, self.d_right], dtype=self.dtype)

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

                if self.adjacency_method is None:
                    self.adjacency = self.compute_None_adjacency()
                elif self.adjacency_method == 'eigenmode_cov':
                    adjacency_init = self.compute_eigenmode_adjacency_cov(channel_test_input_list)

                # Optional: continuity across RBs (comment these two lines to carry state)
                # self.S_0 = np.zeros([self.N_v, self.d_left, self.d_right], dtype=self.dtype)

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
        if self.adjacency_method is None:
            adjacency_init = self.compute_None_adjacency()
        elif self.adjacency_method == 'eigenmode_cov':
            adjacency_init = self.compute_eigenmode_adjacency_cov(Y_3D_list)

        if self.enable_window:
            Y_3D_win_list = self.form_window_input_signal_list(Y_3D_list, curr_window_weights)
        else:
            # TODO: not adapted to twomode input yet
            forget = getattr(self, "forget_length", 0)
            Y_3D_win = np.concatenate([Y_3D, np.zeros([Y_3D.shape[0], forget, Y_3D.shape[2]], dtype=self.dtype)], axis=1)
            
        Y_3D_win_list = [arr * self.input_scale for arr in Y_3D_win_list]

        if self.edge_weight_update_method == 'grad_descent' and self.num_epochs > 0:
            self.adjacency = self.optimize_adjacency_grad_descent(
                Y_3D_win_list,
                Y_target_3D_list,
                meta_input,
                adjacency_init,
            )
        else:
            self.adjacency = adjacency_init.astype(np.float32, copy=False)

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
    def compute_None_adjacency(self):

        adj_None = np.zeros((self.N_v, self.N_v), dtype=float)
        np.fill_diagonal(adj_None, 1.0)

        return adj_None


    def compute_eigenmode_adjacency_cov_v0(
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

    def compute_eigenmode_adjacency_cov(
        self,
        channel_input,      # list of arrays, each (B, R_sel, T_sel)
        subspace_rank_tx=2, # q_t
        subspace_rank_rx=2, # q_r
        shrinkage=0.0,
        transpose_if_needed=False
    ):
        """
        Covariance-based *joint* Tx/Rx subspace adjacency (Kronecker-space chordal affinity).

        For each vertex v, estimate dominant Tx and Rx subspaces U_t[v] and U_r[v] from
        time-averaged covariances R_tx, R_rx. The adjacency between vertices a,b is

            A[a,b] = ( || U_r[a]^H U_r[b] ||_F^2 / q_r ) * ( || U_t[a]^H U_t[b] ||_F^2 / q_t )

        which equals the normalized chordal affinity of the *joint* subspaces span(U_r ⊗ U_t)
        without explicitly forming Kronecker products. Values are in [0,1], diagonal = 1,
        and the result is symmetrized.

        Args:
            channel_input: list of length N_v; each item is array (B, R_sel, T_sel) of complex64
                        mini-batches for that vertex/link.
            subspace_rank_tx: q_t, target Tx subspace rank (capped by T_sel).
            subspace_rank_rx: q_r, target Rx subspace rank (capped by R_sel).
            shrinkage: scalar in [0,1]; if >0, apply diagonal shrinkage to covariances.
            transpose_if_needed: if True and R_sel < T_sel, transpose per-batch matrices so
                                that the "Rx" dimension is the larger one (keeps conventions).

        Returns:
            adj: (N_v, N_v) float64 numpy array with values in [0,1].
        """
        import numpy as np

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

        # --- Per-vertex subspace estimation ---
        for v in range(N_v):
            Yv = channel_input[v]
            if Yv.ndim != 3:
                raise ValueError("Each list element must be 3D (B, R_sel, T_sel)")
            if transpose_if_needed and Yv.shape[1] < Yv.shape[2]:
                Yv = np.transpose(Yv, (0, 2, 1))  # now (B, T_sel, R_sel); next lines adapt via shapes

            B, dim1, dim2 = Yv.shape

            # Interpret as (B, R_sel, T_sel) regardless of optional transpose
            # If we transposed above, then dim1 is T_sel and dim2 is R_sel; swap back logically.
            if transpose_if_needed and Yv.shape[1] > Yv.shape[2]:
                R_sel, T_sel = dim2, dim1
                # Build covariances by viewing Hb as (R_sel x T_sel) again:
                R_tx = np.zeros((T_sel, T_sel), dtype=np.complex64)
                R_rx = np.zeros((R_sel, R_sel), dtype=np.complex64)
                for b in range(B):
                    Hb = Yv[b].T  # (R_sel, T_sel)
                    R_tx += Hb.conj().T @ Hb
                    R_rx += Hb @ Hb.conj().T
            else:
                R_sel, T_sel = dim1, dim2
                R_tx = np.zeros((T_sel, T_sel), dtype=np.complex64)
                R_rx = np.zeros((R_sel, R_sel), dtype=np.complex64)
                for b in range(B):
                    Hb = Yv[b]  # (R_sel, T_sel)
                    R_tx += Hb.conj().T @ Hb
                    R_rx += Hb @ Hb.conj().T

            R_tx /= max(B, 1)
            R_rx /= max(B, 1)

            if shrinkage > 0.0:
                tr_tx = float(np.real(np.trace(R_tx))) / max(T_sel, 1)
                tr_rx = float(np.real(np.trace(R_rx))) / max(R_sel, 1)
                R_tx = (1.0 - shrinkage) * R_tx + shrinkage * tr_tx * np.eye(T_sel, dtype=R_tx.dtype)
                R_rx = (1.0 - shrinkage) * R_rx + shrinkage * tr_rx * np.eye(R_sel, dtype=R_rx.dtype)

            # Eigen-decomp (Hermitian), take top-q
            eval_tx, evec_tx = np.linalg.eigh(R_tx)
            eval_rx, evec_rx = np.linalg.eigh(R_rx)
            idx_tx = np.argsort(eval_tx)[::-1]
            idx_rx = np.argsort(eval_rx)[::-1]
            q_t = max(0, min(subspace_rank_tx, T_sel))
            q_r = max(0, min(subspace_rank_rx, R_sel))
            Ut = evec_tx[:, idx_tx[:q_t]] if q_t > 0 else np.zeros((T_sel, 0), dtype=R_tx.dtype)
            Ur = evec_rx[:, idx_rx[:q_r]] if q_r > 0 else np.zeros((R_sel, 0), dtype=R_rx.dtype)

            # QR for numerical stability (keeps orthonormal columns)
            if q_t > 0:
                Ut, _ = np.linalg.qr(Ut)
            if q_r > 0:
                Ur, _ = np.linalg.qr(Ur)

            Tx_bases.append(Ut)  # (T_sel x q_t)
            Rx_bases.append(Ur)  # (R_sel x q_r)

        # --- Joint subspace chordal affinity via product of Tx/Rx overlaps ---
        def joint_affinity(Ur_a, Ut_a, Ur_b, Ut_b):
            """
            A_ab = (||Ur_a^H Ur_b||_F^2 / q_r) * (||Ut_a^H Ut_b||_F^2 / q_t),
            with row-padding if needed.
            """
            # Handle empty subspaces
            q_r_a = Ur_a.shape[1]
            q_r_b = Ur_b.shape[1]
            q_t_a = Ut_a.shape[1]
            q_t_b = Ut_b.shape[1]
            q_r = min(q_r_a, q_r_b)
            q_t = min(q_t_a, q_t_b)
            if q_r == 0 or q_t == 0:
                return 0.0

            # Pad rows if antenna counts differ
            n_r = max(Ur_a.shape[0], Ur_b.shape[0])
            n_t = max(Ut_a.shape[0], Ut_b.shape[0])
            if Ur_a.shape[0] != n_r: Ur_a = pad_rows(Ur_a, n_r)
            if Ur_b.shape[0] != n_r: Ur_b = pad_rows(Ur_b, n_r)
            if Ut_a.shape[0] != n_t: Ut_a = pad_rows(Ut_a, n_t)
            if Ut_b.shape[0] != n_t: Ut_b = pad_rows(Ut_b, n_t)

            # Compute Frobenius norms of overlap matrices
            Sr = Ur_a.conj().T @ Ur_b          # (q_r_a x q_r_b)
            St = Ut_a.conj().T @ Ut_b          # (q_t_a x q_t_b)

            # Use only the leading min-dim blocks to normalize by q_r, q_t fairly
            Sr_eff = Sr[:q_r, :q_r]
            St_eff = St[:q_t, :q_t]

            num_r = np.linalg.norm(Sr_eff, 'fro')**2
            num_t = np.linalg.norm(St_eff, 'fro')**2

            # Normalize to [0,1] individually, then multiply (equivalent to Kronecker chordal)
            ar = float(num_r / max(q_r, 1))
            at = float(num_t / max(q_t, 1))
            val = ar * at
            # Numerical safety
            if not np.isfinite(val):
                val = 0.0
            return float(np.clip(val, 0.0, 1.0))

        adj = np.zeros((N_v, N_v), dtype=np.float64)
        for a in range(N_v):
            for b in range(N_v):
                adj[a, b] = joint_affinity(Rx_bases[a], Tx_bases[a], Rx_bases[b], Tx_bases[b])

        # Force symmetry and self-loops
        adj = 0.5 * (adj + adj.T)
        np.fill_diagonal(adj, 1.0)

        return adj
    
    def optimize_adjacency_grad_descent(self, Y_inputs, Y_targets, meta_input, adjacency_init):
        if self.num_epochs <= 0:
            return adjacency_init.astype(np.float32, copy=False)

        lower_initial = adjacency_init[np.tril_indices(self.N_v, k=-1)].astype(np.float32, copy=False)
        if not np.any(lower_initial):
            lower_initial = lower_initial + 1e-2

        adjacency_var = tf.Variable(lower_initial, dtype=tf.float32)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        Y_inputs_tf = [tf.convert_to_tensor(arr, dtype=self.tf_dtype) for arr in Y_inputs]
        Y_targets_tf = [tf.convert_to_tensor(arr, dtype=self.tf_dtype) for arr in Y_targets]
        W_in_left_slices, W_in_right_slices = self._prepare_vertex_slices(meta_input)

        best_loss = np.inf
        best_weights = lower_initial.copy()

        for curr_epoch in range(self.num_epochs):
            with tf.GradientTape() as tape:
                adjacency_matrix = self._build_adjacency_matrix_from_weights(adjacency_var)
                loss = self._compute_training_loss_tf(
                    Y_inputs_tf,
                    Y_targets_tf,
                    W_in_left_slices,
                    W_in_right_slices,
                    adjacency_matrix,
                )

            gradients = tape.gradient(loss, adjacency_var)
            if gradients is None:
                break

            optimizer.apply_gradients([(gradients, adjacency_var)])
            adjacency_var.assign(tf.clip_by_value(adjacency_var, 0.0, 5.0))

            curr_loss = float(loss.numpy())
            print(f"Epoch {curr_epoch+1}/{self.num_epochs}, Loss: {curr_loss:.6f}")

            if curr_loss < best_loss:
                best_loss = curr_loss
                best_weights = adjacency_var.numpy()

        optimized_adjacency = self._build_adjacency_matrix_from_weights(
            tf.constant(best_weights, dtype=tf.float32)
        )
        
        return optimized_adjacency.numpy().astype(np.float32, copy=False)

    def _prepare_vertex_slices(self, meta_input):
        W_in_left_slices = []
        W_in_right_slices = []
        for entry in meta_input:
            rx_idx = tf.constant(entry["rx_ant_idx"], dtype=tf.int32)
            tx_idx_list = entry["tx_ant_idx"]
            if not tx_idx_list:
                raise ValueError("Empty tx index list in meta input")
            start = tx_idx_list[0] * self.window_length
            end = (tx_idx_list[-1] + 1) * self.window_length
            tx_idx = tf.constant(np.arange(start, end, dtype=np.int32), dtype=tf.int32)
            W_in_left_slices.append(tf.gather(self.W_in_left_tf, rx_idx, axis=1))
            W_in_right_slices.append(tf.gather(self.W_in_right_tf, tx_idx, axis=0))
        return W_in_left_slices, W_in_right_slices


    def _build_adjacency_matrix_from_weights(self, weights):
        adjacency = tf.zeros((self.N_v, self.N_v), dtype=tf.float32)
        adjacency = tf.tensor_scatter_nd_update(adjacency, self.lower_tri_indices_tf, weights)
        adjacency = adjacency + tf.transpose(adjacency)
        adjacency = tf.nn.relu(adjacency)
        adjacency = tf.linalg.set_diag(adjacency, tf.zeros(self.N_v, dtype=tf.float32))
        adjacency = tf.math.divide_no_nan(adjacency, tf.reduce_max(adjacency) + 1e-6)
        adjacency = 0.5 * (adjacency + tf.transpose(adjacency))
        adjacency = tf.linalg.set_diag(adjacency, tf.ones(self.N_v, dtype=tf.float32))
        return adjacency


    def _compute_training_loss_tf(self, Y_inputs_tf, Y_targets_tf, W_in_left_slices, W_in_right_slices, adjacency_matrix):
        S_3D_transit = self.state_transit_tf(
            Y_inputs_tf,
            W_in_left_slices,
            W_in_right_slices,
            adjacency_matrix,
        )
        target_vec, pred_vec = self._predict_from_states_tf(
            S_3D_transit,
            Y_inputs_tf,
            Y_targets_tf,
        )
        return self.cal_nmse_tf(target_vec, pred_vec)


    def _predict_from_states_tf(self, S_3D_transit, Y_inputs_tf, Y_targets_tf):
        predictions_flat = []
        targets_flat = []
        T = tf.shape(S_3D_transit)[0]

        for v in range(self.N_v):
            S_transit_v = S_3D_transit[:, v, :]
            Yin_v_flat = tf.reshape(Y_inputs_tf[v], (T, -1))
            S_v_time = tf.concat([S_transit_v, Yin_v_flat], axis=-1)
            S_v = tf.transpose(S_v_time)

            target_v = tf.reshape(Y_targets_tf[v], (T, -1))
            Y_v = tf.transpose(target_v)

            W_out_v = tf.matmul(Y_v, self.reg_p_inv_tf(S_v))
            Y_hat_v = tf.matmul(W_out_v, S_v)

            predictions_flat.append(tf.reshape(tf.transpose(Y_hat_v), [-1]))
            targets_flat.append(tf.reshape(tf.transpose(Y_v), [-1]))

        pred_all = tf.concat(predictions_flat, axis=0)
        target_all = tf.concat(targets_flat, axis=0)
        return target_all, pred_all


    def state_transit_tf(self, Y_inputs_tf, W_in_left_slices, W_in_right_slices, adjacency_matrix):
        T = Y_inputs_tf[0].shape[0]
        S_prev = tf.zeros((self.N_v, self.d_left, self.d_right), dtype=self.tf_dtype)
        states_over_time = []

        adjacency_complex = tf.cast(adjacency_matrix, self.tf_dtype)

        for t in range(T):
            neighbor_states = []
            for u in range(self.N_v):
                state_u = S_prev[u, ...]
                transformed = tf.matmul(self.W_N_left_blocks_tf[u], state_u)
                transformed = tf.matmul(transformed, self.W_N_right_blocks_tf[u])
                neighbor_states.append(transformed)
            neighbor_states = tf.stack(neighbor_states, axis=0)

            new_states = []
            step_features = []
            for v in range(self.N_v):
                input_contrib = tf.matmul(
                    tf.matmul(W_in_left_slices[v], Y_inputs_tf[v][t, ...]),
                    W_in_right_slices[v],
                )
                weights = adjacency_complex[v, :]
                neighborhood_contrib = tf.tensordot(weights, neighbor_states, axes=1)
                S_new = self.complex_tanh_tf(input_contrib + neighborhood_contrib)
                new_states.append(S_new)
                step_features.append(tf.reshape(S_new, [-1]))

            S_prev = tf.stack(new_states, axis=0)
            states_over_time.append(tf.stack(step_features, axis=0))

        return tf.stack(states_over_time, axis=0)


    def reg_p_inv_tf(self, X):
        X = tf.cast(X, self.tf_dtype)
        F = tf.shape(X)[0]
        identity = tf.eye(F, dtype=self.tf_dtype)
        gram = tf.matmul(X, tf.transpose(tf.math.conj(X))) + tf.cast(self.reg, self.tf_dtype) * identity
        gram_inv = tf.linalg.inv(gram)
        return tf.matmul(tf.transpose(tf.math.conj(X)), gram_inv)


    def cal_nmse_tf(self, H, H_hat):
        H = tf.cast(H, self.tf_dtype)
        H_hat = tf.cast(H_hat, self.tf_dtype)
        mse = tf.reduce_sum(tf.abs(H - H_hat) ** 2)
        denom = tf.reduce_sum((tf.abs(H) + tf.abs(H_hat)) ** 2) + 1e-12
        return tf.math.real(mse / denom)


    def complex_tanh(self, Y):
        return np.tanh(np.real(Y)) + 1j * np.tanh(np.imag(Y))


    def complex_tanh_tf(self, Y):
        real = tf.math.tanh(tf.math.real(Y))
        imag = tf.math.tanh(tf.math.imag(Y))
        return tf.complex(real, imag)