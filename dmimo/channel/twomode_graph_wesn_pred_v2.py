import numpy as np
import tensorflow as tf

from dmimo.config import Ns3Config, RCConfig
from dmimo.channel import lmmse_channel_estimation

class twomode_graph_wesn_pred_v2:

    def __init__(self, rc_config, num_freq_re, num_rx_ant, num_tx_ant, adjacency_method=None, type=np.complex64):
        
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

        self.adjacency_method = adjacency_method

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

        self._build_vertex_metadata()

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

        self._precompute_input_blocks()
        self._precompute_neighbor_blocks()

    def _build_vertex_metadata(self):

        self.tx_window_factor = self.window_length if self.enable_window else 1

        rx_nodes = self._node_slices(self.N_r, first_node_ants=4, rest_node_ants=2)
        tx_nodes = self._node_slices(self.N_t, first_node_ants=4, rest_node_ants=2)

        self.vertex_meta = []
        max_rx = 0
        max_tx = 0
        max_skip = 0
        max_target = 0

        for r_i, r_idx in enumerate(rx_nodes):
            rx_idx = np.asarray(r_idx, dtype=int)
            for t_i, t_idx in enumerate(tx_nodes):
                tx_idx = np.asarray(t_idx, dtype=int)
                tx_rows = self._compute_tx_rows(tx_idx)

                pad_rx_slice = slice(0, rx_idx.size)
                pad_tx_slice = slice(0, tx_rows.size)

                skip_dim = rx_idx.size * tx_rows.size
                target_dim = rx_idx.size * tx_idx.size

                meta_entry = {
                    "rx_node": r_i,
                    "rx_idx": rx_idx,
                    "tx_idx": tx_idx,
                    "tx_node": t_i,
                    "tx_rows": tx_rows,
                    "n_rx": rx_idx.size,
                    "n_tx": tx_idx.size,
                    "pad_rx_slice": pad_rx_slice,
                    "pad_tx_slice": pad_tx_slice,
                    "skip_dim": skip_dim,
                    "target_dim": target_dim,
                }
                self.vertex_meta.append(meta_entry)

                max_rx = max(max_rx, rx_idx.size)
                max_tx = max(max_tx, tx_idx.size)
                max_skip = max(max_skip, skip_dim)
                max_target = max(max_target, target_dim)

        self.max_rx_block = max_rx
        self.max_tx_ant = max_tx
        self.max_tx_block = max_tx * self.tx_window_factor

        self.state_dim = self.d_left * self.d_right
        self.max_skip_dim = max_skip
        self.max_target_dim = max_target
        self.max_feature_dim = self.state_dim + self.max_skip_dim

        for entry in self.vertex_meta:
            skip_dim = entry["skip_dim"]
            target_dim = entry["target_dim"]
            entry["feature_dim"] = self.state_dim + skip_dim
            entry["feature_slice"] = slice(0, entry["feature_dim"])
            entry["target_slice"] = slice(0, target_dim)

    def _compute_tx_rows(self, tx_idx_array):

        tx_idx_array = np.asarray(tx_idx_array, dtype=int)
        if self.tx_window_factor == 1:
            return tx_idx_array.copy()

        rows = [
            np.arange(idx * self.tx_window_factor, (idx + 1) * self.tx_window_factor, dtype=int)
            for idx in tx_idx_array
        ]
        if rows:
            return np.concatenate(rows, axis=0)
        return np.asarray([], dtype=int)

    def _precompute_input_blocks(self):

        max_rx = getattr(self, "max_rx_block", 0)
        max_tx = getattr(self, "max_tx_block", 0)

        if max_rx == 0 or max_tx == 0:
            self.W_in_left_blocks = np.zeros((self.N_v, self.d_left, 0), dtype=self.dtype)
            self.W_in_right_blocks = np.zeros((self.N_v, 0, self.d_right), dtype=self.dtype)
            return

        W_in_left_blocks = np.zeros((self.N_v, self.d_left, max_rx), dtype=self.W_in_left.dtype)
        W_in_right_blocks = np.zeros((self.N_v, max_tx, self.d_right), dtype=self.W_in_right.dtype)

        for v, info in enumerate(self.vertex_meta):
            n_rx = info["n_rx"]
            if n_rx:
                W_in_left_blocks[v, :, :n_rx] = self.W_in_left[:, info["rx_idx"]]

            tx_rows = info["tx_rows"]
            if tx_rows.size:
                W_in_right_blocks[v, :tx_rows.size, :] = self.W_in_right[tx_rows, :]

        self.W_in_left_blocks = W_in_left_blocks.astype(self.dtype, copy=False)
        self.W_in_right_blocks = W_in_right_blocks.astype(self.dtype, copy=False)

    def _precompute_neighbor_blocks(self):

        self.W_N_left_blocks = np.stack(
            [
                self.W_N_left[:, u * self.d_left : (u + 1) * self.d_left]
                for u in range(self.N_v)
            ],
            axis=0,
        ).astype(self.dtype, copy=False)

        self.W_N_right_blocks = np.stack(
            [
                self.W_N_right[u * self.d_right : (u + 1) * self.d_right, :]
                for u in range(self.N_v)
            ],
            axis=0,
        ).astype(self.dtype, copy=False)
    
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
        
        if self.adjacency_method is None:
            self.adjacency = self.compute_None_adjacency()
        elif self.adjacency_method == 'eigenmode_cov':
            self.adjacency = self.compute_eigenmode_adjacency_cov(h_freq_csi_history)
        else:
            raise ValueError(f"Unsupported adjacency_method: {self.adjacency_method}")

        chan_pred = np.zeros(h_freq_csi_history[0,...].shape, dtype=self.dtype)

        total_chunks = num_freq_res * num_ofdm_syms
        features_accum = None
        targets_accum = None
        offset = 0

        # --------- (A) FEATURE BUILD PHASE: stack all RBs (and OFDM syms) ----------
        for freq_re in range(num_freq_res):
            for ofdm_sym in range(num_ofdm_syms):
                channel_train_input_array, meta_input = self.extract_tx_rx_node_pairs_numpy(
                    channel_train_input[..., freq_re, ofdm_sym]
                )
                channel_train_gt_array, _ = self.extract_tx_rx_node_pairs_numpy(
                    channel_train_gt[..., freq_re, ofdm_sym]
                )

                features_chunk, targets_chunk = self.build_S_Y(
                    channel_train_input_array,
                    channel_train_gt_array,
                    meta_input,
                    curr_window_weights=None,
                )

                T_chunk = features_chunk.shape[-1]
                if features_accum is None:
                    total_T = T_chunk * total_chunks
                    features_accum = np.zeros(
                        (self.N_v, self.max_feature_dim, total_T),
                        dtype=self.dtype,
                    )
                    targets_accum = np.zeros(
                        (self.N_v, self.max_target_dim, total_T),
                        dtype=self.dtype,
                    )

                features_accum[:, :, offset:offset + T_chunk] = features_chunk
                targets_accum[:, :, offset:offset + T_chunk] = targets_chunk
                offset += T_chunk

        if features_accum is None:
            raise RuntimeError("No training chunks were processed")

        total_T_actual = offset
        features_accum = features_accum[:, :, :total_T_actual]
        targets_accum = targets_accum[:, :, :total_T_actual]

        self.W_out_matrix = np.zeros(
            (self.N_v, self.max_target_dim, self.max_feature_dim),
            dtype=self.dtype,
        )

        for v, info in enumerate(self.vertex_meta):
            feature_slice = info["feature_slice"]
            target_slice = info["target_slice"]

            S_v = features_accum[v, feature_slice, :]
            Y_v = targets_accum[v, target_slice, :]
            G_v = self.reg_p_inv(S_v)
            W_out_v = Y_v @ G_v
            self.W_out_matrix[v, target_slice, feature_slice] = W_out_v.astype(self.dtype, copy=False)

        # --------- (C) PREDICTION PHASE with per-vertex W_out ----------
        chan_pred = np.squeeze(np.zeros(h_freq_csi_history[0, ...].shape, dtype=self.dtype))

        for freq_re in range(num_freq_res):
            for ofdm_sym in range(num_ofdm_syms):
                channel_test_input_array, meta_test = self.extract_tx_rx_node_pairs_numpy(
                    channel_train_gt[..., freq_re, ofdm_sym]
                )

                Y_3D_win = self.form_window_input_signal_list(channel_test_input_array, window_weights=None)
                Y_3D_win = Y_3D_win * self.input_scale

                S_3D_transit = self.state_transit(Y_3D_win, meta_test)
                T_test = S_3D_transit.shape[0]
                t_last = T_test - 1

                H_hat_full = np.zeros((self.N_r, self.N_t), dtype=self.dtype)

                for v, info in enumerate(self.vertex_meta):
                    feature_slice = info["feature_slice"]
                    target_slice = info["target_slice"]
                    n_rx_v = info["n_rx"]
                    n_tx_v = info["n_tx"]
                    expected_tx = info["pad_tx_slice"].stop

                    S_transit_v = S_3D_transit[t_last, v, :]
                    Yin_v = Y_3D_win[v, t_last, :n_rx_v, :expected_tx]
                    Yin_v_flat = Yin_v.reshape(-1)
                    feature_vec = np.concatenate([S_transit_v, Yin_v_flat], axis=0)

                    W_out_v = self.W_out_matrix[v, target_slice, feature_slice]
                    y_last = W_out_v @ feature_vec

                    H_v = y_last.reshape(n_rx_v, n_tx_v, order='C')
                    H_hat_full[np.ix_(info["rx_idx"], info["tx_idx"])] = H_v

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
        """Extract per-vertex channel blocks as a zero-padded array."""

        if h_freq_csi_history.ndim != 6:
            raise ValueError("Expected shape (B,1,1,R,1,T)")

        T = h_freq_csi_history.shape[0]

        padded_pairs = np.zeros(
            (self.N_v, T, self.max_rx_block, self.max_tx_ant),
            dtype=self.dtype,
        )
        meta = []

        for v, info in enumerate(self.vertex_meta):
            rx_idx = info["rx_idx"]
            tx_idx = info["tx_idx"]

            block = h_freq_csi_history[:, 0, 0, rx_idx, 0, ...]
            block = block[:, :, tx_idx]

            n_rx = info["n_rx"]
            n_tx = info["n_tx"]

            padded_pairs[v, :, :n_rx, :n_tx] = block

            meta.append({
                "rx_node": info["rx_node"],
                "rx_ant_idx": rx_idx.tolist(),
                "tx_node": info["tx_node"],
                "tx_ant_idx": tx_idx.tolist(),
                "block_shape": tuple(block.shape),
            })

        return padded_pairs, meta


    def build_S_Y(self, channel_input, channel_output, meta_input, curr_window_weights):
        if self.adjacency is None:
            raise RuntimeError("Adjacency matrix has not been computed before build_S_Y call.")

        if channel_input.ndim != 4 or channel_input.shape[0] != self.N_v:
            raise ValueError("channel_input must be (N_v, T, max_rx, max_tx_ant)")
        if channel_output.shape != channel_input.shape:
            raise ValueError("channel_output must match channel_input shape")

        Y_3D_win = self.form_window_input_signal_list(channel_input, curr_window_weights)
        Y_3D_win = Y_3D_win * self.input_scale

        S_3D_transit = self.state_transit(Y_3D_win, meta_input)
        T = S_3D_transit.shape[0]

        features = np.zeros((self.N_v, self.max_feature_dim, T), dtype=self.dtype)
        targets = np.zeros((self.N_v, self.max_target_dim, T), dtype=self.dtype)

        for v, info in enumerate(self.vertex_meta):
            n_rx = info["n_rx"]
            n_tx = info["n_tx"]
            expected_tx = info["pad_tx_slice"].stop

            S_transit_v = S_3D_transit[:, v, :]  # (T, state_dim)
            Yin_v = Y_3D_win[v, :, :n_rx, :expected_tx]
            Yin_v_flat = Yin_v.reshape(T, -1)
            S_v_time_major = np.concatenate([S_transit_v, Yin_v_flat], axis=-1)

            feature_slice = info["feature_slice"]
            features[v, feature_slice, :] = S_v_time_major.T.astype(self.dtype, copy=False)

            Yv = channel_output[v, :, :n_rx, :n_tx]
            Yv_flat = Yv.reshape(T, -1, order='C')
            target_slice = info["target_slice"]
            targets[v, target_slice, :] = Yv_flat.T.astype(self.dtype, copy=False)

        return features, targets


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
        pairs_array,
        window_weights=None,
    ):
        if pairs_array.ndim != 4 or pairs_array.shape[0] != self.N_v:
            raise ValueError("pairs_array must be (N_v, T, max_rx, max_tx)")

        if self.enable_window:
            L = int(self.window_length)
            if L <= 0:
                raise ValueError("window_length must be >= 1")
        else:
            L = 1

        out = np.zeros(
            (self.N_v, pairs_array.shape[1], self.max_rx_block, self.max_tx_block),
            dtype=self.dtype,
        )

        for v, info in enumerate(self.vertex_meta):
            n_rx = info["n_rx"]
            n_tx = info["n_tx"]

            if n_rx == 0 or n_tx == 0:
                raise ValueError("Each vertex must select at least one Rx and Tx antenna")

            data = pairs_array[v, :, :n_rx, :n_tx].astype(self.dtype, copy=False)

            if self.enable_window:
                Y_win = np.zeros((data.shape[0], n_rx, L * n_tx), dtype=self.dtype)
                zero_block = np.zeros((n_rx, n_tx), dtype=self.dtype)
                for k in range(data.shape[0]):
                    for ell in range(L):
                        t = k - ell
                        block = data[t] if t >= 0 else zero_block
                        start = ell * n_tx
                        end = start + n_tx
                        Y_win[k, :, start:end] = block
                padded = Y_win
            else:
                padded = data

            out[v, :, info["pad_rx_slice"], info["pad_tx_slice"]] = padded

        return out


    def state_transit(self, Y_3D, meta):

        if Y_3D.ndim != 4 or Y_3D.shape[0] != self.N_v:
            raise ValueError("Y_3D must be (N_v, T, max_rx, max_tx)")

        if meta is not None and len(meta) != self.N_v:
            raise ValueError("meta length does not match number of vertices")

        T = Y_3D.shape[1]

        for v, info in enumerate(self.vertex_meta):
            n_rx_v = info["n_rx"]
            expected_tx = info["pad_tx_slice"].stop
            if n_rx_v == 0 or expected_tx == 0:
                raise ValueError("Each vertex must select at least one Rx and Tx antenna")
            slice_view = Y_3D[v, :, :n_rx_v, :expected_tx]
            if slice_view.shape[2] != expected_tx:
                raise ValueError("Windowed input width mismatch for vertex %d" % v)
            if meta is not None:
                entry = meta[v]
                if entry["rx_ant_idx"] != info["rx_idx"].tolist() or entry["tx_ant_idx"] != info["tx_idx"].tolist():
                    raise ValueError("Runtime metadata does not match cached vertex configuration")

        Y_pad_transposed = Y_3D.astype(self.dtype, copy=False)
        left_mult = np.matmul(self.W_in_left_blocks[:, None, :, :], Y_pad_transposed)
        input_contrib = np.matmul(left_mult, self.W_in_right_blocks[:, None, :, :])
        input_contrib = np.transpose(input_contrib, (1, 0, 2, 3))  # (T, N_v, d_left, d_right)

        # ------------------------------------------------------------------
        # Vectorized neighbor updates
        # ------------------------------------------------------------------
        S_3D = self.S_0.copy()

        S_3D_list = []

        for t in range(T):
            transformed = np.matmul(S_3D, self.W_N_right_blocks)
            neighbor_states = np.matmul(self.W_N_left_blocks, transformed)
            neighborhood_contrib = np.tensordot(self.adjacency, neighbor_states, axes=([1], [0]))

            S_3D = self.complex_tanh(input_contrib[t] + neighborhood_contrib).astype(self.dtype, copy=False)
            S_3D_list.append(S_3D.reshape(self.N_v, -1))

        self.S_0 = S_3D.copy()

        S_3D = np.stack(S_3D_list, axis=0).astype(self.dtype, copy=False)

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
        h_freq_csi_history,
        subspace_rank_tx=2, # q_t
        subspace_rank_rx=2, # q_r
        center: bool = True,
        unbiased: bool = True,
    ):

        B, _, _, Rx, _, Tx = h_freq_csi_history[..., 0, 0].shape

        rx_nodes = self._node_slices(Rx, first_node_ants=4, rest_node_ants=2)
        tx_nodes = self._node_slices(Tx, first_node_ants=4, rest_node_ants=2)

        per_link_channels = []
        meta  = []

        for r_i, r_idx in enumerate(rx_nodes):
            for t_i, t_idx in enumerate(tx_nodes):
                # Slice: (B, r, t)
                block = h_freq_csi_history[:, 0, 0, r_idx, 0, ...]
                block = block[:, :, t_idx, ...]
                per_link_channels.append(block)
                meta.append({
                    "rx_node": r_i,
                    "rx_ant_idx": r_idx.tolist(),
                    "tx_node": t_i,
                    "tx_ant_idx": t_idx.tolist(),
                    "block_shape": tuple(block.shape)
                })

        Tx_bases, Rx_bases = [], []

        for H in per_link_channels:

            T, Rx, Tx, S, R = H.shape

            Hk = np.reshape(H, (T, Rx, Tx, S * R))
            Hk = np.moveaxis(Hk, (0, 3), (0, 1))  # (T, Rx, Tx, SR) -> (T, SR, Rx, Tx)
            Hk = np.reshape(Hk, (T * S * R, Rx, Tx))  # (K, Rx, Tx)

            K = Hk.shape[0]

            if center:
                mu = Hk.mean(axis=0, keepdims=True)  # (1, Rx, Tx)
                Hk = Hk - mu

            denom = float((K - 1) if (unbiased and K > 1) else K)

            # --- Receive-side covariance: R_rx = E[ H H^H ] ---
            # Compute per-snapshot products (K, Rx, Rx), then average over K
            Hk_H = np.transpose(Hk.conj(), (0, 2, 1))        # (K, Tx, Rx)
            prod_rx = Hk @ Hk_H                               # (K, Rx, Rx)
            R_rx = prod_rx.sum(axis=0) / denom                # (Rx, Rx)

            # --- Transmit-side covariance: R_tx = E[ H^H H ] ---
            HkH = Hk.conj().transpose(0, 2, 1)               # (K, Tx, Rx)
            prod_tx = HkH @ Hk                                # (K, Tx, Tx)
            R_tx = prod_tx.sum(axis=0) / denom                # (Tx, Tx)

            # Eigen-decomp (Hermitian), take top-q
            eval_tx, evec_tx = np.linalg.eigh(R_tx)
            eval_rx, evec_rx = np.linalg.eigh(R_rx)
            idx_tx = np.argsort(eval_tx)[::-1]
            idx_rx = np.argsort(eval_rx)[::-1]
            Ut = evec_tx[:, idx_tx[:subspace_rank_tx]] 
            Ur = evec_rx[:, idx_rx[:subspace_rank_rx]]

            # QR for numerical stability (keeps orthonormal columns)
            Ut, _ = np.linalg.qr(Ut)
            Ur, _ = np.linalg.qr(Ur)
                

            Tx_bases.append(Ut)  # (T_sel x q_t)
            Rx_bases.append(Ur)  # (R_sel x q_r)

        def pad_rows(Q, N_target):
            """Zero-pad rows of Q (n x q) to (N_target x q)."""
            n, q = Q.shape
            if n == N_target:
                return Q
            Qp = np.zeros((N_target, q), dtype=Q.dtype)
            Qp[:n, :] = Q
            return Qp


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

        adj = np.zeros((self.N_v, self.N_v), dtype=np.float64)
        for a in range(self.N_v):
            for b in range(self.N_v):
                adj[a, b] = joint_affinity(Rx_bases[a], Tx_bases[a], Rx_bases[b], Tx_bases[b])

        # Force symmetry and self-loops
        adj = 0.5 * (adj + adj.T)
        np.fill_diagonal(adj, 1.0)

        adj = np.round(adj)

        return adj

    def complex_tanh(self, Y):
        return np.tanh(np.real(Y)) + 1j * np.tanh(np.imag(Y))