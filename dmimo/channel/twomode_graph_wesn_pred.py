import copy
import numpy as np
import tensorflow as tf

from dmimo.config import Ns3Config, RCConfig
from dmimo.channel import lmmse_channel_estimation

class twomode_graph_wesn_pred:

    def __init__(self, rc_config, num_freq_re, num_rx_ant, num_tx_ant, type=np.complex64):
        
        self.rc_config = rc_config
        ns3_config = Ns3Config()

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
        self.num_tx_nodes = int((self.N_t - ns3_config.num_bs_ant)  / ns3_config.num_ue_ant) + 1
        self.num_rx_nodes = int((self.N_r - ns3_config.num_bs_ant)  / ns3_config.num_ue_ant) + 1
        self.N_v = self.num_rx_nodes * self.num_tx_nodes # number of vertices in the graph
        self.N_e = int((self.N_v*(self.N_v-1))/2) # number of edges in the graph (at most. some of them will be zeroed out)

        self.N_in_left = self.N_r
        if self.enable_window:
            self.N_in_right = self.N_t * self.window_length # TODO: only windowing on the transmit antenna axis for now. evaluate windowing on the receive antenna axis later
        else:
            self.N_in_right = self.N_t

        self.d_left = self.N_in_left # TODO: currently just basing on the size of the input. try other configurations
        self.d_right = self.N_in_right

        if self.d_left is None:
            self.d_left = self.N_r
        if self.d_right is None:
            self.d_right = self.N_t        

        self.init_weights()

    def init_weights(self):

        matrices_left = []
        matrices_right = []
        for _ in range(self.N_v):
            result = self.sparse_mat(self.d_left)
            matrices_left.append(result)
            result = self.sparse_mat(self.d_right)
            matrices_right.append(result)
        
        self.W_N_left = tf.Variable(np.concatenate(matrices_left, axis=1), trainable=False)
        self.W_N_right = tf.Variable(np.concatenate(matrices_right, axis=1), trainable=False)
        
        self.W_res_left = self.sparse_mat(self.d_left)
        self.W_res_right = self.sparse_mat(self.d_right)

        self.W_in_left = 2 * (self.RS.rand(self.d_left, self.N_in_left) - 0.5) # TODO: check if I should make this complex later
        self.W_in_right = 2 * (self.RS.rand(self.N_in_right, self.d_right) - 0.5) # TODO: check if I should make this complex later

        # TODO: using a vectorization trick to learn one vectorized W_out instead of left and right W_outs.
        # This is mathematically equivalent to 
        # self.W_out_left = self.RS.randn(self.N_r, self.d_left)
        # self.W_out_right = self.RS.randn(self.d_right + self.N_in_right, self.N_t)
        # and multiplying the two W_outs on either side of the feature queue 
        self.feature_dim = int(2 * self.d_left * self.d_right * (self.window_length + 1))
        self.W_out = self.RS.randn(self.N_r * self.N_t, self.feature_dim).astype(self.dtype)        

        self.S_0_rx = np.zeros([self.N_v, self.d_left, self.d_right], dtype=self.dtype)
        self.S_0_tx = np.zeros([self.N_v, self.d_left, self.d_right], dtype=self.dtype)
    
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
                self.S_0_rx = np.zeros([self.d_left, self.d_right], dtype=self.dtype)
                self.S_0_tx = np.zeros([self.d_left, self.d_right], dtype=self.dtype)

                S_f, Y_f = self.build_S_Y(channel_train_input_list, channel_train_gt_list, curr_window_weights=None)
                S_list.append(S_f); Y_list.append(Y_f)
        
        S_all = np.concatenate(S_list, axis=1)  # (F, sum_T)
        Y_all = np.concatenate(Y_list, axis=1)  # (N_r*N_t, sum_T)

        # --------- (B) SINGLE READOUT SOLVE (shared across RBs) ----------
        # Prefer ridge for stability:
        G = self.reg_p_inv(S_all)               # (sum_T, F)  :=  S_all^H (S_all S_all^H + λI)^{-1}
        self.W_out = Y_all @ G                  # (N_r*N_t, F)

        # --------- (C) PREDICTION PHASE with the shared W_out ----------
        for freq_re in range(num_freq_res):
            for ofdm_sym in range(num_ofdm_syms):
                # Use last known channel as test input; predict next step
                channel_test_input = channel_train_gt[:, 0, tx_node, :, rx_node, :, freq_re, ofdm_sym]

                # Optional: either carry S_0 across RBs for smoothness,
                # or reset it per RB. Start with reset; then try carry-over.
                self.S_0 = np.zeros([self.d_left, self.d_right], dtype=self.dtype)

                channel_pred_temp = self.test_train_predict(channel_test_input, curr_window_weights=None)
                channel_pred_temp = channel_pred_temp[:, :, -1:]       # keep last step
                channel_pred_temp = np.squeeze(channel_pred_temp)      # [N_r, N_t]
                chan_pred[:, tx_node, :, rx_node, :, freq_re, ofdm_sym] = channel_pred_temp

                
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


    def build_S_Y(self, channel_input, channel_output, curr_window_weights):
        # channel_input, channel_output: [T, N_r, N_t]
        Y_3D_list = channel_input
        Y_target_3D_list = channel_output

        if self.enable_window:
            Y_3D_win_list = self.form_window_input_signal_list(Y_3D_list, curr_window_weights)
        else:
            # TODO: not adapted to graph input yet
            # Safe fallback if forget_length not set:
            forget = getattr(self, "forget_length", 0)
            Y_3D_win = np.concatenate([Y_3D, np.zeros([Y_3D.shape[0], forget, Y_3D.shape[2]], dtype=self.dtype)], axis=1)

        Y_3D_win_list = [arr * self.input_scale for arr in Y_3D_win_list]
        S_3D_transit = self.state_transit(Y_3D_win_list)
        S_3D = np.concatenate([S_3D_transit, Y_3D_win], axis=-1)

        T = S_3D.shape[0]
        S = np.column_stack([S_3D[t].reshape(-1, order='C') for t in range(T)])  # (feature_dim, T)
        Y = np.column_stack([Y_target_3D[t].reshape(-1, order='C') for t in range(T)])  # (N_r*N_t, T)
        return S, Y


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
        H_hat = tf.cast(H_hat, dtype=H.dtype)
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

    def test_train_predict(self, channel_train_input, curr_window_weights):
        self.S_0 = np.zeros([self.d_left, self.d_right], dtype=self.dtype)

        Y_3D_org = channel_train_input

        Y_3D = self.form_window_input_signal_list(Y_3D_org, curr_window_weights)

        S_3D = self.state_transit(Y_3D * self.input_scale)

        S_3D = np.concatenate([S_3D, Y_3D], axis=-1)

        # vectorization trick. equivalent to having two W_out matrices on either side of the feature matrix being fed to the output
        T = S_3D.shape[0]
        S = np.column_stack([
            S_3D[t].reshape(-1, order='C') for t in range(T)
        ])  # (feature_dim, T)

        curr_channel_pred = self.W_out @ S

        curr_channel_pred = curr_channel_pred.reshape([self.N_r, self.N_t, -1])

        return curr_channel_pred

    def state_transit(self, Y_3D_list):

        T = Y_3D_list[0].shape[0] # number of samples

        S_3D_rx = copy.deepcopy(self.S_0_rx)
        S_3D_tx = copy.deepcopy(self.S_0_tx)
        
        S_4D = []
        for t in range(T):
            S_3D = []
            for v in range(self.N_v):

                S_2D = self.complex_tanh(self.W_res_left @ S_2D @ self.W_res_right + self.W_in_left @ Y_3D[t,:,:] @ self.W_in_right)
                S_3D.append(S_2D)
            S_4D.append(np.stack(S_3D, axis=0))

        S_4D = np.stack(S_4D, axis=0)

        return S_4D

    def complex_tanh(self, Y):
        return np.tanh(np.real(Y)) + 1j * np.tanh(np.imag(Y))