import copy
import numpy as np
import tensorflow as tf

from dmimo.config import Ns3Config, RCConfig
from dmimo.channel import lmmse_channel_estimation
import matplotlib.pyplot as plt
import itertools

class gesn_pred_freq_mimo:

    def __init__(self, architecture, len_features=None, num_rx_ant=8, num_tx_ant=8, max_adjacency='all', method='per_node_pair', num_neurons=None):
        
        ns3_config = Ns3Config()
        self.rc_config = RCConfig()

        self.syms_per_subframe = 14
        self.nfft = 512  # TODO: remove hardcoded param value
        self.subcarriers_per_RB = 12
        self.N_RB = int(np.ceil(self.nfft / self.subcarriers_per_RB))
        self.num_rx_ant = num_rx_ant

        self.num_bs_ant = ns3_config.num_bs_ant
        self.num_ue_ant = ns3_config.num_ue_ant
        
        if architecture == 'baseline':
            self.N_t = ns3_config.num_bs_ant
            self.N_r = ns3_config.num_bs_ant
        elif architecture == 'SU_MIMO':
            self.N_t = num_tx_ant
            self.N_r = ns3_config.num_bs_ant * 2
        elif architecture == 'MU_MIMO':
            self.N_t = num_tx_ant
            self.N_r = num_rx_ant
        else:
            raise ValueError("\n The architecture specified is not defined")

        self.sparsity = self.rc_config.W_tran_sparsity
        self.spectral_radius = self.rc_config.W_tran_radius
        self.max_forget_length = self.rc_config.max_forget_length
        self.initial_forget_length = self.rc_config.initial_forget_length
        self.forget_length = self.rc_config.initial_forget_length
        self.forget_length_search_step = self.rc_config.forget_length_search_step
        self.input_scale = self.rc_config.input_scale
        self.window_length = self.rc_config.window_length
        self.learning_delay = self.rc_config.learning_delay
        self.reg = self.rc_config.regularization
        self.enable_window = self.rc_config.enable_window
        self.history_len = self.rc_config.history_len

        seed = 10
        self.RS = np.random.RandomState(seed)
        self.type = self.rc_config.type # 'real', 'complex'

        # Calculate weight matrix dimensions
        if method == 'per_antenna_pair':
            # one antenna pair is one vertex
            raise ValueError("\n The GESN method specified has not been completely implemented")

        elif method == 'per_node_pair':
            # one tx-rx node pair is one vertex
            self.num_tx_nodes = int((self.N_t - ns3_config.num_bs_ant)/ns3_config.num_ue_ant) + 1
            self.num_rx_nodes = int((self.N_r - ns3_config.num_bs_ant)/ns3_config.num_ue_ant) + 1
            self.N_v = self.num_tx_nodes * self.num_rx_nodes                                            # number of vertices in the graph
            if self.rc_config.treatment == 'SISO':
                self.N_f = self.N_RB                                                                        # length of feature vector for each vertex
            else:
                if len_features == None:
                    self.N_f = self.N_RB * ns3_config.num_bs_ant * ns3_config.num_bs_ant                    # length of feature vector for each vertex
                else:
                    self.N_f = len_features                                                                 # length of feature vector for each vertex

        else:
            raise ValueError("\n The GESN method specified is not defined")
        if num_neurons is None:
            self.N_n = self.rc_config.num_neurons * self.N_v
            self.N_n_per_vertex = self.rc_config.num_neurons
        else:
            self.N_n = num_neurons * self.N_v
            self.N_n_per_vertex = num_neurons

        if self.enable_window:
            self.N_in = self.N_f * self.window_length
        else:
            self.N_in = self.N_f
        self.N_out = self.N_f * self.N_v
        self.S_0 = np.zeros([self.N_n], dtype='complex')

        # Initialize adjacency matrix (currently static for all time steps)
        if max_adjacency == 'all':
            self.max_adjacency = self.N_v
        elif max_adjacency == 'k_nearest_neighbors':
            raise ValueError("\n The knn clustering method has not yet been implemented")
        else:
            self.max_adjacency = max_adjacency

        # Initialize weight matrices
        self.init_weights()

        self.train_rls = False
        self.DF_rls = self.rc_config.DF_rls

        if self.train_rls or self.DF_rls:
            # for RLS algorithm
            self.psi = np.identity(self.N_in + self.N_n)
            if self.type == 'complex':
                self.psi = self.psi.astype(complex)
            self.psi_inv = np.linalg.inv(self.psi)

            # self.RLS_lambda = 0.9995, 0.9998
            # self.RLS_w = 1 / (1 + np.exp(-(self.EbNo - 11)))

            # self.RLS_lambda = 0.99999
            self.RLS_lambda = self.rc_config.RLS_lambda
            self.RLS_w = 1

    def get_csi_history(self, first_slot_idx, csi_delay, rg_csi, dmimo_chans):

        first_csi_history_idx = first_slot_idx - (csi_delay * self.history_len)
        channel_history_slots = np.arange(first_csi_history_idx, first_slot_idx, csi_delay)

        h_freq_csi_list = []
        for loop_idx, slot_idx in enumerate(channel_history_slots):
            # h_freq_csi has shape [batch_size, num_rx, num_rx_ant, num_tx, num_txs_ant, num_ofdm_sym, fft_size]
            h_freq_csi, err_var_csi = lmmse_channel_estimation(dmimo_chans, rg_csi, slot_idx=slot_idx)
            # h_freq_csi = h_freq_csi[:, :, :self.num_rx_ant, ...]  # TODO: use node selection mask
            h_freq_csi_list.append(np.expand_dims(h_freq_csi, axis=0))

        h_freq_csi_history = np.concatenate(h_freq_csi_list, axis=0)

        return h_freq_csi_history

    def get_ideal_csi_history(self, first_slot_idx, csi_delay, dmimo_chans, batch_size=1):
        
        # Get channel estimate history starting from (csi_delay * self.history_len) slots in the past to the most up-to-date fed back estimate
        # Here "first_slot_idx" is used as the index of the current slot. 
        # e.g. if first_slot_idx is 12, we are in the 12th slot. 
        # if csi_delay = 6 and self.history_len = 2, then we want to use the estimates for slots 12-6x2 = 0 and 12-6x1 = 6 for training
        # and we try to predict the unknown channel for slot 12, which is the index of the current slot

        # csi_step_size = csi_delay // 2
        # first_csi_history_idx = first_slot_idx - csi_delay - csi_step_size * (self.history_len * 2 - 1)
        # channel_history_slots = np.arange(first_csi_history_idx, first_slot_idx - csi_delay + 1, csi_step_size)
        first_csi_history_idx = first_slot_idx - (csi_delay * self.history_len)  # TODO: currently only for self.history_len = 2
        channel_history_slots = np.arange(first_csi_history_idx, first_slot_idx, csi_delay)

        h_freq_csi_history_0, _ = dmimo_chans.load_channel(slot_idx=channel_history_slots[0], batch_size=batch_size)
            
        h_freq_csi_history = np.zeros(np.concatenate((channel_history_slots.shape, np.asarray(h_freq_csi_history_0.shape))), dtype=complex)
        h_freq_csi_history[0, ...] = h_freq_csi_history_0
        for loop_idx, slot_idx in enumerate(channel_history_slots[1:]):
            h_freq_csi_history[loop_idx+1, ...], _ = dmimo_chans.load_channel(slot_idx=slot_idx, batch_size=batch_size)

        return h_freq_csi_history


    def predict(self, h_freq_csi_history):

        if self.rc_config.treatment == 'SISO':
            channel_pred = self.siso_predict(h_freq_csi_history)
        elif self.rc_config.treatment == 'MIMO':
            channel_pred = self.mimo_predict(h_freq_csi_history)
        else:
            raise ValueError("\n Only SISO and MIMO treatment has been defined.")
        
        return channel_pred

    def generate_antenna_selections(self, num_transmitters, num_receivers):
        """
        Generate all possible antenna selections for transmitters and receivers and convert one-hot encoding to indices.

        :param num_transmitters: Number of transmitters
        :param num_receivers: Number of receivers
        :return: List of all possible selections with indices
        """
        # Define number of antennas for transmitters and receivers
        tx_antennas = [4 if i == 0 else 2 for i in range(num_transmitters)]
        rx_antennas = [4 if i == 0 else 2 for i in range(num_receivers)]

        # Generate one-hot encoding options for each transmitter and receiver
        tx_selections = [list(np.eye(ant, dtype=int)) for ant in tx_antennas]
        rx_selections = [list(np.eye(ant, dtype=int)) for ant in rx_antennas]
        
        # Create all possible combinations of selections
        all_combinations = itertools.product(
            itertools.product(*tx_selections),
            itertools.product(*rx_selections)
        )
        
        # Convert one-hot encodings to indices
        selections = []
        tx_offsets = [sum(tx_antennas[:i]) for i in range(num_transmitters)]
        rx_offsets = [sum(rx_antennas[:i]) for i in range(num_receivers)]

        for tx_choice, rx_choice in all_combinations:
            tx_indices = [
                offset + np.argmax(choice) for offset, choice in zip(tx_offsets, tx_choice)
            ]
            rx_indices = [
                offset + np.argmax(choice) for offset, choice in zip(rx_offsets, rx_choice)
            ]
            selections.append({
                "transmitter_indices": tx_indices,
                "receiver_indices": rx_indices
            })
        
        return selections


    def siso_predict(self, h_freq_csi_history):

        h_freq_csi_history = np.squeeze(np.asarray(h_freq_csi_history).transpose([0,1,2,3,4,5,7,6]))
        h_freq_csi_history = self.rb_mapper(h_freq_csi_history)

        num_time_steps = h_freq_csi_history.shape[0] * h_freq_csi_history.shape[-1]
        h_freq_csi_history_reshaped = np.moveaxis(h_freq_csi_history, -1, 1)
        h_freq_csi_history_reshaped = h_freq_csi_history_reshaped.reshape((num_time_steps,) + h_freq_csi_history.shape[1:-1])

        num_training_steps = (h_freq_csi_history.shape[0]-1)*self.syms_per_subframe

        antenna_selections = self.generate_antenna_selections(self.num_tx_nodes, self.num_rx_nodes)

        N_r = h_freq_csi_history_reshaped.shape[1]
        N_t = h_freq_csi_history_reshaped.shape[2]

        channel_pred = np.zeros(h_freq_csi_history_reshaped[:self.syms_per_subframe,...].shape, dtype=complex)

        # Loop over all possible graphs
        for i in range(len(antenna_selections)):
            
            # Find antenna elements of current graph
            tx_ant_idx = antenna_selections[i]['transmitter_indices']
            rx_ant_idx = antenna_selections[i]['receiver_indices']
            
            # Get input-label pair for training data and input for testing data
            curr_channels = h_freq_csi_history_reshaped[:,rx_ant_idx, :, :][:, :, tx_ant_idx, :]
            channel_train_input_list = [curr_channels[:num_training_steps, rx_idx, tx_idx, :] for rx_idx in range(len(rx_ant_idx)) for tx_idx in range(len(tx_ant_idx))]
            channel_train_gt_list = [curr_channels[-num_training_steps:, rx_idx, tx_idx, :] for rx_idx in range(len(rx_ant_idx)) for tx_idx in range(len(tx_ant_idx))]
            channel_test_input_list = [curr_channels[-self.syms_per_subframe:, rx_idx, tx_idx, :] for rx_idx in range(len(rx_ant_idx)) for tx_idx in range(len(tx_ant_idx))]

            # Train the model
            pred_channel_training = self.fitting_time(channel_train_input_list, channel_train_gt_list)

            # Generate output from trained model
            channel_pred_temp = self.test_train_predict(channel_test_input_list)

            # Store output
            for count_rx, rx in enumerate(rx_ant_idx):
                for count_tx, tx in enumerate(tx_ant_idx):
                    node_idx = count_rx * self.num_tx_nodes + count_tx

                    channel_pred[:, rx, tx, :] = channel_pred_temp[node_idx].transpose()

        channel_pred = tf.convert_to_tensor(channel_pred)
        channel_pred = tf.transpose(channel_pred, perm=[1,2,3,0])
        return channel_pred
    
    def siso_predict_tmp(self, h_freq_csi_history):

        h_freq_csi_history = np.squeeze(np.asarray(h_freq_csi_history).transpose([0,1,2,3,4,5,7,6]))
        h_freq_csi_history = self.rb_mapper(h_freq_csi_history)

        num_time_steps = h_freq_csi_history.shape[0] * h_freq_csi_history.shape[-1]
        h_freq_csi_history_reshaped = np.moveaxis(h_freq_csi_history, -1, 1)
        h_freq_csi_history_reshaped = h_freq_csi_history_reshaped.reshape((num_time_steps,) + h_freq_csi_history.shape[1:-1])

        num_training_steps = (h_freq_csi_history.shape[0]-1)*self.syms_per_subframe

        channel_train_input_list = []
        channel_train_gt_list = []
        channel_test_input_list = []

        for tx_node_idx in range(self.num_tx_nodes):
            for rx_node_idx in range(self.num_rx_nodes):
                
                if tx_node_idx == 0:
                    tx_ant_idx = np.arange(0,self.num_bs_ant)
                else:
                    tx_ant_idx = np.arange(self.num_bs_ant + (tx_node_idx-1)*self.num_ue_ant,self.num_bs_ant + (tx_node_idx)*self.num_ue_ant)
                tx_ant_idx = tx_ant_idx[0]
                
                if rx_node_idx == 0:
                    rx_ant_idx = np.arange(0,self.num_bs_ant)
                else:
                    rx_ant_idx = np.arange(self.num_bs_ant + (rx_node_idx-1)*self.num_ue_ant,self.num_bs_ant + (rx_node_idx)*self.num_ue_ant)
                rx_ant_idx = rx_ant_idx[0]
                
                channel_train_input_list.append(h_freq_csi_history_reshaped[:num_training_steps,rx_ant_idx, :, :][:, tx_ant_idx, :])
                channel_train_gt_list.append(h_freq_csi_history_reshaped[-num_training_steps:,rx_ant_idx, :, :][:, tx_ant_idx, :])
                channel_test_input_list.append(h_freq_csi_history_reshaped[-self.syms_per_subframe:,rx_ant_idx,...][:, tx_ant_idx, :]) # TRY: If this doesn't work, try making them all 2x2 matrices

        pred_channel_training = self.fitting_time(channel_train_input_list, channel_train_gt_list)

        channel_pred_temp = self.test_train_predict(channel_test_input_list)

        channel_pred = tf.convert_to_tensor(channel_pred_temp)
        return channel_pred
    
    def mimo_predict(self, h_freq_csi_history):

        h_freq_csi_history = np.squeeze(np.asarray(h_freq_csi_history).transpose([0,1,2,3,4,5,7,6]))
        h_freq_csi_history = self.rb_mapper(h_freq_csi_history)

        num_time_steps = h_freq_csi_history.shape[0] * h_freq_csi_history.shape[-1]
        h_freq_csi_history_reshaped = np.moveaxis(h_freq_csi_history, -1, 1)
        h_freq_csi_history_reshaped = h_freq_csi_history_reshaped.reshape((num_time_steps,) + h_freq_csi_history.shape[1:-1])

        num_training_steps = (h_freq_csi_history.shape[0]-1)*self.syms_per_subframe

        channel_train_input_list = []
        channel_train_gt_list = []
        channel_test_input_list = []

        for tx_node_idx in range(self.num_tx_nodes):
            for rx_node_idx in range(self.num_rx_nodes):
                
                if tx_node_idx == 0:
                    tx_ant_idx = np.arange(0,self.num_bs_ant)
                else:
                    tx_ant_idx = np.arange(self.num_bs_ant + (tx_node_idx-1)*self.num_ue_ant,self.num_bs_ant + (tx_node_idx)*self.num_ue_ant)
                
                if rx_node_idx == 0:
                    rx_ant_idx = np.arange(0,self.num_bs_ant)
                else:
                    rx_ant_idx = np.arange(self.num_bs_ant + (rx_node_idx-1)*self.num_ue_ant,self.num_bs_ant + (rx_node_idx)*self.num_ue_ant)
                
                channel_train_input_list.append(h_freq_csi_history_reshaped[:num_training_steps,rx_ant_idx, :, :][:,:, tx_ant_idx, :])
                channel_train_gt_list.append(h_freq_csi_history_reshaped[-num_training_steps:,rx_ant_idx, :, :][:,:, tx_ant_idx, :])
                channel_test_input_list.append(h_freq_csi_history_reshaped[-self.syms_per_subframe:,rx_ant_idx,...][:, :, tx_ant_idx, :]) # TRY: If this doesn't work, try making them all 2x2 matrices

        self.fitting_time(channel_train_input_list, channel_train_gt_list)

        channel_pred_temp = self.test_train_predict(channel_test_input_list)
        
        channel_pred = np.zeros(h_freq_csi_history[0,...].shape,dtype=complex)
        
        for tx_node_idx in range(self.num_tx_nodes):
            for rx_node_idx in range(self.num_rx_nodes):
                if tx_node_idx == 0:
                    tx_ant_idx = np.arange(0,self.num_bs_ant)
                else:
                    tx_ant_idx = np.arange(self.num_bs_ant + (tx_node_idx-1)*self.num_ue_ant,self.num_bs_ant + (tx_node_idx)*self.num_ue_ant)
                
                if rx_node_idx == 0:
                    rx_ant_idx = np.arange(0,self.num_bs_ant)
                else:
                    rx_ant_idx = np.arange(self.num_bs_ant + (rx_node_idx-1)*self.num_ue_ant,self.num_bs_ant + (rx_node_idx)*self.num_ue_ant)

                vertex_idx = tx_node_idx * self.num_rx_nodes + rx_node_idx

                if vertex_idx in np.arange(0, self.num_rx_nodes):
                    N_t = self.num_bs_ant
                else:
                    N_t = self.num_ue_ant
                
                if vertex_idx % self.num_rx_nodes == 0:
                    N_r = self.num_bs_ant
                else:
                    N_r = self.num_ue_ant

                curr_chan_pred = channel_pred_temp[vertex_idx]

                channel_pred[np.ix_(rx_ant_idx, tx_ant_idx, np.arange(channel_pred.shape[2]), np.arange(channel_pred.shape[3]))] = curr_chan_pred.reshape(N_r, N_t, self.N_RB, -1)

        channel_pred = tf.convert_to_tensor(channel_pred)
        return channel_pred


    def init_weights(self):

        matrices = []
        for _ in range(self.max_adjacency):
            result = self.sparse_mat(self.N_n_per_vertex, self.N_n_per_vertex)
            matrices.append(result)
        self.W_N = np.concatenate(matrices, axis=1)

        self.W_in = 2 * (self.RS.rand(self.N_n_per_vertex, self.N_in) - 0.5)
        self.W_tran = np.concatenate([self.W_N, self.W_in], axis=1)

        self.W_out = self.RS.randn(self.N_out, (self.N_n_per_vertex + self.N_in) * self.N_v) + 1j * self.RS.randn(self.N_out, (self.N_n_per_vertex + self.N_in) * self.N_v)

    def sparse_mat(self, m, n):
        if self.type == 'real':
            W = self.RS.rand(m, m) - 0.5
            W[self.RS.rand(*W.shape) < self.sparsity] = 0
        else:
            W = 2*(self.RS.rand(m, n) - 0.5) + 2j*(self.RS.rand(m, n) - 0.5)
            W[self.RS.rand(*W.shape) < self.sparsity] = 0+1j*0
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        W = W * (self.spectral_radius / radius)
        return W

    def complex_to_real_target(self, Y_target_2D):
        Y_target_2D_real_list = []
        for t in range(self.N_t):
            target = Y_target_2D[t, :].reshape(1, -1) # (1, N_symbols * (N_fft+N_cp))
            real_target = np.concatenate((np.real(target), np.imag(target)), axis=0)  # (2, N_symbols * (N_fft+N_cp))
            Y_target_2D_real_list.append(real_target)
        Y_target_2D_real = np.concatenate(Y_target_2D_real_list, axis=0)
        return Y_target_2D_real

    def fitting_time(self, input, target):

        internal_states_history = self.state_transit(input)

        time_steps = input[0].shape[0]

        pred_channel = np.zeros((self.W_out.shape[0], time_steps),dtype=complex)

        for vertex_idx in range(self.N_v):

            curr_state_inds = np.arange(vertex_idx*self.N_n_per_vertex, (vertex_idx+1)*self.N_n_per_vertex)
            curr_W_out_inds = np.arange(vertex_idx*(self.N_n_per_vertex + self.N_in), (vertex_idx+1)*(self.N_n_per_vertex + self.N_in))
            curr_target = target[vertex_idx]
            curr_target = curr_target.reshape(curr_target.shape[0],-1).transpose(1,0)
            
            if self.rc_config.treatment == 'SISO':
                curr_output_inds = np.arange(vertex_idx*self.N_RB, (vertex_idx+1)*self.N_RB)
            elif self.rc_config.treatment == 'MIMO':
                curr_output_inds = np.arange(vertex_idx*self.num_bs_ant*self.num_bs_ant*self.N_RB, (vertex_idx+1)*self.num_bs_ant*self.num_bs_ant*self.N_RB)

            curr_input = input[vertex_idx].transpose()
            S_2D = np.concatenate([internal_states_history[curr_state_inds,:], curr_input], axis=0)

            self.W_out[np.ix_(curr_output_inds[:curr_target.shape[0]], curr_W_out_inds)] = curr_target @ self.reg_p_inv(S_2D)

            pred_channel[curr_output_inds[:curr_target.shape[0]], :] = self.W_out[np.ix_(curr_output_inds[:curr_target.shape[0]], curr_W_out_inds)] @ S_2D

        return pred_channel

    def cal_nmse(self, H, H_hat):
        H_hat = tf.cast(H_hat, dtype=H.dtype)
        mse = np.sum(np.abs(H - H_hat) ** 2)
        normalization_factor = np.sum((np.abs(H) + np.abs(H_hat)) ** 2)
        nmse = mse / normalization_factor
        return nmse

    def reg_p_inv(self, X):
        N = X.shape[0]
        return np.conj(X.T)@np.linalg.pinv(X@np.conj(X.T)+self.reg*np.eye(N))

    def test_train_predict(self, channel_train_input):

        self.S_0 = np.zeros([self.N_n], dtype='complex')
        Y_2D_org = channel_train_input

        internal_states_history = self.state_transit(Y_2D_org)

        pred_channel = []

        for vertex_idx in range(self.N_v):

            curr_state_inds = np.arange(vertex_idx*self.N_n_per_vertex, (vertex_idx+1)*self.N_n_per_vertex)
            curr_W_out_inds = np.arange(vertex_idx*(self.N_n_per_vertex + self.N_in), (vertex_idx+1)*(self.N_n_per_vertex + self.N_in))

            curr_input = Y_2D_org[vertex_idx]
            if self.rc_config.treatment == 'SISO':
                curr_output_inds = np.arange(vertex_idx*self.N_RB, (vertex_idx+1)*self.N_RB)
            elif self.rc_config.treatment == 'MIMO':
                curr_output_inds = np.arange(vertex_idx*self.num_bs_ant*self.num_bs_ant*self.N_RB, (vertex_idx+1)*self.num_bs_ant*self.num_bs_ant*self.N_RB)
            len_curr_input_features = curr_input[0,...].reshape(-1).shape[0]
            curr_output_inds = curr_output_inds[:len_curr_input_features]

            curr_input = Y_2D_org[vertex_idx].transpose()
            S_2D = np.concatenate([internal_states_history[curr_state_inds,:], curr_input], axis=0)

            curr_output = self.W_out[np.ix_(curr_output_inds, curr_W_out_inds)] @ S_2D
            pred_channel.append(curr_output)

        # curr_channel_pred = self.W_out @ S_2D
        return pred_channel

    def state_transit(self, Y_4D):

        T = Y_4D[0].shape[0] # number of samples

        internal_states_history = []

        internal_states = copy.deepcopy(self.S_0)
        for t in range(T):

            for vertex_idx in range(self.N_v):    

                curr_u = Y_4D[vertex_idx][t, ...] * self.input_scale
                curr_u_reshaped = curr_u.reshape(-1)

                state_matrix_inds = np.arange(vertex_idx*self.N_n_per_vertex, (vertex_idx+1)*self.N_n_per_vertex)
                internal_states[state_matrix_inds] = self.complex_tanh(self.W_in[:, :curr_u_reshaped.shape[0]] @ curr_u_reshaped  
                                                                       + self.W_N @ self.return_neighbours_states(vertex_idx, internal_states)) # WHY: why use the first few W_in weights
            internal_states_history.append(internal_states)
        
        internal_states_history = np.stack(internal_states_history, axis=1)
        
        return internal_states_history
    
    def return_neighbours_states(self, vertex_idx, internal_states):

        if self.max_adjacency == self.N_v:
            return internal_states



    def complex_tanh(self, Y):
        return np.tanh(np.real(Y)) + 1j * np.tanh(np.imag(Y))
    
    def rb_mapper(self, H):

        num_full_rbs = self.nfft // self.subcarriers_per_RB
        remainder_subcarriers = self.nfft % self.subcarriers_per_RB

        # Initialize an array to store the averaged RBs
        rb_data = np.zeros((H.shape[0], H.shape[1], H.shape[2], num_full_rbs + 1, 14), dtype=complex)

        # Compute mean across each full RB
        for rb in range(num_full_rbs):
            start = rb * self.subcarriers_per_RB
            end = start + self.subcarriers_per_RB
            rb_data[:, :, :, rb, :] = np.mean(H[:, :, :, start:end, :], axis=3)

        # Calculate the mean for the remaining subcarriers
        if remainder_subcarriers > 0:
            rb_data[:, :, :, -1, :] = np.mean(H[:, :, :, -remainder_subcarriers:, :], axis=3)
        
        return rb_data

    def rb_demapper(self, H):

        # expected shape of H: [num_rx_ant, num_tx_ant, num_RBs, num_ofdm_syms]

        H = tf.transpose(H, perm=[0,1,3,2])
        H = H[None, None, :, None, ...]

        demapped_H = tf.repeat(H, repeats=np.ceil(self.nfft/H.shape[-1]), axis=-1)
        demapped_H = demapped_H[..., :self.nfft]

        return demapped_H
