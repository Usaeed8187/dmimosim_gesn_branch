import copy
import numpy as np
import tensorflow as tf
import warnings
import matplotlib.pyplot as plt
import itertools
import os
import logging
from concurrent.futures import ThreadPoolExecutor
import time

from dmimo.config import Ns3Config, RCConfig
from dmimo.channel import lmmse_channel_estimation

class gesn_pred_freq_dmimo:

    def __init__(self, 
                architecture,
                rc_config,
                len_features=None, 
                num_rx_ant=8, 
                num_tx_ant=8, 
                max_adjacency='all', 
                method='per_antenna_pair', 
                num_neurons=None,
                cp_len=64,
                num_subcarriers=512,
                subcarrier_spacing=15e3,
                batch_size = 1,
                edge_weighting_method='grad_descent'):
        
        ns3_config = Ns3Config()
        self.rc_config = rc_config

        self.syms_per_subframe = 14
        self.nfft = 512  # TODO: remove hardcoded param value
        self.subcarriers_per_RB = 12
        self.N_RB = int(np.ceil(self.nfft / self.subcarriers_per_RB))
        self.num_rx_ant = num_rx_ant

        self.num_bs_ant = ns3_config.num_bs_ant
        self.num_ue_ant = ns3_config.num_ue_ant
        
        if architecture == 'MU_MIMO':
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
        self.edge_weight_update_method = edge_weighting_method # "none", "grad_descent"
        self.cp_len = cp_len
        self.num_subcarriers = num_subcarriers
        self.subcarrier_spacing = subcarrier_spacing
        self.num_epochs = self.rc_config.num_epochs
        self.learning_rate = self.rc_config.lr
        self.edge_weight_initialization = rc_config.weight_initialization # "none", "model_based_freq_corr", "model_based_delays", "uniform", "ones"
        self.batch_size = batch_size
        self.method = method
        self.window_weight_application = 'none'
        self.vector_inputs = rc_config.vector_inputs

        if self.vector_inputs == 'all':
            self.edge_weight_update_method = 'none'
            self.edge_weight_initialization = 'none'

        seed = 10
        self.RS = np.random.RandomState(seed)
        self.type = self.rc_config.type # 'real', 'complex'
        self.dtype = tf.complex64

        # Calculate weight matrix dimensions
        if method == 'per_antenna_pair':
            # one antenna pair is one vertex
            self.num_tx_nodes = int(self.N_t / ns3_config.num_ue_ant)
            self.num_rx_nodes = self.N_r
            self.N_v = self.num_rx_nodes * self.num_tx_nodes                                        # number of vertices in the graph. Will be updated in *predict_per_antenna_pair()
            self.N_e = int((self.N_v*(self.N_v-1))/2)                                                       # number of edges in the graph (assumes fully connected). Will be updated in *predict_per_antenna_pair()
            if self.rc_config.treatment == 'SISO':
                self.N_f = self.N_RB * self.num_ue_ant                                                                       # length of feature vector for each vertex
            else:
                raise ValueError("\n The GESN treatment specified is not defined")                          # length of feature vector for each vertex

        elif method == 'per_node_pair':
            # one tx-rx node pair is one vertex
            self.num_tx_nodes = int((self.N_t - ns3_config.num_bs_ant)/ns3_config.num_ue_ant) + 1
            self.num_rx_nodes = int((self.N_r - ns3_config.num_bs_ant)/ns3_config.num_ue_ant) + 1
            self.N_v = self.num_tx_nodes * self.num_rx_nodes                                            # number of vertices in the graph
            self.N_e = int((self.N_v*(self.N_v-1))/2)                                                   # number of edges in the graph (assumes fully connected)
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
        self.S_0 = tf.zeros([self.N_n], dtype=tf.complex64)

        # Initialize adjacency matrix (currently static for all time steps)
        if max_adjacency == 'all':
            self.max_adjacency = self.N_v
        elif max_adjacency == 'k_nearest_neighbours':
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
        """
        Returns a tf tensor of shape:
        [history_length, batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_sym, fft_size]
        containing channel estimates (complex).
        """
        first_csi_history_idx = first_slot_idx - (csi_delay * self.history_len)
        channel_history_slots = tf.range(first_csi_history_idx, first_slot_idx, csi_delay)

        h_freq_csi_history = tf.zeros((tf.size(channel_history_slots), self.batch_size, 1, (self.num_rx_nodes-1)*self.num_ue_ant+self.num_bs_ant,
                                       1, (self.num_tx_nodes-1)*self.num_ue_ant+self.num_bs_ant, self.syms_per_subframe, self.num_subcarriers), dtype=tf.complex64)
        for loop_idx, slot_idx in enumerate(channel_history_slots):
            # h_freq_csi has shape [batch_size, num_rx, num_rx_ant, num_tx, num_txs_ant, num_ofdm_sym, fft_size]
            folder_path = "ns3/channel_estimates_{}_{}_rx_{}_tx_{}".format(self.rc_config.mobility, self.rc_config.drop_idx,
                                                                                                self.N_r, self.N_t)
            file_path = "{}/dmimochans_{}".format(folder_path, slot_idx)
            try:
                data = np.load("{}.npz".format(file_path))
                h_freq_csi = data['h_freq_csi']
            except:
                h_freq_csi, _ = lmmse_channel_estimation(dmimo_chans, rg_csi, slot_idx=slot_idx)
                os.makedirs(folder_path, exist_ok=True)
                np.savez(file_path, h_freq_csi=h_freq_csi)
            indices = tf.constant([[loop_idx]])
            updates = tf.expand_dims(h_freq_csi, axis=0)
            h_freq_csi_history = tf.tensor_scatter_nd_update(h_freq_csi_history, indices, updates)

        return h_freq_csi_history

    def get_csi_history_mass(self, first_slot_idx, csi_delay, rg_csi, dmimo_chans):
        """
        Returns a tf tensor of shape:
        [history_length, batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_sym, fft_size]
        containing channel estimates (complex).
        """
        first_csi_history_idx = first_slot_idx - (csi_delay * self.history_len)
        channel_history_slots = tf.range(first_csi_history_idx, first_slot_idx, csi_delay)

        h_freq_csi_history = tf.zeros((tf.size(channel_history_slots), self.batch_size, 1, self.num_rx_nodes+1,
                                       1, (self.num_tx_nodes-1)*self.num_ue_ant+self.num_bs_ant, self.syms_per_subframe, self.num_subcarriers), dtype=tf.complex64)
        for loop_idx, slot_idx in enumerate(channel_history_slots):
            # h_freq_csi has shape [batch_size, num_rx, num_rx_ant, num_tx, num_txs_ant, num_ofdm_sym, fft_size]
            folder_path = "ns3/mass_channel_estimates_{}_{}_rx_{}_tx_{}".format(self.rc_config.mobility, self.rc_config.drop_idx,
                                                                                                self.N_r, self.N_t)
            file_path = "{}/dmimochans_{}".format(folder_path, slot_idx)
            try:
                data = np.load("{}.npz".format(file_path))
                h_freq_csi = data['h_freq_csi']
            except:
                h_freq_csi, _ = lmmse_channel_estimation(dmimo_chans, rg_csi, slot_idx=slot_idx)
                
                h_freq_csi = tf.gather(h_freq_csi, tf.range(0, h_freq_csi.shape[2], 2), axis=2)

                os.makedirs(folder_path, exist_ok=True)
                np.savez(file_path, h_freq_csi=h_freq_csi)
                
            indices = tf.constant([[loop_idx]])
            updates = tf.expand_dims(h_freq_csi, axis=0)
            h_freq_csi_history = tf.tensor_scatter_nd_update(h_freq_csi_history, indices, updates)

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


    def predict(self, h_freq_csi_history, gt_channel=None):
        
        self.gt_channel = gt_channel

        if self.rc_config.treatment == 'SISO':
            if self.method == 'per_antenna_pair':
                channel_pred = self.siso_predict_per_antenna_pair(h_freq_csi_history)
            else:
                raise ValueError("\n Not defined for MASS.")
        else:
            raise ValueError("\n Not defined for MASS.")
        
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

    def generate_node_selections(self, num_transmitters, num_receivers):
        """
        Generate all possible node selections for transmitters and receivers and convert one-hot encoding to indices.

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

    def siso_predict_per_antenna_pair(self, h_freq_csi_history):

        h_freq_csi_history = np.squeeze(np.asarray(h_freq_csi_history).transpose([0,1,2,3,4,5,7,6]))
        h_freq_csi_history = self.rb_mapper(h_freq_csi_history)

        num_time_steps = h_freq_csi_history.shape[0] * h_freq_csi_history.shape[-1]
        h_freq_csi_history_reshaped = np.moveaxis(h_freq_csi_history, -1, 1)
        h_freq_csi_history_reshaped = h_freq_csi_history_reshaped.reshape((num_time_steps,) + h_freq_csi_history.shape[1:-1])

        channel_pred = np.zeros(h_freq_csi_history_reshaped[:self.syms_per_subframe,...].shape, dtype=complex)

        if self.edge_weight_update_method=='grad_descent':
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        else:
            optimizer = None

        self.adjacency_matrix = tf.ones((self.N_v, self.N_v), dtype=float)

        if self.vector_inputs == 'tx_ants':

            edge_weights = tf.Variable(
                initial_value=tf.random.uniform(
                    [int(self.N_e)], 
                    minval=0.0,
                    maxval=1.0
                ),
                trainable=True,
                dtype=tf.float32,
                name="edge_weights_4_4"
            )

            all_trainable_variables = [edge_weights]

        else:
            raise ValueError("\n Not defined for the MASS implementation")
        
        if self.edge_weight_update_method == 'grad_descent':
            optimizer.build(all_trainable_variables)
        
        self.adjacency_matrix = self.cal_edge_weights_tf(csi_history=h_freq_csi_history_reshaped)
        
        all_tx_ant_idx = []
        
        for tx_node_idx in range(self.num_tx_nodes):
                
            tx_ant_idx = np.arange(tx_node_idx*self.num_ue_ant,(tx_node_idx+1)*self.num_ue_ant)

            all_tx_ant_idx.append(tx_ant_idx)

        channel_train_input_list, channel_train_gt_list, mapping = self.reshape_to_vertices(h_freq_csi_history_reshaped, all_tx_ant_idx, pad=False)        

        channel_test_input_list = channel_train_gt_list.copy()
        lower_triangular_mask = tf.linalg.band_part(tf.ones_like(self.adjacency_matrix), -1, 0) - tf.eye(tf.shape(self.adjacency_matrix)[0])
        curr_weights = tf.boolean_mask(self.adjacency_matrix, lower_triangular_mask > 0)
        edge_weights.assign(curr_weights)
        self.adjacency_matrix = tf.cast(self.adjacency_matrix, tf.complex64)

        self.training(edge_weights, channel_train_input_list, channel_train_gt_list, optimizer)

        # Generate output from trained model
        channel_pred_temp = self.test_train_predict(channel_test_input_list)
        channel_pred_temp = tf.convert_to_tensor(channel_pred_temp)
        channel_pred_temp = channel_pred_temp[..., -self.syms_per_subframe:]

        # Store output
        channel_pred = np.zeros((self.num_rx_nodes, self.N_t, self.N_RB, self.syms_per_subframe), dtype=complex)
        for rx in range(self.num_rx_nodes):
            for tx_node_idx in range(self.num_tx_nodes):

                tx_ant_idx = np.arange(tx_node_idx*self.num_ue_ant,(tx_node_idx+1)*self.num_ue_ant)

                v = rx * self.num_tx_nodes + tx_node_idx              # vertex index
                block = channel_pred_temp[v,...]                      # (n_ant_node*n_rb, n_time)
                block = np.reshape(block, [self.num_ue_ant, self.N_RB, self.syms_per_subframe])
                channel_pred[rx, tx_ant_idx, :, :] = block            # insert in antenna slots

        channel_pred = tf.convert_to_tensor(channel_pred)

        return channel_pred
    
    def reshape_to_vertices(self, H, tx_nodes, pad=True):
        """
        Returns:
        vertices    list (or stacked array) holding one tensor per vertex
        mapping     list of (rx_index, tx_node_index) pairs
        """
        T, N_rx, _, N_RB = H.shape
        channel_train_input_list, channel_train_gt_list, channel_test_input_list, mapping = [], [], [], []

        max_ant = max(len(g) for g in tx_nodes)             # for optional padding
        feat_len = max_ant * N_RB                           # pad target length

        num_training_steps = (self.history_len-1) * self.syms_per_subframe
        
        for rx in range(N_rx):
            for n, ant_idx in enumerate(tx_nodes):
                # Slice out the channel for this (rx, tx_node) pair
                # Result: (T, len(ant_idx), N_RB)
                ch = H[:, rx, ant_idx, :]

                # Flatten antennas and RBs into one axis: (T, len(ant_idx)*N_RB)
                ch = ch.reshape(T, -1)

                if pad and ch.shape[1] < feat_len:
                    pad_width = feat_len - ch.shape[1]
                    ch = np.pad(ch, ((0, 0), (0, pad_width)), mode='constant')

                channel_train_input_list.append(tf.cast(ch[:num_training_steps, ...], tf.complex64))        # each ch is now (T, feature_len)
                channel_train_gt_list.append(tf.cast(ch[-num_training_steps:, ...], tf.complex64))        # each ch is now (T, feature_len)
                mapping.append((rx, n))    # bookkeeping: which vertex this is

        # Optionally stack into single array of shape (N_vertices, T, feature_len)
        channel_train_input_list = np.stack(channel_train_input_list, axis=0) if pad else channel_train_input_list
        channel_train_gt_list = np.stack(channel_train_gt_list, axis=0) if pad else channel_train_gt_list
        
        return channel_train_input_list, channel_train_gt_list, mapping


    # @tf.function(jit_compile=True)
    def training(self, edge_weights, channel_train_input_list, channel_train_gt_list, optimizer):

        if self.edge_weight_update_method=='grad_descent':
            
            # For debugging, calculate the initial loss and print it out along with the initial edge weights
            self.S_0 = tf.zeros([self.N_n], dtype=tf.complex64)
            self.W_out = self.RS.randn(self.N_out, (self.N_n_per_vertex + self.N_in) * self.N_v) + 1j * self.RS.randn(self.N_out, (self.N_n_per_vertex + self.N_in) * self.N_v)
            pred_channel = self.fitting_time(channel_train_input_list, channel_train_gt_list)
            gt_channel = tf.convert_to_tensor(channel_train_gt_list)
            loss = self.cal_nmse(gt_channel, pred_channel)
            # print(f"Initial loss: {float(loss)}")

            best_loss = loss
            best_edge_weights = edge_weights

            start_time = time.time()
            for curr_epoch in range(self.num_epochs):
                
                self.S_0 = tf.zeros([self.N_n], dtype=tf.complex64)
            
                with tf.GradientTape() as tape:
                    
                    # Weight graph edges
                    self.adjacency_matrix = tf.zeros((self.N_v, self.N_v), dtype=tf.float32)
                    self.adjacency_matrix = self.fill_lower_triangle(edge_weights)
                    self.adjacency_matrix = self.adjacency_matrix + tf.transpose(self.adjacency_matrix)
                    self.adjacency_matrix = tf.linalg.set_diag(self.adjacency_matrix, tf.ones([self.N_v], dtype=tf.float32))
                    
                    # Train the model
                    pred_channel = self.fitting_time(channel_train_input_list, channel_train_gt_list)
                    
                    # Calculate training loss
                    loss = self.cal_nmse(gt_channel, pred_channel)

                # Compute gradients and update edge_weights
                tf.get_logger().setLevel(logging.ERROR)
                gradients = tape.gradient(loss, [edge_weights])
                optimizer.apply_gradients(zip(gradients, [edge_weights]))

                if loss < best_loss:
                    best_loss = loss
                    best_edge_weights = tf.identity(edge_weights)

                # print(f"Epoch {curr_epoch + 1}/{self.num_epochs}, Loss: {loss.numpy()}")

            # Use the best edge weights found from gradient descent
            self.adjacency_matrix = tf.zeros((self.N_v, self.N_v), dtype=tf.float32)
            self.adjacency_matrix = self.fill_lower_triangle(best_edge_weights)
            self.adjacency_matrix = self.adjacency_matrix + tf.transpose(self.adjacency_matrix)
            self.adjacency_matrix = tf.linalg.set_diag(self.adjacency_matrix, tf.ones([self.N_v], dtype=tf.float32))
            
            # Re-train the model based on the best edge weights
            self.S_0 = tf.zeros([self.N_n], dtype=tf.complex64)
            self.fitting_time(channel_train_input_list, channel_train_gt_list)

            # print(f"Lowest loss: {float(best_loss)}")

            end_time = time.time()
            # print("total gradient descent time = ", end_time - start_time)

        elif self.edge_weight_update_method == "none":

            self.S_0 = tf.zeros([self.N_n], dtype=tf.complex64)
            self.W_out = self.RS.randn(self.N_out, (self.N_n_per_vertex + self.N_in) * self.N_v) + 1j * self.RS.randn(self.N_out, (self.N_n_per_vertex + self.N_in) * self.N_v)
            pred_channel = self.fitting_time(channel_train_input_list, channel_train_gt_list)
            gt_channel = tf.convert_to_tensor(channel_train_gt_list)
            loss = self.cal_nmse(gt_channel, pred_channel)

        # elif self.edge_weight_update_method == 'model_based':

        #     # Train the model
        #     self.fitting_time(channel_train_input_list, channel_train_gt_list)
            
        #     # Calculate training loss
        #     training_loss_channel_list = [curr_channels[num_training_steps:, rx_idx, tx_idx, :] for rx_idx in range(len(rx_ant_idx)) for tx_idx in range(len(tx_ant_idx))]
        #     training_loss_channel_pred_list = self.test_train_predict(training_loss_channel_list)
        #     training_loss_channel_pred = tf.convert_to_tensor(training_loss_channel_pred_list)
        #     training_loss_channel_gt_list = channel_test_input_list
        #     training_loss_channel_gt = tf.convert_to_tensor(training_loss_channel_gt_list)
        #     training_loss_channel_gt = tf.transpose(training_loss_channel_gt_list, perm=[0,2,1])
        #     loss = self.cal_nmse(training_loss_channel_gt, training_loss_channel_pred)

        else:
            raise ValueError("\n The edge weighting method specified is not implemented")

        return loss
    
    @tf.function(jit_compile=True)
    def fill_lower_triangle(self, edge_weights):
        """
        Fills the lower triangular part of a matrix with the given edge_weights.
        
        Parameters:
        - edge_weights: A 1D tensor of weights to be placed in the lower triangular part.

        Returns:
        - A matrix with the lower triangular part filled with edge_weights.
        """

        # 1) Create a boolean mask for lower-triangular entries: i>j
        rows = tf.range(self.N_v, dtype=tf.int32)
        cols = tf.range(self.N_v, dtype=tf.int32)
        row_idx, col_idx = tf.meshgrid(rows, cols, indexing='ij')  # shape [M, M]
        mask_2d = row_idx > col_idx  # True below diag, shape [M, M]

        # 2) Flatten the mask to shape [self.N_v*self.N_v]
        mask_1d = tf.reshape(mask_2d, [-1])              # shape [self.N_v*self.N_v]
        mask_1d_int = tf.cast(mask_1d, dtype=tf.int32)   # 1 where True, else 0

        # 3) Compute a running index for each True entry via cumsum
        #    e.g. mask_1d = [F, F, T, T, ...] -> cumsum -> [0, 0, 1, 2, ...]
        cumsum_idx = tf.math.cumsum(mask_1d_int)  # shape [self.N_v*self.N_v]

        # 4) Each True position i gets index (cumsum_idx[i] - 1).
        #    Positions where mask_1d[i] = 0 get -1.
        gather_indices = cumsum_idx - 1  # shape [self.N_v*self.N_v], range: [-1..N_v-1]

        # 5) Safely clip indices to [0..N_v-1], then gather from edge_weights
        #    (we'll zero out the invalid positions later by multiplying with mask)
        safe_indices = tf.clip_by_value(gather_indices, 0, self.N_v - 1)
        gathered_values = tf.gather(edge_weights, safe_indices)  # shape [self.N_v*self.N_v]

        # 6) Zero-out positions where mask is False:
        #    multiply by float-cast mask_1d so that non-lower-tri entries are 0
        gathered_values *= tf.cast(mask_1d, tf.float32)

        # 7) Reshape to [self.N_v, self.N_v]
        updated_adjacency_matrix = tf.reshape(gathered_values, [self.N_v, self.N_v])

        return updated_adjacency_matrix


    def cal_edge_weights_tf(self, csi_history):

        """
        Compute adjacency matrix based on edge weighting method using TensorFlow (vectorized).
        
        Args:
            csi_history (tf.Tensor): CSI history tensor of shape (batch_size, num_rx_antennas, num_tx_antennas, subcarriers).
            N_v (int): Number of vertices in the graph.
            edge_weighting_method (str): Method for edge weighting ('none', 'model_based', 'grad_descent').
        
        Returns:
            tf.Tensor: Adjacency matrix of shape (N_v, N_v).
        """
        
        if self.method == 'per_antenna_pair' and self.edge_weight_initialization == 'model_based_freq_corr':
            num_time_steps, num_rx, num_tx, num_RBs = csi_history.shape
            csi_history_reordered = np.transpose(csi_history, (1, 2, 0, 3))

            bases = []

            for rx in range(self.num_rx_nodes):
                for tx in range(self.num_tx_nodes):

                    tx_ant_idx = np.arange(tx*self.num_ue_ant,(tx+1)*self.num_ue_ant)

                    M = csi_history_reordered[rx, tx_ant_idx, :, :]
                    M = M.transpose(1, 2, 0).reshape(-1, len(tx_ant_idx)).astype(np.complex64)

                    # M -= M.mean(axis=0, keepdims=True)

                    _, _, Vh = np.linalg.svd(M, full_matrices=False)
                    V2 = Vh.conj().T[:, :2]                   # (m_g, 2)

                    if len(tx_ant_idx) == 2:
                        V2 = np.pad(V2, ((0, 2), (0, 0)))    # (4, 2)

                    Q, _ = np.linalg.qr(V2)

                    bases.append(V2)

            bases = np.stack(bases, axis=0)                  # (12, 4, 2)

            r = 2
            Vh = bases.conj().transpose(0, 2, 1)             # (12, 2, 4)
            inner = Vh[:, None] @ bases[None, :]             # (12, 12, 2, 2)
            fro2  = np.sum(np.abs(inner)**2, axis=(2, 3))    # ||Vi^H Vj||_F^2
            adjacency_matrix = np.sqrt(np.maximum(r - fro2, 0.0))           # (12, 12)
            np.fill_diagonal(adjacency_matrix, 1.0)
            adjacency_matrix = adjacency_matrix.astype(np.float32)
            
        else:
            raise ValueError("\n Not defined for the MASS implementation")

        return adjacency_matrix
    
    def estimate_single_aoa_aod(self, csi_history, wavelength=0.0857, d=0.04285):
        """
        Estimate one AoA and one AoD for each antenna pair.
        
        Args:
        - csi_history: CSI history of shape (time_steps, num_rx, num_tx, num_RBs)
        - wavelength: Wavelength of the transmitted signal
        - d: Antenna spacing (in meters)

        Returns:
        - aoas: Array of AoAs for each antenna pair
        - aods: Array of AoDs for each antenna pair
        """
        num_time_steps, num_rx, num_tx, num_RBs = csi_history.shape
        
        aoas = np.zeros((num_time_steps, num_rx, num_tx, num_RBs))
        aods = np.zeros((num_time_steps, num_rx, num_tx, num_RBs))
        
        for t in range(num_time_steps):
            for rb in range(num_RBs):
                for rx in range(num_rx):
                    for tx in range(num_tx):
                        # Get the phase difference across antennas
                        H = csi_history[t, :, :, rb]
                        
                        # Estimate AoA (phase differences across receive antennas)
                        rx_phases = np.angle(H[:, tx])  # Phase differences at RX for this TX
                        aoa = np.arcsin(np.mean(rx_phases) * wavelength / (2 * np.pi * d))
                        aoas[t, rx, tx, rb] = np.degrees(aoa)
                        
                        # Estimate AoD (phase differences across transmit antennas)
                        tx_phases = np.angle(H[rx, :])  # Phase differences at TX for this RX
                        aod = np.arcsin(np.mean(tx_phases) * wavelength / (2 * np.pi * d))
                        aods[t, rx, tx, rb] = np.degrees(aod)
        
        return aoas, aods

    
    def steering_vector(self, angle, num_antennas, d, wavelength):
        """Compute the steering vector for a ULA."""
        k = 2 * np.pi / wavelength
        return np.exp(-1j * k * d * np.arange(num_antennas) * np.sin(np.radians(angle)))

    def music_spectrum(self, R, num_antennas, d, wavelength, angle_range):
        """Compute the MUSIC spectrum."""
        eigvals, eigvecs = np.linalg.eig(R)
        noise_subspace = eigvecs[:, np.argsort(eigvals)[:-1]]  # Exclude largest eigenvalues (signal subspace)
        
        angles = np.linspace(angle_range[0], angle_range[1], 360)
        spectrum = []
        
        for angle in angles:
            sv = self.steering_vector(angle, num_antennas, d, wavelength)
            spectrum.append(1 / np.abs(sv.conj().T @ noise_subspace @ noise_subspace.conj().T @ sv))
        
        return angles, np.abs(spectrum)

    def estimate_aoa_aod(self, csi_history, d=0.04285, wavelength=0.0857, angle_range=(0, 180)):
        """Estimate AoAs and AoDs from CSI history."""
        num_time_steps, num_rx_antennas, num_tx_antennas, num_RBs = csi_history.shape
        
        aoas = []  # Store AoA estimates
        aods = []  # Store AoD estimates

        for t in range(num_time_steps):
            for rb in range(num_RBs):
                # Extract channel matrix for this time step and RB
                H = csi_history[t, :, :, rb]

                # Step 1: Compute spatial covariance matrices
                R_r = H @ H.conj().T  # Receive covariance matrix
                R_t = H.conj().T @ H  # Transmit covariance matrix

                # Step 2: MUSIC AoA estimation
                aoa_angles, aoa_spectrum = self.music_spectrum(R_r, num_rx_antennas, d, wavelength, angle_range)
                aoa_peaks = aoa_angles[np.argsort(-aoa_spectrum)[:2]]  # Top 2 AoA peaks (example)
                aoas.append(aoa_peaks)

                # Step 3: MUSIC AoD estimation
                aod_angles, aod_spectrum = self.music_spectrum(R_t, num_tx_antennas, d, wavelength, angle_range)
                aod_peaks = aod_angles[np.argsort(-aod_spectrum)[:2]]  # Top 2 AoD peaks (example)
                aods.append(aod_peaks)

        return np.array(aoas), np.array(aods)


    def init_weights(self):

        matrices = []
        for _ in range(self.N_v):
            result = self.sparse_mat(self.N_n_per_vertex, self.N_n_per_vertex)
            matrices.append(result)
        self.W_N = tf.Variable(np.concatenate(matrices, axis=1), trainable=False)
        self.W_N = tf.cast(self.W_N, dtype=tf.complex64)

        # self.W_in = 2 * (self.RS.rand(self.N_n_per_vertex, self.N_in) - 0.5)
        self.W_in = tf.Variable(2 * (self.RS.rand(self.N_n_per_vertex, self.N_in) - 0.5), trainable=False)  # Input weights (fixed)
        self.W_in = tf.cast(self.W_in, dtype=tf.complex64)
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
    
    # def fitting_time_parallel(self, input, target):
        
    #     internal_states_history = self.state_transit_parallel(input)
    #     internal_states_history_reshaped = tf.reshape(internal_states_history, [-1, self.N_v, tf.shape(internal_states_history)[1]])
    #     internal_states_history_reshaped = tf.transpose(internal_states_history_reshaped, perm=[1,2,0])
        
    #     input = tf.convert_to_tensor(input)
    #     S_3D = tf.concat([internal_states_history_reshaped, input], axis=-1)
    #     S_3D_reshaped = tf.transpose(S_3D, perm=[0, 2, 1])

    #     target = tf.convert_to_tensor(input)
    #     target_reshaped = tf.transpose(target, perm=[0, 2, 1])
    #     self.W_out = tf.matmul(target_reshaped, self.reg_p_inv_parallel(S_3D_reshaped))
    
    
    def fitting_time(self, input, target, curr_window_weights=None):

        # internal_states_history = self.state_transit_parallel(input)
        if self.enable_window:
            input = self.form_window_input_signal(input, curr_window_weights)  # [N_r * window_length, N_symbols * (N_fft + N_cp)+delay]
        internal_states_history = self.state_transit_parallel_v2(input)
        
        self.W_out = tf.cast(self.W_out, tf.complex64)
        pred_channel = []

        for vertex_idx in range(self.N_v):

            curr_state_inds = tf.range(vertex_idx * self.N_n_per_vertex, (vertex_idx + 1) * self.N_n_per_vertex)
            curr_W_out_inds = tf.range(vertex_idx * (self.N_n_per_vertex + self.N_in), (vertex_idx + 1) * (self.N_n_per_vertex + self.N_in))
            
            if self.vector_inputs == 'tx_ants' or self.vector_inputs == 'rx_ants' or self.vector_inputs == 'all':
                if self.rc_config.treatment == 'SISO':
                    curr_output_inds = tf.range(vertex_idx*self.N_f, (vertex_idx+1)*self.N_f)
                elif self.rc_config.treatment == 'MIMO':
                    curr_output_inds = tf.range(vertex_idx*self.num_bs_ant*self.num_bs_ant*self.N_f, (vertex_idx+1)*self.num_bs_ant*self.num_bs_ant*self.N_f)
            else:
                if self.rc_config.treatment == 'SISO':
                    curr_output_inds = tf.range(vertex_idx*self.N_RB, (vertex_idx+1)*self.N_RB)
                elif self.rc_config.treatment == 'MIMO':
                    curr_output_inds = tf.range(vertex_idx*self.num_bs_ant*self.num_bs_ant*self.N_RB, (vertex_idx+1)*self.num_bs_ant*self.num_bs_ant*self.N_RB)

            curr_target = tf.transpose(tf.reshape(target[vertex_idx], [tf.shape(target[vertex_idx])[0], -1]))
            curr_input = tf.transpose(input[vertex_idx])
            curr_internal_states_history = tf.gather(internal_states_history, curr_state_inds, axis=0)
            S_2D = tf.concat([curr_internal_states_history, curr_input], axis=0)

            curr_W_out_indices = tf.stack(tf.meshgrid(curr_output_inds, curr_W_out_inds, indexing="ij"), axis=-1)
            # updates = tf.matmul(curr_target, self.reg_p_inv_parallel(S_2D))
            updates = tf.matmul(curr_target, self.reg_p_inv(S_2D))
            self.W_out = tf.tensor_scatter_nd_update(
                self.W_out,
                curr_W_out_indices,
                updates
            )

            pred_channel.append(tf.transpose(updates @ S_2D))

        pred_channel = tf.convert_to_tensor(pred_channel)

        return pred_channel
    
    def form_window_input_signal(self, Y_2D_complex, curr_window_weights):
        # Y_2D: [N_r, N_symbols * (N_fft + N_cp)]
        Y_2D_complex = tf.convert_to_tensor(Y_2D_complex)

        if self.type == 'real':
            Y_2D = np.concatenate((Y_2D_complex.real, Y_2D_complex.imag), axis=0)
        else:
            Y_2D = copy.deepcopy(Y_2D_complex)
        Y_2D_window = []
        for n in range(self.window_length):
            shift_y_2d = np.roll(Y_2D, shift=n, axis=-1)
            if self.type == 'real':
                shift_y_2d[:, :n] = 0.
            else:
                shift_y_2d[:, :n] = 0. + 0.j
            Y_2D_window.append(shift_y_2d) # a method to explore
        
        # Y_2D_window = np.concatenate(Y_2D_window, axis = 0) # [N_r * window_length, N_symbols * (N_fft + N_cp)+delay]
        if self.type == 'real':
            Y_2D_window = np.concatenate(Y_2D_window, axis=-1)
        else:
            Y_2D_window = np.concatenate(Y_2D_window, axis=-1)
        
        return Y_2D_window


    def cal_nmse(self, H, H_hat):
        """
        Calculate NMSE between H and H_hat using TensorFlow operations.
        Args:
            H: Ground truth tensor.
            H_hat: Predicted tensor.
        Returns:
            NMSE: Normalized Mean Squared Error (TensorFlow tensor).
        """
        H_hat = tf.cast(H_hat, dtype=H.dtype)
        mse = tf.reduce_sum(tf.abs(H - H_hat) ** 2)
        # mean_H = tf.reduce_mean(H)
        # variance_H = tf.reduce_mean(tf.square(tf.abs(H - mean_H)))
        # nmse = mse / (variance_H + 1e-8)  # Add epsilon for numerical stability
        normalization_factor = tf.reduce_sum((tf.abs(H) + tf.abs(H_hat)) ** 2)
        nmse = tf.math.real(mse / normalization_factor)
        return nmse

    @tf.function(jit_compile=False)
    def reg_p_inv(self, X):
        
        N = tf.shape(X)[0]
        X_conj_T = tf.transpose(tf.math.conj(X))
        identity_matrix = tf.eye(N, dtype=X.dtype)
        regularized_matrix = tf.matmul(X, X_conj_T) + self.reg * identity_matrix
        regularized_matrix_inv = tf.linalg.inv(regularized_matrix)

        return tf.matmul(X_conj_T, regularized_matrix_inv)
    
    def reg_p_inv_parallel(self, X):
        
        N = tf.shape(X)[1]
        X_conj_T = tf.transpose(tf.math.conj(X), perm=[0, 2, 1])
        identity_matrix = tf.eye(N, dtype=X.dtype)
        regularized_matrix = tf.matmul(X, X_conj_T) + self.reg * identity_matrix
        regularized_matrix_inv = tf.linalg.inv(regularized_matrix)

        return tf.matmul(X_conj_T, regularized_matrix_inv)

    def test_train_predict_parallel(self, channel_train_input, curr_window_weights=None):

        if self.enable_window:
            channel_train_input = self.form_window_input_signal(channel_train_input, curr_window_weights)  # [N_r * window_length, N_symbols * (N_fft + N_cp)+delay]

        self.S_0 = tf.zeros([self.N_n], dtype=tf.complex64)
        Y_2D_org = tf.convert_to_tensor(channel_train_input)

        # internal_states_history = self.state_transit(Y_2D_org)
        # internal_states_history = self.state_transit_parallel(Y_2D_org)
        internal_states_history = self.state_transit_parallel_v2(Y_2D_org)
        internal_states_history_reshaped = tf.reshape(internal_states_history, [-1, self.N_v, tf.shape(internal_states_history)[1]])
        internal_states_history_reshaped = tf.transpose(internal_states_history_reshaped, perm=[1,2,0])

        S_3D = tf.concat([internal_states_history_reshaped, Y_2D_org], axis=-1)
        S_3D_reshaped = tf.transpose(S_3D, perm=[0, 2, 1])

        pred_channel = tf.matmul(self.W_out, S_3D_reshaped)

        return pred_channel

    # @tf.function(jit_compile=True)
    def test_train_predict(self, channel_train_input, curr_window_weights=None):

        if self.enable_window:
            channel_train_input = self.form_window_input_signal(channel_train_input, curr_window_weights)  # [N_r * window_length, N_symbols * (N_fft + N_cp)+delay]
            channel_train_input = tf.cast(channel_train_input, self.dtype)

        self.S_0 = tf.zeros([self.N_n], dtype=tf.complex64)
        Y_2D_org = channel_train_input
        self.W_out = tf.cast(self.W_out, tf.complex64)

        # internal_states_history = self.state_transit(Y_2D_org)
        # internal_states_history = self.state_transit_parallel(Y_2D_org)
        internal_states_history = self.state_transit_parallel_v2(Y_2D_org)

        pred_channel = []

        for vertex_idx in range(self.N_v):

            curr_state_inds = tf.range(vertex_idx * self.N_n_per_vertex, (vertex_idx + 1) * self.N_n_per_vertex)
            curr_W_out_inds = tf.range(vertex_idx * (self.N_n_per_vertex + self.N_in), (vertex_idx + 1) * (self.N_n_per_vertex + self.N_in))

            curr_input = Y_2D_org[vertex_idx]
            if self.vector_inputs == 'tx_ants' or self.vector_inputs == 'rx_ants' or self.vector_inputs == 'all':
                if self.rc_config.treatment == 'SISO':
                    curr_output_inds = tf.range(vertex_idx * self.N_f, (vertex_idx + 1) * self.N_f)
                elif self.rc_config.treatment == 'MIMO':
                    curr_output_inds = tf.range(
                        vertex_idx * self.num_bs_ant * self.num_bs_ant * self.N_f,
                        (vertex_idx + 1) * self.num_bs_ant * self.num_bs_ant * self.N_f,
                    )
            else:
                if self.rc_config.treatment == 'SISO':
                    curr_output_inds = tf.range(vertex_idx * self.N_RB, (vertex_idx + 1) * self.N_RB)
                elif self.rc_config.treatment == 'MIMO':
                    curr_output_inds = tf.range(
                        vertex_idx * self.num_bs_ant * self.num_bs_ant * self.N_RB,
                        (vertex_idx + 1) * self.num_bs_ant * self.num_bs_ant * self.N_RB,
                    )

            len_curr_input_features = tf.shape(curr_input[0, ...])[0]
            curr_output_inds = curr_output_inds[:len_curr_input_features]

            curr_input = tf.transpose(Y_2D_org[vertex_idx])
            curr_internal_states_history = tf.gather(internal_states_history, curr_state_inds, axis=0)
            S_2D = tf.concat([curr_internal_states_history, curr_input], axis=0)

            curr_W_out_indices = tf.stack(tf.meshgrid(curr_output_inds, curr_W_out_inds, indexing="ij"), axis=-1)
            curr_W_out = tf.gather_nd(self.W_out, curr_W_out_indices)
            curr_output = tf.matmul(curr_W_out, S_2D)
            pred_channel.append(curr_output)

        return pred_channel


    def state_transit_parallel_v2(self, Y_4D):

        T = Y_4D[0].shape[0] # number of samples
        Y_4D = tf.convert_to_tensor(Y_4D)
        Y_4D = Y_4D * self.input_scale
        Y_4D = tf.cast(Y_4D, dtype=tf.complex64)

        internal_states_history = []

        internal_states = tf.identity(self.S_0)

        for t in range(T):
            # Gather inputs for all vertices at the current time step
            curr_inputs = tf.squeeze(tf.gather(Y_4D, t, axis=1))  # Shape: [N_v, num_rbs]
            curr_inputs = tf.transpose(curr_inputs, perm=[1,0])  # Shape: [num_rbs, N_v]

            # neighbors_states = tf.stack(
            #     [self.return_neighbours_states(vertex_idx, internal_states) for vertex_idx in range(self.N_v)],
            #     axis=0
            # )  # Shape: [N_v*N_n_per_vertex, N_v]
            # neighbors_states = tf.transpose(neighbors_states, perm=[1,0])  # Shape: [N_v*N_n_per_vertex, N_v]
            neighbors_states = self.return_neighbours_states_v2(internal_states)

            # Compute state updates for all vertices
            state_updates = self.complex_tanh(
                self.W_in @ curr_inputs + self.W_N @ neighbors_states
            )  # Shape: [N_n_per_vertex, N_v]

            internal_states = tf.reshape(state_updates, [-1])

            # Store the updated states
            internal_states_history.append(internal_states)

        # Stack the internal states across time steps
        internal_states_history = tf.stack(internal_states_history, axis=1)  # Shape: [N_v * N_n_per_vertex, num_time_steps]
        return internal_states_history


    # @tf.function(jit_compile=True)
    def state_transit_parallel(self, Y_4D):

        T = Y_4D[0].shape[0] # number of samples
        Y_4D = tf.convert_to_tensor(Y_4D)
        Y_4D = Y_4D * self.input_scale
        Y_4D = tf.cast(Y_4D, dtype=tf.complex64)

        internal_states_history = []

        internal_states = tf.identity(self.S_0)

        for t in range(T):
            # Gather inputs for all vertices at the current time step
            curr_inputs = tf.stack([Y_4D[vertex, t, ...] for vertex in range(self.N_v)], axis=0)  # Shape: [N_v, num_rbs]
            curr_inputs = tf.transpose(curr_inputs, perm=[1,0])  # Shape: [num_rbs, N_v]

            neighbors_states = tf.stack(
                [self.return_neighbours_states(vertex_idx, internal_states) for vertex_idx in range(self.N_v)],
                axis=0
            )  # Shape: [N_v*N_n_per_vertex, N_v]
            neighbors_states = tf.transpose(neighbors_states, perm=[1,0])  # Shape: [N_v*N_n_per_vertex, N_v]

            # Compute state updates for all vertices
            state_updates = self.complex_tanh(
                self.W_in @ curr_inputs + self.W_N @ neighbors_states
            )  # Shape: [N_n_per_vertex, N_v]

            vertex_indices = tf.range(self.N_v)[:, tf.newaxis] * self.N_n_per_vertex
            vertex_indices = tf.reshape(vertex_indices, [-1, 1]) + tf.range(self.N_n_per_vertex)
            vertex_indices = tf.reshape(vertex_indices, [-1, 1])  # Shape: [N_v * N_n_per_vertex, 1]

            internal_states = tf.reshape(state_updates, [-1])

            # Store the updated states
            internal_states_history.append(internal_states)

        # Stack the internal states across time steps
        internal_states_history = tf.stack(internal_states_history, axis=1)  # Shape: [N_v * N_n_per_vertex, num_time_steps]
        return internal_states_history

    def state_transit(self, Y_4D):

        T = Y_4D[0].shape[0] # number of samples

        internal_states_history = []

        internal_states = tf.identity(self.S_0)
        
        for t in range(T):

            for vertex_idx in range(self.N_v):    

                curr_u = Y_4D[vertex_idx][t, ...] * self.input_scale
                curr_u_reshaped = tf.reshape(curr_u, [-1, 1])

                neighbours_states = self.return_neighbours_states(vertex_idx, internal_states)
                neighbours_states = neighbours_states[:, np.newaxis]
                neighbours_states = tf.cast(neighbours_states, self.W_N.dtype)
                state_matrix_inds = tf.range(vertex_idx * self.N_n_per_vertex, (vertex_idx + 1) * self.N_n_per_vertex)
                
                internal_states = tf.tensor_scatter_nd_update(
                    internal_states,
                    tf.reshape(state_matrix_inds, [-1, 1]),
                    tf.squeeze(self.complex_tanh(
                        self.W_in[:, :tf.shape(curr_u_reshaped)[0]] @ curr_u_reshaped + self.W_N @ neighbours_states
                    ))
                )

            internal_states_history.append(internal_states)
        
        internal_states_history = tf.stack(internal_states_history, axis=1)
        
        return internal_states_history
    
    # @tf.function(jit_compile=True)
    def return_neighbours_states_v2(self, internal_states):

        # 1) Reshape 'internal_states' to [N_v, N_n_per_vertex]
        #    so row j corresponds to vertex j’s internal subunits.
        internal_states_2d = tf.reshape(internal_states, [self.N_v, self.N_n_per_vertex])
        
        # 2) adjacency_matrix has shape [N_v, N_v].
        #    We'll make it [N_v, N_v, 1] so we can broadcast across subunits:
        #       result[i, j, k] = adjacency[i, j] * internal_states_2d[j, k]
        adjacency_3d = self.adjacency_matrix[:, :, tf.newaxis]            # => [N_v, N_v, 1]
        
        # 3) Expand 'internal_states_2d' to [1, N_v, N_n_per_vertex] 
        #    so it lines up with adjacency’s second dimension (j):
        internal_states_3d = internal_states_2d[tf.newaxis, :, :]    # => [1, N_v, N_n_per_vertex]
        
        # 4) Broadcast-multiply to get shape [N_v, N_v, N_n_per_vertex]
        #    - i dimension = adjacency’s first axis
        #    - j dimension = adjacency’s second axis + internal_states’ first axis
        #    - k dimension = internal_states’ second axis
        adjacency_3d = tf.cast(adjacency_3d, internal_states_3d.dtype)
        result_3d = adjacency_3d * internal_states_3d  # => [N_v, N_v, N_n_per_vertex]
        
        # 5) Reshape to [N_v, N_v*N_n_per_vertex], so each row i is
        #    the flattened neighbor-states for vertex i
        result_2d = tf.reshape(result_3d, [self.N_v, self.N_v * self.N_n_per_vertex])
        
        # 6) Transpose to match your original final shape: [N_v*N_n_per_vertex, N_v]
        neighbors_states = tf.transpose(result_2d, perm=[1, 0])

        return neighbors_states

    # @tf.function(jit_compile=True)
    def return_neighbours_states(self, vertex_idx, internal_states):

        if self.max_adjacency == self.N_v:

            curr_adjacency = self.adjacency_matrix[vertex_idx, :]
            repeated_adjacency = tf.repeat(curr_adjacency, self.N_n_per_vertex)
            repeated_adjacency = tf.cast(repeated_adjacency, dtype=internal_states.dtype)
            return internal_states * repeated_adjacency

    def complex_tanh(self, Y):
        real_part = tf.math.tanh(tf.math.real(Y))
        imag_part = tf.math.tanh(tf.math.imag(Y))

        return tf.complex(real_part, imag_part)
    
    def rb_mapper(self, H):

        num_full_rbs = self.nfft // self.subcarriers_per_RB
        remainder_subcarriers = self.nfft % self.subcarriers_per_RB

        # Initialize an array to store the averaged RBs
        rb_data = np.zeros((H.shape[0], H.shape[1], H.shape[2], num_full_rbs + 1, 14), dtype=np.complex64)

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
    
