import copy
import numpy as np
import tensorflow as tf
import warnings
import matplotlib.pyplot as plt
import itertools
import os
import logging
from concurrent.futures import ThreadPoolExecutor

from dmimo.config import Ns3Config, RCConfig
from dmimo.channel import lmmse_channel_estimation

class gesn_pred_freq_mimo:

    def __init__(self, 
                architecture, 
                len_features=None, 
                num_rx_ant=8, 
                num_tx_ant=8, 
                max_adjacency='all', 
                method='per_node_pair', 
                num_neurons=None,
                cp_len=64,
                num_subcarriers=512,
                subcarrier_spacing=15e3,
                num_epochs=20,
                learning_rate = 0.075,
                edge_weighting_method='grad_descent'):
        
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
        self.edge_weighting_method = edge_weighting_method # "none", "model_based", "model_free"
        self.cp_len = cp_len
        self.num_subcarriers = num_subcarriers
        self.subcarrier_spacing = subcarrier_spacing
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_initialization = "model_based" # 'model_based', 'ones'

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
        self.S_0 = tf.zeros([self.N_n], dtype=tf.complex128)

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

        antenna_selections = self.generate_antenna_selections(self.num_tx_nodes, self.num_rx_nodes)

        channel_pred = np.zeros(h_freq_csi_history_reshaped[:self.syms_per_subframe,...].shape, dtype=complex)

        # Loop over all possible graphs (MIMO to SISO simplification)
        for i in range(len(antenna_selections)):

            tx_ant_idx = antenna_selections[i]['transmitter_indices']
            rx_ant_idx = antenna_selections[i]['receiver_indices']

            curr_channels = h_freq_csi_history_reshaped[:,rx_ant_idx, :, :][:, :, tx_ant_idx, :]
            channel_test_input_list = [curr_channels[-self.syms_per_subframe:, rx_idx, tx_idx, :] for rx_idx in range(len(rx_ant_idx)) for tx_idx in range(len(tx_ant_idx))]
            
            # Train the model and calculate training loss:
            loss = self.training(h_freq_csi_history_reshaped, antenna_selections, i)
            
            # Generate output from trained model
            channel_pred_temp = self.test_train_predict(channel_test_input_list)

            # Store output
            for count_rx, rx in enumerate(rx_ant_idx):
                for count_tx, tx in enumerate(tx_ant_idx):
                    node_idx = count_rx * self.num_tx_nodes + count_tx

                    channel_pred[:, rx, tx, :] = tf.transpose(channel_pred_temp[node_idx])
            
        channel_pred = tf.convert_to_tensor(channel_pred)
        channel_pred = tf.transpose(channel_pred, perm=[1,2,3,0])
        return channel_pred


    def training(self, h_freq_csi_history_reshaped, antenna_selections, i):

        num_training_steps = h_freq_csi_history_reshaped.shape[0]-self.syms_per_subframe
        
        # Find antenna elements of current graph
        tx_ant_idx = antenna_selections[i]['transmitter_indices']
        rx_ant_idx = antenna_selections[i]['receiver_indices']
        
        # Get input-label pair for training data and testing data
        curr_channels = h_freq_csi_history_reshaped[:,rx_ant_idx, :, :][:, :, tx_ant_idx, :]
        channel_train_input_list = [curr_channels[:num_training_steps, rx_idx, tx_idx, :] for rx_idx in range(len(rx_ant_idx)) for tx_idx in range(len(tx_ant_idx))]
        channel_train_gt_list = [curr_channels[-num_training_steps:, rx_idx, tx_idx, :] for rx_idx in range(len(rx_ant_idx)) for tx_idx in range(len(tx_ant_idx))]
        channel_test_input_list = [curr_channels[-self.syms_per_subframe:, rx_idx, tx_idx, :] for rx_idx in range(len(rx_ant_idx)) for tx_idx in range(len(tx_ant_idx))]
        
        if self.weight_initialization == 'model_based':
            self.adjacency_matrix = self.cal_edge_weights(csi_history=curr_channels)
        elif self.weight_initialization == 'ones':
            self.adjacency_matrix = np.ones((self.N_v, self.N_v), dtype=float)
        tril_indices = np.tril_indices(self.adjacency_matrix.shape[0], -1)
        edge_weights = self.adjacency_matrix[tril_indices]

        if self.edge_weighting_method=='grad_descent':
            
            # For debugging, calculate the initial loss and print it out along with the initial edge weights
            self.S_0 = tf.zeros([self.N_n], dtype=tf.complex128)
            self.fitting_time(channel_train_input_list, channel_train_gt_list)
            input_channel_list = [curr_channels[self.syms_per_subframe:num_training_steps, rx_idx, tx_idx, :] for rx_idx in range(len(rx_ant_idx)) for tx_idx in range(len(tx_ant_idx))] # Taking the second-last subframe as input
            pred_channel_list = self.test_train_predict(input_channel_list)
            pred_channel = tf.convert_to_tensor(pred_channel_list)
            gt_channel_list = [curr_channels[-self.syms_per_subframe:, rx_idx, tx_idx, :] for rx_idx in range(len(rx_ant_idx)) for tx_idx in range(len(tx_ant_idx))]  # Taking the last subframe as ground truth
            gt_channel = tf.convert_to_tensor(gt_channel_list)
            gt_channel = tf.transpose(gt_channel, perm=[0,2,1])
            loss = self.cal_nmse(gt_channel, pred_channel)
            print(f"\n\nInitial edge weights: {edge_weights}, Loss: {loss.numpy()}")

            # Reset network weights
            # self.init_weights()

            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            edge_weights = tf.Variable(edge_weights, trainable=True)

            best_loss = loss
            best_edge_weights = edge_weights

            for curr_epoch in range(self.num_epochs):

                self.S_0 = tf.zeros([self.N_n], dtype=tf.complex128)
            
                with tf.GradientTape() as tape:
            
                    # Weight graph edges
                    self.adjacency_matrix = tf.zeros((self.N_v, self.N_v), dtype=tf.float32)
                    self.adjacency_matrix = tf.tensor_scatter_nd_update(
                        self.adjacency_matrix,
                        indices=tf.convert_to_tensor(np.stack(tril_indices, axis=-1), dtype=tf.int32),
                        updates=tf.cast(edge_weights, dtype=tf.float32)  # Ensure updates are float32
                    )
                    self.adjacency_matrix = self.adjacency_matrix + tf.transpose(self.adjacency_matrix)
                    self.adjacency_matrix = tf.linalg.set_diag(self.adjacency_matrix, tf.ones([self.N_v], dtype=tf.float32))
                    
                    # Train the model
                    pred_channel_training = self.fitting_time(channel_train_input_list, channel_train_gt_list)
                    
                    # Calculate training loss
                    input_channel_list = [curr_channels[self.syms_per_subframe:num_training_steps, rx_idx, tx_idx, :] for rx_idx in range(len(rx_ant_idx)) for tx_idx in range(len(tx_ant_idx))] # Taking the second-last subframe as input
                    pred_channel_list = self.test_train_predict(input_channel_list)
                    pred_channel = tf.convert_to_tensor(pred_channel_list)
                    gt_channel_list = [curr_channels[-self.syms_per_subframe:, rx_idx, tx_idx, :] for rx_idx in range(len(rx_ant_idx)) for tx_idx in range(len(tx_ant_idx))] # Taking the last subframe as ground truth
                    gt_channel = tf.convert_to_tensor(gt_channel_list)
                    gt_channel = tf.transpose(gt_channel, perm=[0,2,1])
                    loss = self.cal_nmse(gt_channel, pred_channel)

                # Compute gradients and update edge_weights
                tf.get_logger().setLevel(logging.ERROR)
                gradients = tape.gradient(loss, [edge_weights])
                optimizer.apply_gradients(zip(gradients, [edge_weights]))

                if loss < best_loss:
                    best_loss = loss
                    best_edge_weights = tf.identity(edge_weights)

                print(f"Epoch {curr_epoch + 1}/{self.num_epochs}, Loss: {loss.numpy()}, edge_weights: {edge_weights}")

            # Use the best edge weights found from gradient descent
            self.adjacency_matrix = tf.zeros((self.N_v, self.N_v), dtype=tf.float32)
            self.adjacency_matrix = tf.tensor_scatter_nd_update(
                self.adjacency_matrix,
                indices=tf.convert_to_tensor(np.stack(tril_indices, axis=-1), dtype=tf.int32),
                updates=tf.cast(best_edge_weights, dtype=tf.float32)
            )
            self.adjacency_matrix = self.adjacency_matrix + tf.transpose(self.adjacency_matrix)
            self.adjacency_matrix = tf.linalg.set_diag(self.adjacency_matrix, tf.ones([self.N_v], dtype=tf.float32))
            
            # Re-train the model based on the best edge weights
            self.S_0 = tf.zeros([self.N_n], dtype=tf.complex128)
            self.fitting_time(channel_train_input_list, channel_train_gt_list)

        elif self.edge_weighting_method == "model_based":

            # Train the model
            pred_channel_training = self.fitting_time(channel_train_input_list, channel_train_gt_list)
            
            # Calculate training loss
            training_loss_channel_list = [curr_channels[num_training_steps:, rx_idx, tx_idx, :] for rx_idx in range(len(rx_ant_idx)) for tx_idx in range(len(tx_ant_idx))]
            training_loss_channel_pred_list = self.test_train_predict(training_loss_channel_list)
            training_loss_channel_pred = tf.convert_to_tensor(training_loss_channel_pred_list)
            training_loss_channel_gt_list = channel_test_input_list
            training_loss_channel_gt = tf.convert_to_tensor(training_loss_channel_gt_list)
            training_loss_channel_gt = tf.transpose(training_loss_channel_gt_list, perm=[0,2,1])
            loss = self.cal_nmse(training_loss_channel_gt, training_loss_channel_pred)
        else:
            raise ValueError("\n The edge weighting method specified is not implemented")

        return loss

    def cal_edge_weights(self, csi_history):
        
        if self.edge_weighting_method == 'none':
            adjacency_matrix = np.ones((self.N_v, self.N_v)) - np.eye(self.N_v)
        elif self.edge_weighting_method == 'model_based' or self.edge_weighting_method== 'grad_descent':
            # Calculate delays:
            h_time_csi = np.fft.ifft(csi_history, axis=-1)
            power = np.abs(h_time_csi) ** 2
            range_power = np.max(power, axis=-1) - np.min(power, axis=-1)
            threshold = np.min(power, axis=-1) + 0.1 * range_power
            threshold = threshold[..., np.newaxis]
            significant_taps = power > threshold
            
            # Generate all combinations of TX and RX antennas
            num_rx_antennas = h_time_csi.shape[1]
            num_tx_antennas = h_time_csi.shape[2]
            antenna_combinations = list(itertools.product(range(num_rx_antennas), range(num_tx_antennas)))
            adjacency_matrix = np.ones((self.N_v, self.N_v))
            for idx1, (rx_1, tx_1) in enumerate(antenna_combinations):
                for idx2 in range(idx1 + 1, len(antenna_combinations)):  # Avoid already-selected pairs
                    rx_2, tx_2 = antenna_combinations[idx2]

                    # print('calculating correlation between {}_{} and {}_{}'.format(rx_1, tx_1, rx_2, tx_2))
                    # print('idx_1: {}, idx_2: {}'.format(idx1, idx2))

                    taps_1 = significant_taps[:, rx_1, tx_1, :].flatten()  # Flatten across symbols and resource blocks
                    taps_2 = significant_taps[:, rx_2, tx_2, :].flatten()

                    # Compute correlation coefficient
                    corr_value = np.corrcoef(taps_1, taps_2)[0, 1]
                    
                    # store weights (correlation coefficients) in adjacency matrix
                    adjacency_matrix[idx1, idx2] = corr_value
                    adjacency_matrix[idx2, idx1] = corr_value  # Symmetric for undirected graph
        else:
            raise ValueError("\n The edge weighting method specified is not implemented")


        return adjacency_matrix

    
    # def siso_predict_tmp(self, h_freq_csi_history):

    #     h_freq_csi_history = np.squeeze(np.asarray(h_freq_csi_history).transpose([0,1,2,3,4,5,7,6]))
    #     h_freq_csi_history = self.rb_mapper(h_freq_csi_history)

    #     num_time_steps = h_freq_csi_history.shape[0] * h_freq_csi_history.shape[-1]
    #     h_freq_csi_history_reshaped = np.moveaxis(h_freq_csi_history, -1, 1)
    #     h_freq_csi_history_reshaped = h_freq_csi_history_reshaped.reshape((num_time_steps,) + h_freq_csi_history.shape[1:-1])

    #     num_training_steps = (h_freq_csi_history.shape[0]-1)*self.syms_per_subframe

    #     channel_train_input_list = []
    #     channel_train_gt_list = []
    #     channel_test_input_list = []

    #     for tx_node_idx in range(self.num_tx_nodes):
    #         for rx_node_idx in range(self.num_rx_nodes):
                
    #             if tx_node_idx == 0:
    #                 tx_ant_idx = np.arange(0,self.num_bs_ant)
    #             else:
    #                 tx_ant_idx = np.arange(self.num_bs_ant + (tx_node_idx-1)*self.num_ue_ant,self.num_bs_ant + (tx_node_idx)*self.num_ue_ant)
    #             tx_ant_idx = tx_ant_idx[0]
                
    #             if rx_node_idx == 0:
    #                 rx_ant_idx = np.arange(0,self.num_bs_ant)
    #             else:
    #                 rx_ant_idx = np.arange(self.num_bs_ant + (rx_node_idx-1)*self.num_ue_ant,self.num_bs_ant + (rx_node_idx)*self.num_ue_ant)
    #             rx_ant_idx = rx_ant_idx[0]
                
    #             channel_train_input_list.append(h_freq_csi_history_reshaped[:num_training_steps,rx_ant_idx, :, :][:, tx_ant_idx, :])
    #             channel_train_gt_list.append(h_freq_csi_history_reshaped[-num_training_steps:,rx_ant_idx, :, :][:, tx_ant_idx, :])
    #             channel_test_input_list.append(h_freq_csi_history_reshaped[-self.syms_per_subframe:,rx_ant_idx,...][:, tx_ant_idx, :]) # TRY: If this doesn't work, try making them all 2x2 matrices

    #     pred_channel_training = self.fitting_time(channel_train_input_list, channel_train_gt_list)

    #     channel_pred_temp = self.test_train_predict(channel_test_input_list)

    #     channel_pred = tf.convert_to_tensor(channel_pred_temp)
    #     return channel_pred
    
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
        for _ in range(self.N_v):
            result = self.sparse_mat(self.N_n_per_vertex, self.N_n_per_vertex)
            matrices.append(result)
        self.W_N = tf.Variable(np.concatenate(matrices, axis=1), trainable=False)

        # self.W_in = 2 * (self.RS.rand(self.N_n_per_vertex, self.N_in) - 0.5)
        self.W_in = tf.Variable(2 * (self.RS.rand(self.N_n_per_vertex, self.N_in) - 0.5), trainable=False)  # Input weights (fixed)
        self.W_in = tf.cast(self.W_in, dtype=tf.complex128)
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
        # internal_states_history = self.state_transit_parallel(input)

        time_steps = input[0].shape[0]

        pred_channel = tf.zeros((self.W_out.shape[0], time_steps), dtype=tf.complex128)

        for vertex_idx in range(self.N_v):

            curr_state_inds = tf.range(vertex_idx * self.N_n_per_vertex, (vertex_idx + 1) * self.N_n_per_vertex)
            curr_W_out_inds = tf.range(vertex_idx * (self.N_n_per_vertex + self.N_in), (vertex_idx + 1) * (self.N_n_per_vertex + self.N_in))
            
            if self.rc_config.treatment == 'SISO':
                curr_output_inds = tf.range(vertex_idx*self.N_RB, (vertex_idx+1)*self.N_RB)
            elif self.rc_config.treatment == 'MIMO':
                curr_output_inds = tf.range(vertex_idx*self.num_bs_ant*self.num_bs_ant*self.N_RB, (vertex_idx+1)*self.num_bs_ant*self.num_bs_ant*self.N_RB)

            curr_target = tf.transpose(tf.reshape(target[vertex_idx], [tf.shape(target[vertex_idx])[0], -1]))
            curr_input = tf.transpose(input[vertex_idx])
            curr_internal_states_history = tf.gather(internal_states_history, curr_state_inds, axis=0)
            S_2D = tf.concat([curr_internal_states_history, curr_input], axis=0)

            curr_W_out_indices = tf.stack(tf.meshgrid(curr_output_inds, curr_W_out_inds, indexing="ij"), axis=-1)
            updates = tf.matmul(curr_target, self.reg_p_inv(S_2D))
            self.W_out = tf.tensor_scatter_nd_update(
                self.W_out,
                curr_W_out_indices,
                updates
            )

            pred_channel_indices = tf.stack(tf.meshgrid(curr_output_inds[:tf.shape(curr_target)[0]], tf.range(time_steps), indexing="ij"), axis=-1)
            curr_W_out = tf.gather_nd(self.W_out, curr_W_out_indices)
            curr_W_out = tf.cast(curr_W_out, S_2D.dtype)
            pred_channel = tf.tensor_scatter_nd_update(
                pred_channel,
                pred_channel_indices,
                tf.matmul(curr_W_out, S_2D)
            )

        return pred_channel

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

    def reg_p_inv(self, X):
        
        N = tf.shape(X)[0]
        X_conj_T = tf.transpose(tf.math.conj(X))
        identity_matrix = tf.eye(N, dtype=X.dtype)
        regularized_matrix = tf.matmul(X, X_conj_T) + self.reg * identity_matrix
        regularized_matrix_inv = tf.linalg.inv(regularized_matrix)

        return tf.matmul(X_conj_T, regularized_matrix_inv)

    def test_train_predict(self, channel_train_input):

        self.S_0 = tf.zeros([self.N_n], dtype=tf.complex128)
        Y_2D_org = channel_train_input

        internal_states_history = self.state_transit(Y_2D_org)

        pred_channel = []

        for vertex_idx in range(self.N_v):

            curr_state_inds = tf.range(vertex_idx * self.N_n_per_vertex, (vertex_idx + 1) * self.N_n_per_vertex)
            curr_W_out_inds = tf.range(vertex_idx * (self.N_n_per_vertex + self.N_in), (vertex_idx + 1) * (self.N_n_per_vertex + self.N_in))

            curr_input = Y_2D_org[vertex_idx]
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

        # curr_channel_pred = self.W_out @ S_2D
        return pred_channel

    def state_transit_parallel(self, Y_4D):

        T = Y_4D[0].shape[0] # number of samples
        Y_4D = Y_4D * self.input_scale

        internal_states_history = []

        internal_states = tf.identity(self.S_0)

        for t in range(T):
            # Gather inputs for all vertices at the current time step
            curr_inputs = tf.stack([Y_4D[vertex][t, ...] for vertex in range(self.N_v)], axis=0)  # Shape: [N_v, num_rbs]
            curr_inputs = tf.reshape(curr_inputs, [self.N_v, -1, 1])  # Shape: [N_v, num_rbs, 1]

            neighbors_states = tf.stack(
                [self.return_neighbours_states(vertex_idx, internal_states) for vertex_idx in range(self.N_v)],
                axis=0
            )  # Shape: [N_v, N_neighbors]

            neighbors_states = tf.expand_dims(neighbors_states, axis=-1)  # Shape: [N_v, N_neighbors, 1]

            # Compute state updates for all vertices
            state_updates = self.complex_tanh(
                self.W_in @ curr_inputs + self.W_N @ neighbors_states
            )  # Shape: [N_v, N_n_per_vertex, 1]

            vertex_indices = tf.range(self.N_v)[:, tf.newaxis] * self.N_n_per_vertex
            vertex_indices = tf.reshape(vertex_indices, [-1, 1]) + tf.range(self.N_n_per_vertex)
            vertex_indices = tf.reshape(vertex_indices, [-1, 1])  # Shape: [N_v * N_n_per_vertex, 1]

            internal_states = tf.tensor_scatter_nd_update(
                internal_states,
                tf.cast(vertex_indices, tf.int32),
                tf.squeeze(state_updates)
            )

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
