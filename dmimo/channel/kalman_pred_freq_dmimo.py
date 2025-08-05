import numpy as np
import tensorflow as tf
import os

from dmimo.config import Ns3Config, RCConfig
from dmimo.channel import lmmse_channel_estimation

class kalman_pred_freq_dmimo:
    """Simple Kalman-filter based channel predictor for dMIMO frequency domain channels."""
    def __init__(self,
                 architecture,
                 rc_config: RCConfig,
                 num_rx_ant=8,
                 num_tx_ant=8,
                 cp_len=64,
                 num_subcarriers=512,
                 subcarrier_spacing=15e3,
                 batch_size=1):
        ns3_config = Ns3Config()
        self.rc_config = rc_config
        self.syms_per_subframe = 14
        self.nfft = 512
        self.subcarriers_per_RB = 12
        self.N_RB = int(np.ceil(self.nfft / self.subcarriers_per_RB))
        self.num_rx_ant = num_rx_ant
        self.num_bs_ant = ns3_config.num_bs_ant
        self.num_ue_ant = ns3_config.num_ue_ant
        if architecture == 'MU_MIMO':
            self.N_t = num_tx_ant
            self.N_r = num_rx_ant
        else:
            raise ValueError("The architecture specified is not defined")
        self.cp_len = cp_len
        self.num_subcarriers = num_subcarriers
        self.subcarrier_spacing = subcarrier_spacing
        self.batch_size = batch_size
        self.history_len = rc_config.history_len

    def get_csi_history_mass(self, first_slot_idx, csi_delay, rg_csi, dmimo_chans):
        """Load a history of channel estimates used for prediction."""
        first_csi_history_idx = first_slot_idx - (csi_delay * self.history_len)
        channel_history_slots = tf.range(first_csi_history_idx, first_slot_idx, csi_delay)
        h_freq_csi_history = tf.zeros((tf.size(channel_history_slots), self.batch_size, 1, self.N_r + 1,
                                       1, self.N_t, self.syms_per_subframe, self.num_subcarriers),
                                      dtype=tf.complex64)
        for loop_idx, slot_idx in enumerate(channel_history_slots):
            folder_path = "ns3/mass_channel_estimates_{}_{}_rx_{}_tx_{}".format(self.rc_config.mobility,
                                                                                self.rc_config.drop_idx,
                                                                                self.N_r, self.N_t)
            file_path = "{}/dmimochans_{}".format(folder_path, slot_idx)
            try:
                data = np.load("{}.npz".format(file_path))
                h_freq_csi = data['h_freq_csi']
            except Exception:
                h_freq_csi, _ = lmmse_channel_estimation(dmimo_chans, rg_csi, slot_idx=slot_idx)
                h_freq_csi = tf.gather(h_freq_csi, tf.range(0, h_freq_csi.shape[2], 2), axis=2)
                os.makedirs(folder_path, exist_ok=True)
                np.savez(file_path, h_freq_csi=h_freq_csi)
            indices = tf.constant([[loop_idx]])
            updates = tf.expand_dims(h_freq_csi, axis=0)
            h_freq_csi_history = tf.tensor_scatter_nd_update(h_freq_csi_history, indices, updates)
        return h_freq_csi_history

    def rb_mapper(self, H):
        num_full_rbs = self.nfft // self.subcarriers_per_RB
        remainder_subcarriers = self.nfft % self.subcarriers_per_RB
        rb_data = np.zeros((H.shape[0], H.shape[1], H.shape[2], num_full_rbs + 1, 14), dtype=np.complex64)
        for rb in range(num_full_rbs):
            start = rb * self.subcarriers_per_RB
            end = start + self.subcarriers_per_RB
            rb_data[:, :, :, rb, :] = np.mean(H[:, :, :, start:end, :], axis=3)
        if remainder_subcarriers > 0:
            rb_data[:, :, :, -1, :] = np.mean(H[:, :, :, -remainder_subcarriers:, :], axis=3)
        return rb_data

    def predict(self, h_freq_csi_history):
        """Perform one-step Kalman prediction given channel history."""
        h = np.squeeze(h_freq_csi_history).transpose([0,1,2,4,3])  # [T, rx, tx, subcarriers, ofdm]
        h_rb = self.rb_mapper(h)  # [T, rx, tx, RB, ofdm]
        x_est = h_rb[0]
        P = np.ones_like(x_est.real)
        Q = 1e-5
        R = 1e-3
        for t in range(1, h_rb.shape[0]):
            P = P + Q
            K = P / (P + R)
            x_est = x_est + K * (h_rb[t] - x_est)
            P = (1 - K) * P
        pred = x_est
        pred = pred[None, None, :, None, :, :, :]
        return tf.convert_to_tensor(pred, dtype=tf.complex64)