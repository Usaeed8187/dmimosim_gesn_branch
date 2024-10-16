import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
import sionna
from sionna.utils import flatten_dims
from sionna.ofdm import RemoveNulledSubcarriers

from .fiveG_precoding import baseline_fiveG_precoder, dMIMO_p1_fiveG_precoder


class fiveGPrecoder(Layer):
    """5G Precoder for Baseline and """

    def __init__(self,
                 architecture,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)

        self.architecture = architecture        

    def call(self, inputs):

        if len(inputs) == 3:
            x, precoding_matrices, cqi_snr = inputs
        else:
            ValueError("calling 5G precoder with incorrect params")

        # x has shape
        # [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        #
        # precoding_matrices has shape
        # [num_rx, fft_size, num_tx_ant, num_rx_ant]

        # Transformations to bring h and x in the desired shapes

        # Transpose x:
        # [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
        x_precoded = tf.transpose(x, [0, 1, 3, 4, 2])
        x_precoded = tf.cast(x_precoded, self._dtype)
        x_precoded = x_precoded[..., np.newaxis]

        # Transpose precoding_matrices:
        # [batch_size, num_tx, num_ofdm_symbols, fft_size, num_tx_ant, num_streams_per_tx]
        precoding_matrices = precoding_matrices[np.newaxis, :, np.newaxis, :, :, :]
        precoding_matrices = np.repeat(precoding_matrices, x_precoded.shape[2], axis=2)
        precoding_matrices = np.repeat(precoding_matrices, x_precoded.shape[0], axis=0)
        precoding_matrices = tf.cast(precoding_matrices, dtype=x_precoded.dtype)


        # Precoding
        if self.architecture == 'baseline':
            x_precoded = baseline_fiveG_precoder(x_precoded,
                                            precoding_matrices)
        elif self.architecture == 'dMIMO_p1':
            x_precoded = dMIMO_p1_fiveG_precoder(x_precoded,
                                            precoding_matrices)

        # Transpose output to desired shape:
        # [batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        x_precoded = tf.transpose(x_precoded, [0, 1, 4, 2, 3])


        return x_precoded

    def reconstruct_channel(self, precoding_matrices, snr_assumed_dBm, n_var, bs_txpwr_dbm):

        rx_sig_pow = n_var * 10**(snr_assumed_dBm/10)
        tx_sig_pow = 10**(bs_txpwr_dbm/10)
        s = np.sqrt(rx_sig_pow / tx_sig_pow)

        h_freq_csi_reconstructed = precoding_matrices * s

        reshaped_array = h_freq_csi_reconstructed.transpose(2, 1, 0)
        reshaped_array = reshaped_array[np.newaxis, np.newaxis, :, np.newaxis, :, np.newaxis, :]
        repeated_array = np.repeat(reshaped_array, 14, axis=5)
        h_freq_csi_reconstructed = tf.convert_to_tensor(repeated_array)

        if self.architecture == 'baseline':
            padding = self.num_BS_Ant - h_freq_csi_reconstructed.shape[2]
        elif self.architecture == 'dMIMO_phase1':
            padding = self.num_UE_Ant - h_freq_csi_reconstructed.shape[2]
        else:
            padding = 0

        padding_mask = [
            [0, 0],  # No padding on the 1st dimension
            [0, 0],  # No padding on the 2nd dimension
            [0, padding],  # Pad the 3rd dimension from 2 to 4 (2 zeros after)
            [0, 0],  # No padding on the 4th dimension
            [0, 0],  # No padding on the 5th dimension
            [0, 0],  # No padding on the 6th dimension
            [0, 0],  # No padding on the 7th dimension
        ]

        h_freq_csi_reconstructed = tf.pad(h_freq_csi_reconstructed, padding_mask)

        return h_freq_csi_reconstructed