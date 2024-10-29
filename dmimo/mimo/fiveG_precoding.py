# Zero-Forcing (ZF) Precoder for dMIMO channels
import numpy as np
import tensorflow as tf

from sionna.utils import matrix_inv


def baseline_fiveG_precoder(x, precoding_matrices):
    """
    SU-MIMO precoding using 5G method, using the precoder returned by the user.

    :param x: data stream symbols
    :param precoding_matrices: precoding matrices for each RB
    :return: precoded data symbols
    """

    # Input dimensions:
    # x has shape: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
    # h has shape: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_rx_ant, num_tx_ant]

    # Precode
    x_precoded = tf.squeeze(tf.matmul(precoding_matrices, x), -1)

    return x_precoded

def dMIMO_p1_fiveG_max_min_precoder(x, h):
    """
    Phase 1 optimization based precoding

    :param x: data stream symbols
    :param h: channel coefficients
    :return: precoded data symbols
    """

    # Input dimensions:
    # x: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
    # h: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]


    num_streams_per_tx = x.shape[-2]
    
    
    if return_precoding_matrix:
        return x_precoded, g
    else:
        return x_precoded
