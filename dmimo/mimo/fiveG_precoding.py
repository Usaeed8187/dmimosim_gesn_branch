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

def dMIMO_p1_fiveG_precoder(x, h, ue_indices, ue_ranks, return_precoding_matrix=False):
    """
    MU-MIMO zero-forcing precoding supporting rank adaptation.

    :param x: data stream symbols
    :param h: channel coefficients
    :param ue_indices: receiver antenna indices for all users
    :param ue_ranks: number of streams (ranks) for all users
    :param return_precoding_matrix: return precoding matrix
    :return: precoded data symbols
    """

    # Input dimensions:
    # x: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
    # h: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_rx_ant, num_tx_ant]
    num_streams_per_tx = x.shape[-1]
    total_rx_ant, total_tx_ant = h.shape[-2:]
    num_user = len(ue_indices)
    num_user_streams = np.sum(ue_ranks)
    num_user_ant = np.sum(len(val) for val in ue_indices)
    assert num_user_streams == num_streams_per_tx, "total number of streams must match"
    assert num_user_ant == total_rx_ant, "number Rx antennas must match"
    assert (num_user_streams <= total_tx_ant) and (num_user_streams <= total_rx_ant), \
        "total number of streams must be less than total number of Tx/Rx antennas"

    if total_rx_ant == num_user_streams:
        # Compute pseudo inverse for precoding
        g = tf.matmul(h, h, adjoint_b=True)
        g = tf.matmul(h, matrix_inv(g), adjoint_a=True)
    else:
        # Rank adaptation support
        h_all = []
        for k in range(num_user):
            # Update effective channels for all users
            num_rx_ant = len(ue_indices[k])  # number of antennas for user k
            h_ue = tf.gather(h, indices=ue_indices[k], axis=-2)
            if ue_ranks[k] == num_rx_ant:
                # full rank
                h_all.append(h_ue)
            else:
                # support only one stream adaptation
                assert(ue_ranks[k] == 1)
                # Calculate MRC weights
                g = tf.math.conj(tf.math.reduce_sum(h_ue, axis=-1, keepdims=True))
                # g = tf.matmul(g, tf.cast(1.0, tf.complex64)/tf.matmul(g, g, adjoint_a=True))
                h_eff = tf.matmul(g, h_ue, adjoint_a=True)
                h_all.append(h_eff)
        # Combine h_eff for all users
        h_zf = tf.concat(h_all, axis=-2)  # [..., num_tx_ant, num_streams_per_tx]

        # Compute pseudo inverse for precoding
        g = tf.matmul(h_zf, h_zf, adjoint_b=True)
        g = tf.matmul(h_zf, matrix_inv(g), adjoint_a=True)

    # Normalize each column to unit power
    norm = tf.sqrt(tf.reduce_sum(tf.abs(g)**2, axis=-2, keepdims=True))
    g = g/tf.cast(norm, g.dtype)

    # Expand last dim of `x` for precoding
    x_precoded = tf.expand_dims(x, -1)

    # Precode
    x_precoded = tf.squeeze(tf.matmul(g, x_precoded), -1)

    if return_precoding_matrix:
        return x_precoded, g
    else:
        return x_precoded
