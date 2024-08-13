# Precoder for dMIMO channels
import numpy as np
import tensorflow as tf

from sionna.utils import matrix_inv


def mumimo_bd_precoder(x, h, ue_indices, ue_ranks, return_precoding_matrix=False, use_zero_forcing=False):
    """
    MU-MIMO precoding using BD method.

    :param x: data stream symbols
    :param h: channel coefficients
    :param ue_indices: receiver antenna indices for all users
    :param ue_ranks: number of streams (ranks) for all users
    :param return_precoding_matrix: return precoding matrix
    :return: precoded data symbols
    """

    # Input dimensions:
    # x: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
    # h: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx, num_tx_ant]
    # total_rx_ant = num_streams_per_tx

    total_rx_ant, total_tx_ant = h.shape[-2:]
    num_user = len(ue_indices)

    v_all = []
    for k in range(num_user):
        # Step 1: block diagonalization to minimize MUI
        num_rx_ant = len(ue_indices[k])  # number of antennas for user k
        rx_indices_comp = np.delete(np.arange(0, total_rx_ant, 1), ue_indices[k], axis=0)
        H_t = tf.gather(h, indices=rx_indices_comp, axis=-2)  # [..., total_rx_ant-num_rx_ant, num_tx_ant]
        s, u, v = tf.linalg.svd(H_t, compute_uv=True, full_matrices=True)
        # Make the signs of eigen vectors consistent
        v = tf.sign(v[..., :1, :]) * v
        # null space bases for use k
        v_c = v[..., -num_rx_ant:]  # [..., num_tx_ant, num_rx_ant]
        # effective channel for user k
        H_k = tf.gather(h, indices=ue_indices[k], axis=-2)  # [..., num_rx_ant, num_tx_ant]
        H_eff = tf.linalg.matmul(H_k, v_c)  # [..., num_rx_ant, num_rx_ant]

        if use_zero_forcing:
            # Step 2: compute ZF for individual user k
            g = tf.matmul(H_eff, H_eff, adjoint_b=True)
            g = tf.matmul(H_eff, matrix_inv(g), adjoint_a=True)
            v_eff = tf.linalg.matmul(v_c, g)
            v_all.append(v_eff)

        else:
            # Step 2: compute SVD for individual user
            s2, u2, v2 = tf.linalg.svd(H_eff, compute_uv=True, full_matrices=True)
            # Make the signs of eigen vectors consistent
            v2 = tf.sign(v2[..., :1, :]) * v2
            # rank adaptation
            v2 = v2[..., :ue_ranks[k]]
            ss = tf.linalg.diag(tf.cast(1.0 / s2[..., :ue_ranks[k]], tf.complex64))
            v2 = tf.linalg.matmul(v2, ss)
            v_eff = tf.linalg.matmul(v_c, v2)  # [..., num_tx_ant, num_rx_ant]
            v_all.append(v_eff)

    # combine v_eff for all users
    v_bd = tf.concat(v_all, axis=-1)  # [..., num_tx_ant, num_streams_per_tx]

    # Precoding
    x_precoded = tf.expand_dims(x, -1)  # expand last dim of `x` for precoding
    x_precoded = tf.squeeze(tf.matmul(v_bd, x_precoded), -1)

    if return_precoding_matrix:
        return x_precoded, v_bd
    else:
        return x_precoded


def mumimo_bd_equalizer(y, h, ue_indices, ue_ranks):
    """
    MU-MIMO equalizer for BD precoder
    :param y: received signals
    :param h: effective channel coefficients
    :param ue_indices: receiver antenna indices for all users
    :param ue_ranks: number of streams (ranks) for all users
    :return: equalized signals
    """

    # Input dimensions:
    # y: [batch_size, num_rx, num_ofdm_sym, fft_size, num_rx_ant/num_streams_per_tx]
    # h: [batch_size, num_tx, num_ofdm_sym, fft_size, num_streams_per_tx, num_tx_ant]
    num_streams, num_tx_ant = h.shape[-2:]
    assert num_streams <= num_tx_ant, "Number of stream should not exceed number of antennas"

    total_rx_ant, total_tx_ant = h.shape[-2:]
    num_user = len(ue_indices)

    w_all = []
    for k in range(num_user):
        # Step 1: block diagonalization to minimize MUI
        num_rx_ant = len(ue_indices[k])  # number of antennas for user k
        rx_indices_comp = np.delete(np.arange(0, total_rx_ant, 1), ue_indices[k], axis=0)
        H_t = tf.gather(h, indices=rx_indices_comp, axis=-2)  # [..., total_rx_ant-num_rx_ant, num_tx_ant]
        s, u, v = tf.linalg.svd(H_t, compute_uv=True, full_matrices=True)
        # Make the signs of eigen vectors consistent
        v = tf.sign(v[..., :1, :]) * v
        # null space bases for use k
        v_c = v[..., -num_rx_ant:]
        # effective channel for user k
        H_k = tf.gather(h, indices=ue_indices[k], axis=-2)  # [..., num_rx_ant, num_tx_ant]
        H_eff = tf.linalg.matmul(H_k, v_c)  # [..., num_rx_ant, num_rx_ant]

        # Step 2: compute SVD for individual user
        s2, u2, v2 = tf.linalg.svd(H_eff, compute_uv=True, full_matrices=True)
        w = tf.linalg.adjoint(tf.sign(v2[..., :1, :]) * u2)

        # Rank adaptation support (extract only relevant streams)
        w = w[..., :ue_ranks[k], :]
        w_all.append(w)

    # Expand last dim of `y` for equalization
    y_equalized = tf.expand_dims(y, -1)

    # Equalizing
    y_equalized = [tf.squeeze(tf.matmul(w_all[k], y_equalized[:, k:k+1]), -1) for k in range(num_user)]
    y_equalized = tf.concat(y_equalized, axis=1)

    return y_equalized

