# Precoder for dMIMO channels
import numpy as np
import tensorflow as tf

from sionna.utils import matrix_inv


def mumimo_zf_precoder(x, h, return_precoding_matrix=False):
    """
    MU-MIMO precoding using ZF method (for testing purpose),
    treating all receiving antennas as independent ones.

    :param x: data stream symbols
    :param h: channel coefficients
    :param return_precoding_matrix: return precoding matrix
    :return: precoded data symbols
    """

    # Input dimensions:
    # x has shape: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
    # h has shape: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx, num_tx_ant]

    # Compute pseudo inverse for precoding
    g = tf.matmul(h, h, adjoint_b=True)
    g = tf.matmul(h, matrix_inv(g), adjoint_a=True)

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


def mumimo_bd_precoder(x, h, rx_indices, return_precoding_matrix=False, use_zero_forcing=True):
    """
    MU-MIMO precoding using BD method, assuming all receiving UE has equal number of antennas/number data streams
            gNobeB as twice the number of antennas if enabled.

    :param x: data stream symbols
    :param h: channel coefficients
    :param rx_indices: receiver antenna indices for all users
    :param return_precoding_matrix: return precoding matrix
    :return: precoded data symbols
    """

    # Input dimensions:
    # x: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
    # h: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx, num_tx_ant]
    # total_rx_ant = num_streams_per_tx

    total_rx_ant, total_tx_ant = h.shape[-2:]
    num_user = len(rx_indices)

    v_all = []
    for k in range(num_user):
        # Step 1: block diagonalization to minimize MUI
        num_rx_ant = len(rx_indices[k])  # number of antennas for user k
        rx_indices_comp = np.delete(np.arange(0, total_rx_ant, 1), rx_indices[k], axis=0)
        H_t = tf.gather(h, indices=rx_indices_comp, axis=-2)  # [..., total_rx_ant-num_rx_ant, num_tx_ant]
        s, u, v = tf.linalg.svd(H_t, compute_uv=True, full_matrices=True)
        # Make the signs of eigen vectors consistent
        v = tf.sign(v[..., :1, :]) * v
        # null space bases for use k
        v_c = v[..., -num_rx_ant:]  # [..., num_tx_ant, num_rx_ant]
        # effective channel for user k
        H_k = tf.gather(h, indices=rx_indices[k], axis=-2)  # [..., num_rx_ant, num_tx_ant]
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
            ss = tf.linalg.diag(tf.cast(1.0 / s2, tf.complex64))
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


def sumimo_bd_equalizer(y, h, rx_indices):
    """
    MU-MIMO equalizer for BD precoder
    :param y: received signals
    :param h: effective channel coefficients
    :param rx_indices: receiver antenna indices for all users
    :return: equalized signals
    """

    # Input dimensions:
    # y: [batch_size, num_rx, num_ofdm_sym, fft_size, num_rx_ant/num_streams_per_tx]
    # h: [batch_size, num_tx, num_ofdm_sym, fft_size, num_streams_per_tx, num_tx_ant]
    num_streams, num_tx_ant = h.shape[-2:]
    assert num_streams <= num_tx_ant, "Number of stream should not exceed number of antennas"

    total_rx_ant, total_tx_ant = h.shape[-2:]
    num_user = len(rx_indices)

    # v_all = []
    w_all = []
    for k in range(num_user):
        # Step 1: block diagonalization to minimize MUI
        num_rx_ant = len(rx_indices[k])  # number of antennas for user k
        rx_indices_comp = np.delete(np.arange(0, total_rx_ant, 1), rx_indices[k], axis=0)
        H_t = tf.gather(h, indices=rx_indices_comp, axis=-2)  # [..., total_rx_ant-num_rx_ant, num_tx_ant]
        s, u, v = tf.linalg.svd(H_t, compute_uv=True, full_matrices=True)
        # Make the signs of eigen vectors consistent
        v = tf.sign(v[..., :1, :]) * v
        # null space bases for use k
        v_c = v[..., -num_rx_ant:]
        # effective channel for user k
        H_k = tf.gather(h, indices=rx_indices[k], axis=-2)  # [..., num_rx_ant, num_tx_ant]
        H_eff = tf.linalg.matmul(H_k, v_c)  # [..., num_rx_ant, num_rx_ant]

        # Step 2: compute SVD for individual user
        s2, u2, v2 = tf.linalg.svd(H_eff, compute_uv=True, full_matrices=True)
        w = tf.linalg.adjoint(tf.sign(v2[..., :1, :]) * u2)
        w_all.append(w)

    # Expand last dim of `y` for equalization
    y_equalized = tf.expand_dims(y, -1)

    # Equalizing
    y_equalized = [tf.squeeze(tf.matmul(w_all[k], y_equalized[:, k:k+1]), -1) for k in range(num_user)]
    y_equalized = tf.concat(y_equalized, axis=1)

    return y_equalized

