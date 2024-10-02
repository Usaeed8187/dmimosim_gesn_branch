# SLNR precoding for dMIMO channels
import numpy as np
import tensorflow as tf


def mumimo_slnr_precoder(x, h, no, ue_indices, return_precoding_matrix=False):
    """
    MU-MIMO precoding based on signal-to-leakage-noise-ratio (SLNR) criterion

    :param x: data stream symbols
    :param h: channel coefficients
    :param no: noise variance
    :param ue_indices: receiver antenna indices for all users
    :param return_precoding_matrix: return precoding matrix
    :return: precoded data symbols
    """

    # Input dimensions:
    # x: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
    # h: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_rxs_ant, num_txs_ant]

    num_streams_per_tx = x.shape[-1]
    total_rx_ant, num_tx_ant = h.shape[-2:]
    assert total_rx_ant >= num_streams_per_tx, "inputs with incorrect dimensions"
    num_user = len(ue_indices)

    F_all = []
    for k in range(num_user):
        # number of antennas for user k
        num_rx_ant = len(ue_indices[k])
        # antenna indices for users other than k
        rx_indices_comp = np.delete(np.arange(0, total_rx_ant, 1), ue_indices[k], axis=0)
        # effective channel for user k
        H_k = tf.gather(h, indices=ue_indices[k], axis=-2)  # [..., num_rx_ant, num_tx_ant]
        # complement channels to user k
        H_t = tf.gather(h, indices=rx_indices_comp, axis=-2)  # [..., total_rx_ant-num_rx_ant, num_tx_ant]
        # compute inputs to the SLNR algorithm
        A_k = tf.matmul(tf.linalg.adjoint(H_k), H_k)  # [..., num_tx_ant, num_tx_ant]
        scaled_sigma = tf.linalg.diag((total_rx_ant/num_rx_ant * no) * tf.ones((num_tx_ant), dtype=tf.complex64))
        scaled_sigma = tf.reshape(scaled_sigma, (1, 1, 1, 1, *A_k.shape[-2:]))
        C_k = A_k + scaled_sigma + tf.matmul(tf.linalg.adjoint(H_t), H_t)  # [..., num_tx_ant, num_tx_ant]
        # step 1: compute Cholesky decomposition on C_k and obtain Q_k
        G_k = tf.linalg.cholesky(C_k)  # [..., num_tx_ant, num_tx_ant]
        Q_k = tf.linalg.adjoint(tf.linalg.inv(G_k))
        # step 2: compute eigen-decomposition on A_p and obtain U_k
        A_p = tf.matmul(tf.matmul(tf.linalg.adjoint(Q_k), A_k), Q_k)  # [..., num_tx_ant, num_tx_ant]
        s_k, u_k, v_k = tf.linalg.svd(A_p)
        # make the signs of eigen vectors consistent
        v_sign = tf.linalg.diag(tf.sign(v_k[..., 0, :]))
        u_k = tf.matmul(u_k, v_sign)
        # Step 3: compute P_k
        P_k = tf.matmul(Q_k, u_k)  # [..., num_tx_ant, num_tx_ant]
        F_k = P_k[..., :, :num_rx_ant]  # [..., num_tx_ant, num_rx_ant]
        # normalization to unit power
        norm = tf.sqrt(tf.reduce_sum(tf.abs(F_k)**2, axis=-2, keepdims=True))
        F_k = F_k/tf.cast(num_rx_ant*norm, F_k.dtype)
        # save for current user
        F_all.append(F_k)

    # combine precoding vectors for all users
    F_all = tf.concat(F_all, axis=-1)  # [..., num_tx_ant, num_streams_per_tx]

    # Precoding
    x_precoded = tf.expand_dims(x, -1)  # expand last dim of `x` for precoding
    x_precoded = tf.squeeze(tf.matmul(F_all, x_precoded), -1)

    if return_precoding_matrix:
        return x_precoded, F_all
    else:
        return x_precoded


def mumimo_slnr_equalizer(y, h, no, ue_indices):
    """
    MU-MIMO precoding based on signal-to-leakage-noise-ratio (SLNR) criterion

    :param y: data stream symbols
    :param h: channel coefficients
    :param no: noise variance
    :param ue_indices: receiver antenna indices for all users
    :return: precoded data symbols
    """

    # Input dimensions:
    # y: [batch_size, num_rx, num_ofdm_sym, fft_size, num_rx_ant/num_streams_per_tx]
    # h: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_rxs_ant, num_txs_ant]

    total_rx_ant, num_tx_ant = h.shape[-2:]
    num_user = len(ue_indices)

    G_all = []
    for k in range(num_user):
        # number of antennas for user k
        num_rx_ant = len(ue_indices[k])
        # antenna indices for users other than k
        rx_indices_comp = np.delete(np.arange(0, total_rx_ant, 1), ue_indices[k], axis=0)
        # effective channel for user k
        H_k = tf.gather(h, indices=ue_indices[k], axis=-2)  # [..., num_rx_ant, num_tx_ant]
        # complement channels to user k
        H_t = tf.gather(h, indices=rx_indices_comp, axis=-2)  # [..., total_rx_ant-num_rx_ant, num_tx_ant]
        # compute inputs to the SLNR algorithm
        A_k = tf.matmul(tf.linalg.adjoint(H_k), H_k)  # [..., num_tx_ant, num_tx_ant]
        scaled_sigma = tf.linalg.diag((total_rx_ant/num_rx_ant * no) * tf.ones((num_tx_ant), dtype=tf.complex64))
        scaled_sigma = tf.reshape(scaled_sigma, (1, 1, 1, 1, *A_k.shape[-2:]))
        C_k = A_k + scaled_sigma + tf.matmul(tf.linalg.adjoint(H_t), H_t)  # [..., num_tx_ant, num_tx_ant]
        # step 1: compute Cholesky decomposition on C_k and obtain Q_k
        G_k = tf.linalg.cholesky(C_k)  # [..., num_tx_ant, num_tx_ant]
        Q_k = tf.linalg.adjoint(tf.linalg.inv(G_k))
        # step 2: compute eigen-decomposition on A_p and obtain U_k
        A_p = tf.matmul(tf.matmul(tf.linalg.adjoint(Q_k), A_k), Q_k)  # [..., num_tx_ant, num_tx_ant]
        s_k, u_k, v_k = tf.linalg.svd(A_p)
        # make the signs of eigen vectors consistent
        v_sign = tf.linalg.diag(tf.sign(v_k[..., 0, :]))  # [..., num_tx_ant, num_tx_ant]
        u_k = tf.matmul(u_k, v_sign)
        # Step 3: compute P_k
        P_k = tf.matmul(Q_k, u_k)  # [..., num_tx_ant, num_tx_ant]
        F_k = P_k[..., :, :num_rx_ant]  # [..., num_tx_ant, num_rx_ant]
        # normalization to unit power
        norm = tf.sqrt(tf.reduce_sum(tf.abs(F_k)**2, axis=-2, keepdims=True))
        F_k = F_k/tf.cast(num_rx_ant*norm, F_k.dtype)
        # compute equalizer matrix
        G_k = tf.linalg.adjoint(tf.matmul(H_k, F_k))  # [..., num_rx_ant, num_rx_ant]
        G_all.append(G_k)

    # Expand last dim of `y` for equalization
    y_equalized = tf.expand_dims(y, -1)

    # Equalizing
    y_equalized = [tf.squeeze(tf.matmul(G_all[k], y_equalized[:, k:k+1]), -1) for k in range(num_user)]
    y_equalized = tf.concat(y_equalized, axis=1)

    return y_equalized
