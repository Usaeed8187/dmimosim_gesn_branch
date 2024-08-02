# SVD precoding for dMIMO channels
import tensorflow as tf


def sumimo_svd_precoder(x, h, return_precoding_matrix=False):
    """
    SU-MIMO precoding using SVD
    :param x: data stream symbols
    :param h: channel coefficients
    :param return_precoding_matrix: return precoding matrix
    :return: precoded data symbols and (optional) precoding matrix
    """

    # Input dimensions:
    # x: [batch_size, num_tx, num_ofdm_sym, fft_size, num_streams_per_tx]
    # h: [batch_size, num_tx, num_ofdm_sym, fft_size, num_rx_ant, num_tx_ant]
    num_streams = x.shape[-1]
    num_rx_ant, num_tx_ant = h.shape[-2:]
    assert (num_streams <= num_tx_ant) and (num_streams <= num_rx_ant), \
        "Number of stream should not exceed number of antennas"

    # Compute SVD of channel matrix for precoding
    s, u, v = tf.linalg.svd(h, compute_uv=True)

    # Make the signs of eigen vectors consistent
    v = tf.sign(v[..., :1, :]) * v

    # support for rank adaptation
    v = v[..., :num_streams]

    # Expand last dim of `x` for precoding
    x_precoded = tf.expand_dims(x, -1)

    # Precode
    x_precoded = tf.squeeze(tf.matmul(v, x_precoded), -1)

    if return_precoding_matrix:
        return x_precoded, v
    else:
        return x_precoded


def sumimo_svd_equalizer(y, h):
    """
    SU-MIMO equalizer for SVD precoder
    :param y: received signals
    :param h: channel coefficients
    :return: equalized signals
    """

    # Input dimensions:
    # y: [batch_size, num_rx, num_ofdm_sym, fft_size, num_rx_ant]
    # h: [batch_size, num_tx, num_ofdm_sym, fft_size, num_rx_ant, num_tx_ant]
    num_rx_ant, num_tx_ant = h.shape[-2:]
    assert (num_rx_ant <= num_tx_ant), "Number of Rx antennas should not exceed number of Tx antennas"

    # Compute SVD of channel matrix for precoding
    s, u, v = tf.linalg.svd(h, compute_uv=True)

    # Make the signs of eigen vectors consistent
    # Compute equalizing weight
    w = tf.linalg.adjoint(tf.sign(v[..., :1, :]) * u)

    # Expand last dim of `y` for equalization
    y_equalized = tf.expand_dims(y, -1)

    # Equalizing
    y_equalized = tf.squeeze(tf.matmul(w, y_equalized), -1)

    return y_equalized

