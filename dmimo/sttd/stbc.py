"""
Space-time block codes (STBC)

Reference: S. M. Alamouti, "A simple transmit diversity technique for wireless communications,"
in IEEE Journal on Selected Areas in Communications, vol. 16, no. 8, pp. 1451-1458, Oct. 1998.

System model:
  | r0 r1 |  = | h0 h1 |  * | s0 -s1* |
  | r2 r3 |    | h2 h3 |    | s1  s0* |
or
  | r0 r2 |  = | s0   s1  | * | h0 h2 |
  | r1 r3 |    | -s1* s0* |   | h1 h3 |
"""

import tensorflow as tf


def stbc_encode(x):
    """
    Space-time block code (STBC) encoder
    :param x: input symbols, shape [..., num_syms]
    :return: encoded symbols, shape [..., num_syms, 2]
    """

    # check length of input symbols
    assert x.shape[-1] % 2 == 0, "total number of symbols must be even"

    # split input symbols into two symbol sets (x0, x1)
    x = tf.reshape(x, (*x.shape[:-1], -1, 2))  # last dimension is set index

    # --------------------------------------
    #        Alamouti 2x2 scheme
    # --------------------------------------

    # s0, -conj(s1) for antenna 1
    y1 = tf.concat((x[..., 0:1], -tf.math.conj(x[..., 1:2])), axis=-1)  # [..., num_syms/2, 2]
    y1 = tf.reshape(y1, (*y1.shape[:-2], -1, 1))  # [..., num_syms, 1]

    # s1, conj(s0) for antenna 2
    y2 = tf.concat((x[..., 1:2], tf.math.conj(x[..., 0:1])), axis=-1)  # [..., num_syms/2, 2]
    y2 = tf.reshape(y2, (*y2.shape[:-2], -1, 1))  # [..., num_syms, 1]

    # combined output for both antennas (last dimension is tx antenna index)
    y = tf.concat((y1, y2), axis=-1)  # [..., num_syms, 2]

    return y


def stbc_decode(y, h):
    """
    Space-time block code (STBC) decoder
    :param y: received symbols for two antennas, shape [..., num_syms, 2]
    :param h: channel estimation for channels, shape [..., num_syms, 2]
    :return: estimation symbols, shape [..., num_syms]
    """

    # check input data dimension
    assert y.shape == h.shape, "channel estimation must have matched shape as received symbols"
    assert h.shape[-1] == 2, "total number of tx antennas must be two"
    assert h.shape[-2] % 2 == 0, "total number of symbols must be even"

    # split received symbols into two sets (y0, y1), last dimension is receive antenna index
    # see table III in reference
    # r0 = y[..., 0, 0], r2 = y[..., 0, 1]
    # r1 = y[..., 1, 0], r3 = y[..., 1, 1]
    y = tf.reshape(y, (*y.shape[:-2], -1, 2, 2))

    # --------------------------------------
    #   maximum ratio combining for STBC
    # --------------------------------------

    # reshape channel coefficients as [..., nrx, ntx], last dimension is transmit antenna index
    # see table II in reference
    # h0 = h[..., 0, 0], h1 = h[..., 0, 1]
    # h2 = h[..., 1, 0], h3 = h[..., 1, 1]
    h = tf.reshape(h, (*h.shape[:-2], -1, 2, 2))

    # s0 = conj(h0) * r0 + h1 * conj(r1) + conj(h2) * r2 + h3 * conj(r3)
    x1 = tf.math.conj(h[..., 0, 0]) * y[..., 0, 0] + h[..., 0, 1] * tf.math.conj(y[..., 1, 0]) \
       + tf.math.conj(h[..., 1, 0]) * y[..., 0, 1] + h[..., 1, 1] * tf.math.conj(y[..., 1, 1])
    x1 = tf.expand_dims(x1, axis=-1)  # [..., num_syms/2, 1]

    # s1 = conj(h1) * r0 - h0 * conj(r1) + conj(h3) * r2 - h2 * conj(r3)
    x2 = tf.math.conj(h[..., 0, 1]) * y[..., 0, 0] - h[..., 0, 0] * tf.math.conj(y[..., 1, 0]) \
       + tf.math.conj(h[..., 1, 1]) * y[..., 0, 1] - h[..., 1, 0] * tf.math.conj(y[..., 1, 1])
    x2 = tf.expand_dims(x2, axis=-1)  # [..., num_syms/2, 1]

    # combine two sets of symbols and make proper shape
    x = tf.concat((x1, x2), axis=-1)  # [..., num_syms/2, 2]
    x = tf.reshape(x, (*x.shape[:-2], -1))  # [..., num_syms]

    # calculate combining gain per 2x2 channel (CSI)
    s = tf.math.real(h * tf.math.conj(h))
    s = tf.reduce_sum(s, axis=[-1, -2])
    # duplicate CSI for two consecutive symbols
    s = tf.expand_dims(s, axis=-1)
    s = tf.concat((s, s), axis=-1)
    s = tf.reshape(s, (*s.shape[:-2], -1))

    return x, s


# Module test
if __name__ == "__main__":
    # Import sionna modules
    from sionna.utils import BinarySource
    from sionna.mapping import Mapper, Demapper
    from sionna.channel import AWGN
    from sionna.utils import ebnodb2no
    from sionna.utils.metrics import compute_ber

    # Simulation params
    batch_size = 64
    num_frames = 512
    num_symbols = 14  # must be even
    num_bits_per_symbol = 2  # QPSK
    ebno_db = 10.0
    no = ebnodb2no(ebno_db, num_bits_per_symbol, 1.0)

    # Create layer/modules
    binary_source = BinarySource()
    mapper = Mapper("qam", num_bits_per_symbol)
    demapper = Demapper("maxlog", "qam", num_bits_per_symbol, hard_out=True)
    add_noise = AWGN()

    # Transmitter processing
    s = binary_source([batch_size, num_frames, num_symbols * num_bits_per_symbol])
    x = mapper(s)  # [..., num_syms]
    tx = stbc_encode(x)  # [..., num_syms, ntx]

    # Generate Rayleigh fading channel coefficients
    # h has the same shape as tx for convenience
    # two consecutive symbols have same 2x2 channel coefficients:
    #   [..., num_syms, ntx] -> [..., num_syms/2, 2, 2]
    h = tf.complex(tf.math.sqrt(0.25), 0.0) * tf.complex(tf.random.normal(tx.shape), tf.random.normal(tx.shape))

    # reshape tx and h as [..., num_syms/2, nrx, ntx] for 2rx-2tx channels
    tx = tf.reshape(tx, (*tx.shape[:-2], -1, 2, 2))  # [..., num_syms/2, nss, ntx]
    hh = tf.reshape(h, (*h.shape[:-2], -1, 2, 2))    # [..., num_syms/2, nrx, ntx]

    # Channel processing
    ry = tf.linalg.matmul(tx, hh, transpose_b=True)  # [..., num_syms/2, nss, nrx]
    ry = tf.reshape(ry, (*ry.shape[:-3], -1, 2))  # [..., num_syms, nrx]
    ry = add_noise([ry, no])

    # Receiver processing
    yd, csi = stbc_decode(ry, h)  # assuming perfect CSI

    # Demapping
    yd = yd / tf.cast(csi, tf.complex64)  # CSI scaling
    d = demapper([yd, no / csi])

    # Estimate BER
    avg_ber = compute_ber(d, s)
    print("Simulation of STBC in Rayleigh fading channel")
    print("EbNo: {:.1f}dB  BER: {:.2e}".format(ebno_db, avg_ber))

