"""
Simulation of STTD performance in Rayleigh fading channels
"""

# this scripts should be called from the project root folder
# add system folder for the dmimo library
import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sionna.utils import BinarySource
from sionna.mapping import Mapper, Demapper
from sionna.channel import AWGN
from sionna.utils import ebnodb2no, flatten_dims
from sionna.utils.metrics import compute_ber

from dmimo.sttd import stbc_encode, stbc_decode


def sim_sttd_rayleigh(ebno_db, num_rx=10, batch_size=128, num_bits_per_symbol=2):
    runs = len(ebno_db)
    avg_ber = np.zeros(runs)

    num_frames=512
    num_symbols=14
    num_tx = 1  # assuming only one transmitter
    num_tx_ant, num_rx_ant = 2, 2

    # Create layer/modules
    binary_source = BinarySource()
    mapper = Mapper("qam", num_bits_per_symbol)
    demapper = Demapper("maxlog", "qam", num_bits_per_symbol, hard_out=True)
    add_noise = AWGN()

    # Main simulation loop over EbNo settings
    for k in range(runs):
        # Calibrate noise variance
        no = ebnodb2no(ebno_db[k], num_bits_per_symbol, 1.0)

        # Transmitter processing
        s = binary_source([batch_size, num_frames, num_symbols * num_bits_per_symbol])
        x = mapper(s)
        tx = stbc_encode(x)  # [batch_size, num_frames, num_syms, num_tx * num_tx_ant]

        # Generate Rayleigh fading channel coefficients
        h_shape = (batch_size, num_frames, num_symbols//2, num_rx * num_rx_ant, num_tx * num_tx_ant)
        h = tf.complex(tf.math.sqrt(0.25), 0.0) * tf.complex(tf.random.normal(h_shape), tf.random.normal(h_shape))
        # two consecutive symbols have the same channel coefficients
        ch = tf.expand_dims(h, axis=3)
        ch = tf.concat((ch, ch), axis=-3)  # duplicate channels for two consecutive symbols
        ch = flatten_dims(ch, num_dims=2, axis=2)  # [..., num_syms, num_rx * num_rx_ant, num_tx * num_tx_ant)

        # Channel processing
        tx = tf.expand_dims(tx, -1)  # [..., num_syms, num_tx * num_tx_ant, 1]
        ry = tf.linalg.matmul(ch, tx)  # [..., num_syms, num_rx * nm_rx_ant, 1]
        ry = add_noise([ry, no])

        # Reshape ry for STBC decoding
        ry = tf.reshape(ry, (*ry.shape[:-2], num_rx, num_rx_ant))  # [batch_size, num_frames, num_syms, num_rx, num_rx_ant]
        ry = tf.transpose(ry, [0, 1, 3, 2, 4]) # [batch_size, num_frames, num_rx, num_syms, num_rx_ant]

        # Reshape h for STBC decoding
        hy = tf.reshape(h, (batch_size, num_frames, num_symbols//2, num_rx, num_rx_ant, num_tx_ant))  # num_tx = 1
        hy = tf.transpose(hy, [0, 1, 3, 2, 4, 5])  # [batch_size, num_frames, num_rx, num_syms//2, num_rx_ant, num_tx_ant]
        hy = flatten_dims(hy, num_dims=2, axis=3)  # [batch_size, num_frames, num_syms, num_rx_ant, num_tx_ant], num_rx_ant=2

        # Receiver processing
        # assuming perfect CSI
        yd, csi = stbc_decode(ry, hy)  # [batch_size, num_frames, num_rx, num_syms]

        # Combining for all receivers
        yd = tf.reduce_mean(yd, axis=2)
        csi = tf.reduce_mean(csi, axis=2)

        # Demapping
        yd = yd / tf.cast(csi, tf.complex64)  # CSI scaling
        d = demapper([yd, no / csi])

        # Estimate BER
        avg_ber[k] = compute_ber(d, s)

    # QAM constellation for debugging
    qam_const = yd[0]  # first batch of the last run

    return avg_ber, qam_const


# Main function
if __name__ == "__main__":
    ebno_db = np.arange(0.0, 15.0, 1.0)

    sttd_ber_1rx, sttd_y1 = sim_sttd_rayleigh(ebno_db=ebno_db, num_rx=1)
    sttd_ber_2rx, sttd_y2 = sim_sttd_rayleigh(ebno_db=ebno_db - 3.0, num_rx=2)  # EbNo normalized by number of receiver antennas
    sttd_ber_4rx, sttd_y4 = sim_sttd_rayleigh(ebno_db=ebno_db - 6.0, num_rx=4)  # EbNo normalized by number of receiver antennas
    sttd_y1 = np.reshape(sttd_y1, (-1))
    sttd_y2 = np.reshape(sttd_y2, (-1))
    sttd_y4 = np.reshape(sttd_y4, (-1))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].set_xlabel("Real")
    ax[0].set_ylabel("Imag")
    im = ax[0].scatter(sttd_y2.real, sttd_y2.imag)
    ax[1].set_xlabel("Real")
    ax[1].set_ylabel("Imag")
    ax[1].scatter(sttd_y4.real, sttd_y4.imag)
    ax[2].set_xlabel("EbNo (dB)")
    ax[2].set_ylabel("BER")
    ax[2].grid(True, 'both')
    ax[2].set_ylim([10 ** -6, 10 ** -1])
    ax[2].semilogy(ebno_db, sttd_ber_1rx, '*-')
    ax[2].semilogy(ebno_db, sttd_ber_2rx, 'o-')
    ax[2].semilogy(ebno_db, sttd_ber_4rx, 'd-')
    ax[2].legend(["1-UE", "2-UE", "4-UE"])

    plt.show()
