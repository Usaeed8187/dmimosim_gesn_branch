"""
Simulation of STBC performance in Rayleigh fading channels
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
from sionna.utils import ebnodb2no
from sionna.utils.metrics import compute_ber

from dmimo.sttd import stbc_encode, stbc_decode


def sim_stbc_rayleigh(ebno_db, batch_size=128, num_frames=512, num_symbols=14, num_bits_per_symbol=2):
    # Create layer/modules
    binary_source = BinarySource()
    mapper = Mapper("qam", num_bits_per_symbol)
    demapper = Demapper("maxlog", "qam", num_bits_per_symbol, hard_out=True)
    add_noise = AWGN()

    # Main simulation loop over EbNo settings
    runs = len(ebno_db)
    avg_ber = np.zeros(runs)
    for k in range(runs):
        # Calibrate noise variance
        no = ebnodb2no(ebno_db[k], num_bits_per_symbol, 1.0)

        # Transmitter processing
        s = binary_source([batch_size, num_frames, num_symbols * num_bits_per_symbol])
        x = mapper(s)  # [..., num_syms]
        tx = stbc_encode(x)  # [..., num_syms, num_tx_ant]

        # Generate Rayleigh fading channel coefficients
        # h has the same shape as tx for convenience
        # two consecutive symbols have same 2x2 channel coefficients:
        #   [..., num_syms, num_tx_ant] -> [..., num_syms/2, 2, 2]
        h = tf.complex(tf.math.sqrt(0.25), 0.0) * tf.complex(tf.random.normal(tx.shape), tf.random.normal(tx.shape))

        # reshape tx and h as [..., num_syms/2, num_rx_ant, num_tx_ant] for 2rx-2tx channels
        tx = tf.reshape(tx, (*tx.shape[:-2], -1, 2, 2))  # [..., num_syms/2, num_substream, num_tx_ant]
        hh = tf.reshape(h, (*h.shape[:-2], -1, 2, 2))    # [..., num_syms/2, nm_rx_ant, num_tx_ant]

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
        avg_ber[k] = compute_ber(d, s)

    # QAM constellation for debugging
    qam_const = yd[0]  # first batch of the last run

    return avg_ber, qam_const


# Main function
if __name__ == "__main__":
    ebno_db = np.arange(0.0, 15.0, 1.0)
    stbc_ber, stbc_const = sim_stbc_rayleigh(ebno_db=ebno_db)
    stbc_const = np.reshape(stbc_const, (-1))

    fig, ax = plt.subplots(1,2, figsize=(10,4))
    ax[0].set_xlabel("Real")
    ax[0].set_ylabel("Imag")
    ax[0].scatter(stbc_const.real, stbc_const.imag)
    ax[1].set_xlabel("EbNo (dB)")
    ax[1].set_ylabel("BER")
    ax[1].grid(True, 'both')
    ax[1].set_ylim([10 ** -6, 10 ** -1])
    ax[1].semilogy(ebno_db, stbc_ber, 'o-')

    plt.show()
