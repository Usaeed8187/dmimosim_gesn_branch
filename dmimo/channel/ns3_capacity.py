# Estimate channel capacity from ns3-channel coefficients
import numpy as np


def estimate_capacity(h, snrdb=10.0):
    # Inputs:
    #   h: channel coefficients of shape [num_tx_ant, num_rx_ant, num_ofdm_sym, num_subcarrier]
    #   snrdb: signal-to-noise ratio in dB
    # Return:
    #  capacity estimation

    ntx, nrx = h.shape[:2]
    snrdb = snrdb - 10.0 * np.log10(nrx)  # equal power allocation for all streams
    snr = np.power(10.0, snrdb/10.0)

    h = np.reshape(h, (ntx, nrx, -1))
    nsamples = h.shape[-1]

    c_sum = 0
    for k in range(nsamples):
        u, s, vt = np.linalg.svd(h[:, :, k].transpose())
        c = np.log(1.0 + snr * s[:nrx])
        c_sum += np.sum(c, axis=0)

    return c_sum / nsamples


