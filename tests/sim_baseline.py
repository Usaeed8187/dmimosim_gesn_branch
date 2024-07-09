"""
Simulation of baseline scenario with ns-3 channels

This scripts should be called from the "tests" folder
"""

# add system folder for the dmimo library
import sys
import os
sys.path.append(os.path.join('..'))

import matplotlib.pyplot as plt
import numpy as np

from dmimo.config import SimConfig
from dmimo.baseline import sim_baseline_all


# Main function
if __name__ == "__main__":

    # Simulation settings
    cfg = SimConfig()
    cfg.total_slots = 20        # total number of slots in ns-3 channels
    cfg.start_slot_idx = 10     # starting slots (must be greater than csi_delay + 5)
    cfg.csi_delay = 2           # feedback delay in number of subframe
    cfg.num_tx_streams = 2      # 2/4 equal to total number of streams
    cfg.cfo_sigma = 0.0         # in Hz
    cfg.sto_sigma = 0.0         # in nanosecond
    cfg.ns3_folder = "../ns3/channels/"

    # Modulation order: 2/4/6 for QPSK/16QAM/64QAM
    modulation_orders = [2, 4, 6]
    num_modulations = len(modulation_orders)
    ber = np.zeros((2, num_modulations))
    ldpc_ber = np.zeros((2, num_modulations))
    goodput = np.zeros((2, num_modulations))
    throughput = np.zeros((2, num_modulations))

    for k in range(num_modulations):
        cfg.modulation_order = modulation_orders[k]
        rst_svd = sim_baseline_all(cfg, precoding_method="SVD")
        ber[0, k] = rst_svd[0]
        ldpc_ber[0, k] = rst_svd[1]
        goodput[0, k] = rst_svd[2]
        throughput[0, k] = rst_svd[3]
        rst_zf = sim_baseline_all(cfg, precoding_method="ZF")
        ber[1, k] = rst_zf[0]
        ldpc_ber[1, k] = rst_zf[1]
        goodput[1, k] = rst_zf[2]
        throughput[1, k] = rst_zf[3]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].set_title("SU-MIMO (Baseline)")
    ax[0].set_xlabel('Modulation (bits/symbol)')
    ax[0].set_ylabel('BER')
    ax[0].plot(modulation_orders, ber.transpose(), 'o-')
    ax[0].legend(['SVD', 'ZF'])

    ax[1].set_title("SU-MIMO (Baseline)")
    ax[1].set_xlabel('Modulation (bits/symbol)')
    ax[1].set_ylabel('Coded BER')
    ax[1].plot(modulation_orders, ldpc_ber.transpose(), 'd-')
    ax[1].legend(['SVD', 'ZF'])

    ax[2].set_title("SU-MIMO (Baseline)")
    ax[2].set_xlabel('Modulation (bits/symbol)')
    ax[2].set_ylabel('Goodput/Throughput (Mbps)')
    ax[2].plot(modulation_orders, goodput.transpose(), 's-')
    ax[2].plot(modulation_orders, throughput.transpose(), 'd-')
    ax[2].legend(['Goodput-SVD', 'Goodput-ZF', 'Throughput-SVD', 'Throughput-ZF'])

    plt.savefig("../results/baseline_results.png")

    np.savez("../results/baseline_results.npz", ber=ber, ldpc_ber=ldpc_ber,
             goodput=goodput, throughput=throughput)

