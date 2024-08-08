"""
Simulation of MU-MIMO scenario with ns-3 channels

This scripts should be called from the "tests" folder
"""

# add system folder for the dmimo library
import sys
import os
sys.path.append(os.path.join('..'))

import matplotlib.pyplot as plt
import numpy as np

from dmimo.config import SimConfig
from dmimo.mu_mimo import sim_mu_mimo_all


# Main function
if __name__ == "__main__":

    # Simulation settings
    cfg = SimConfig()
    cfg.total_slots = 35        # total number of slots in ns-3 channels
    cfg.start_slot_idx = 15     # starting slots (must be greater than csi_delay + 5)
    cfg.csi_delay = 4           # feedback delay in number of subframe
    cfg.cfo_sigma = 0.0         # in Hz
    cfg.sto_sigma = 0.0         # in nanosecond
    cfg.ns3_folder = "../ns3/channels_medium_mobility/"

    folder_name = os.path.basename(os.path.abspath(cfg.ns3_folder))
    os.makedirs(os.path.join("../results", folder_name), exist_ok=True)
    print("Using channels in {}".format(folder_name))

    for num_tx_streams in [6, 8, 10, 12]:
        # 6/7/8/10/12 equal to total number of streams
        # manual rank adaptation (assuming 2 antennas per UE)
        cfg.num_tx_streams = num_tx_streams
        cfg.num_rx_ue_sel = (num_tx_streams - 4) // 2  # TODO consolidate params
        cfg.ue_indices = np.reshape(np.arange((cfg.num_rx_ue_sel + 2) * 2), (cfg.num_rx_ue_sel + 2, -1))
        cfg.ue_ranks = [2]  # same rank for all UEs

        # Modulation order: 2/4/6 for QPSK/16QAM/64QAM
        modulation_orders = [2, 4, 6]
        num_modulations = len(modulation_orders)
        ber = np.zeros((2, num_modulations))
        ldpc_ber = np.zeros((2, num_modulations))
        goodput = np.zeros((2, num_modulations))
        throughput = np.zeros((2, num_modulations))

        for k in range(num_modulations):
            cfg.modulation_order = modulation_orders[k]

            cfg.precoding_method = "BD"
            rst_bd = sim_mu_mimo_all(cfg)
            ber[0, k] = rst_bd[0]
            ldpc_ber[0, k] = rst_bd[1]
            goodput[0, k] = rst_bd[2]
            throughput[0, k] = rst_bd[3]

            cfg.precoding_method = "ZF"
            rst_zf = sim_mu_mimo_all(cfg)
            ber[1, k] = rst_zf[0]
            ldpc_ber[1, k] = rst_zf[1]
            goodput[1, k] = rst_zf[2]
            throughput[1, k] = rst_zf[3]

        fig, ax = plt.subplots(1, 3, figsize=(15, 4))

        ax[0].set_title("MU-MIMO")
        ax[0].set_xlabel('Modulation (bits/symbol)')
        ax[0].set_ylabel('BER')
        ax[0].plot(modulation_orders, ber.transpose(), 'o-')
        ax[0].legend(['BD', 'ZF'])

        ax[1].set_title("MU-MIMO")
        ax[1].set_xlabel('Modulation (bits/symbol)')
        ax[1].set_ylabel('Coded BER')
        ax[1].plot(modulation_orders, ldpc_ber.transpose(), 'd-')
        ax[1].legend(['BD', 'ZF'])

        ax[2].set_title("MU-MIMO")
        ax[2].set_xlabel('Modulation (bits/symbol)')
        ax[2].set_ylabel('Goodput/Throughput (Mbps)')
        ax[2].plot(modulation_orders, goodput.transpose(), 's-')
        ax[2].plot(modulation_orders, throughput.transpose(), 'd-')
        ax[2].legend(['Goodput-BD', 'Goodput-ZF', 'Throughput-BD', 'Throughput-ZF'])

        plt.savefig("../results/{}/mu_mimo_results_s{}.png".format(folder_name, cfg.num_tx_streams))

        np.savez("../results/{}/mu_mimo_results_s{}.npz".format(folder_name, cfg.num_tx_streams),
                 ber=ber, ldpc_ber=ldpc_ber, goodput=goodput, throughput=throughput)

