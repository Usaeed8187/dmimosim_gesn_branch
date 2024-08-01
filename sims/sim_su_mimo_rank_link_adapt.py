"""
Simulation of SU-MIMO scenario with ns-3 channels

This scripts should be called from the "tests" folder
"""

# add system folder for the dmimo library
import sys
import os
sys.path.append(os.path.join('..'))

import matplotlib.pyplot as plt
import numpy as np

from dmimo.config import SimConfig
from dmimo.su_mimo import sim_su_mimo_all

gpu_num = 1  # Use "" to use the CPU, Use 0 to select first GPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Main function
if __name__ == "__main__":

    # Simulation settings
    cfg = SimConfig()
    cfg.total_slots = 35        # total number of slots in ns-3 channels
    cfg.start_slot_idx = 15     # starting slots (must be greater than csi_delay + 5)
    cfg.csi_delay = 9           # feedback delay in number of subframe
    cfg.cfo_sigma = 0.0         # in Hz
    cfg.sto_sigma = 0.0         # in nanosecond
    cfg.ns3_folder = "ns3/channels/"
    cfg.rank_adapt = True
    cfg.link_adapt = True

    folder_name = os.path.basename(os.path.abspath(cfg.ns3_folder))
    os.makedirs(os.path.join("../results", folder_name), exist_ok=True)
    print("\n Using channels in {}".format(folder_name))

    cfg.precoding_method = "SVD"
    rst_svd = sim_su_mimo_all(cfg)
    ber[0, k] = rst_svd[0]
    ldpc_ber[0, k] = rst_svd[1]
    goodput[0, k] = rst_svd[2]
    throughput[0, k] = rst_svd[3]

    cfg.precoding_method = "ZF"
    rst_zf = sim_su_mimo_all(cfg)
    ber[1, k] = rst_zf[0]
    ldpc_ber[1, k] = rst_zf[1]
    goodput[1, k] = rst_zf[2]
    throughput[1, k] = rst_zf[3]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].set_title("SU-MIMO")
    ax[0].set_xlabel('Modulation (bits/symbol)')
    ax[0].set_ylabel('BER')
    ax[0].plot(modulation_orders, ber.transpose(), 'o-')
    ax[0].legend(['SVD', 'ZF'])

    ax[1].set_title("SU-MIMO")
    ax[1].set_xlabel('Modulation (bits/symbol)')
    ax[1].set_ylabel('Coded BER')
    ax[1].plot(modulation_orders, ldpc_ber.transpose(), 'd-')
    ax[1].legend(['SVD', 'ZF'])

    ax[2].set_title("SU-MIMO")
    ax[2].set_xlabel('Modulation (bits/symbol)')
    ax[2].set_ylabel('Goodput/Throughput (Mbps)')
    ax[2].plot(modulation_orders, goodput.transpose(), 's-')
    ax[2].plot(modulation_orders, throughput.transpose(), 'd-')
    ax[2].legend(['Goodput-SVD', 'Goodput-ZF', 'Throughput-SVD', 'Throughput-ZF'])

    plt.savefig("../results/{}/su_mimo_results_s{}.png".format(folder_name, cfg.num_tx_streams))

    np.savez("../results/{}/su_mimo_results_s{}.npz".format(folder_name, cfg.num_tx_streams),
                ber=ber, ldpc_ber=ldpc_ber, goodput=goodput, throughput=throughput)

