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
from dmimo.mu_mimo_adapt import sim_mu_mimo_all


# Main function
if __name__ == "__main__":

    # Simulation settings
    cfg = SimConfig()
    cfg.total_slots = 35        # total number of slots in ns-3 channels
    cfg.start_slot_idx = 15     # starting slots (must be greater than csi_delay + 5)
    cfg.csi_delay = 4           # feedback delay in number of subframe
    cfg.cfo_sigma = 0.0         # in Hz
    cfg.sto_sigma = 0.0         # in nanosecond
    cfg.ns3_folder = "ns3/channels_medium_mobility/"

    folder_name = os.path.basename(os.path.abspath(cfg.ns3_folder))
    os.makedirs(os.path.join("../results", folder_name), exist_ok=True)
    print("Using channels in {}".format(folder_name))

    cfg.num_rx_ue_sel = 6
    cfg.ue_indices = np.reshape(np.arange((cfg.num_rx_ue_sel + 2) * 2), (cfg.num_rx_ue_sel + 2, -1))

    ber = np.zeros(3)
    ldpc_ber = np.zeros(3)
    goodput = np.zeros(3)
    throughput = np.zeros(3)

    # cfg.num_rx_ue_sel = 7  # TODO consolidate params
    # cfg.ue_indices = np.reshape(np.arange((cfg.num_rx_ue_sel + 2) * 2), (cfg.num_rx_ue_sel + 2, -1))
    # cfg.ue_ranks = [2]  # same rank for all UEs
    # cfg.modulation_order = 2

    # Modulation order: 2/4/6 for QPSK/16QAM/64QAM
    # modulation_orders = [2, 4, 6]

    #############################################
    # Testing with rank and link adaptation
    #############################################

    cfg.precoding_method = "ZF"
    rst_bd = sim_mu_mimo_all(cfg)
    ber[0] = rst_bd[0]
    ldpc_ber[0] = rst_bd[1]
    goodput[0] = rst_bd[2]
    throughput[0] = rst_bd[3]

    #############################################
    # Testing without rank and link adaptation
    #############################################
    
    cfg.rank_adapt = False
    cfg.link_adapt = False
    
    # Test 1 parameters
    num_tx_streams = 6
    cfg.num_tx_streams = num_tx_streams
    cfg.num_rx_ue_sel = (num_tx_streams - 4) // 2  # TODO consolidate params
    cfg.ue_indices = np.reshape(np.arange((cfg.num_rx_ue_sel + 2) * 2), (cfg.num_rx_ue_sel + 2, -1))
    cfg.ue_ranks = [2]  # same rank for all UEs
    cfg.code_rate = 0.5
    cfg.modulation_order = 2

    cfg.precoding_method = "ZF"
    rst_svd = sim_mu_mimo_all(cfg)
    ber[1] = rst_svd[0]
    ldpc_ber[1] = rst_svd[1]
    goodput[1] = rst_svd[2]
    throughput[1] = rst_svd[3]

    # Test 2 parameters
    num_tx_streams = 12
    cfg.num_tx_streams = num_tx_streams
    cfg.num_rx_ue_sel = (num_tx_streams - 4) // 2  # TODO consolidate params
    cfg.ue_indices = np.reshape(np.arange((cfg.num_rx_ue_sel + 2) * 2), (cfg.num_rx_ue_sel + 2, -1))
    cfg.ue_ranks = [2]  # same rank for all UEs
    cfg.code_rate = 0.5
    cfg.modulation_order = 4

    cfg.precoding_method = "ZF"
    rst_svd = sim_mu_mimo_all(cfg)
    ber[2] = rst_svd[0]
    ldpc_ber[2] = rst_svd[1]
    goodput[2] = rst_svd[2]
    throughput[2] = rst_svd[3]

    np.savez("../results/{}/mu_mimo_results_s{}.npz".format(folder_name, cfg.num_tx_streams),
                ber=ber, ldpc_ber=ldpc_ber, goodput=goodput, throughput=throughput)

