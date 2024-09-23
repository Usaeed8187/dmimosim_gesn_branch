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
import tensorflow as tf

from dmimo.config import SimConfig
from dmimo.mu_mimo_kpi import sim_mu_mimo_all

# Main function
if __name__ == "__main__":

    # Simulation settings
    cfg = SimConfig()
    cfg.total_slots = 250        # total number of slots in ns-3 channels
    cfg.start_slot_idx = 30     # starting slots (must be greater than csi_delay + 5)
    cfg.csi_delay = 4           # feedback delay in number of subframe
    cfg.cfo_sigma = 0.0         # in Hz
    cfg.sto_sigma = 0.0         # in nanosecond
    cfg.num_tx_ue_sel = 8
    mobility = 'medium_mobility'
    cfg.ns3_folder = "ns3/channels_" + mobility + '/'

    folder_name = os.path.basename(os.path.abspath(cfg.ns3_folder))
    os.makedirs(os.path.join("results", folder_name), exist_ok=True)
    print("Using channels in {}".format(folder_name))

    rx_ues_arr = [1,2,4,6]

    ber = np.zeros(np.size(rx_ues_arr ))
    ldpc_ber = np.zeros(np.size(rx_ues_arr ))
    goodput = np.zeros(np.size(rx_ues_arr ))
    throughput = np.zeros(np.size(rx_ues_arr ))
    bitrate = np.zeros(np.size(rx_ues_arr ))
    nodewise_goodput = []
    nodewise_throughput = []
    nodewise_bitrate = []
    ranks = []
    ldpc_ber_list = []
    uncoded_ber_list = []
    sinr_dB = []


    # cfg.num_rx_ue_sel = 7  # TODO consolidate params
    # cfg.ue_indices = np.reshape(np.arange((cfg.num_rx_ue_sel + 2) * 2), (cfg.num_rx_ue_sel + 2, -1))
    # cfg.ue_ranks = [2]  # same rank for all UEs
    # cfg.modulation_order = 2

    # Modulation order: 2/4/6 for QPSK/16QAM/64QAM
    # modulation_orders = [2, 4, 6]

    #############################################
    # Testing with rank and link adaptation
    #############################################

    cfg.rank_adapt = True
    cfg.link_adapt = True
    cfg.csi_prediction = True

    for ue_arr_idx in range(np.size(rx_ues_arr)):

        cfg.num_rx_ue_sel = rx_ues_arr[ue_arr_idx]
        cfg.ue_indices = np.reshape(np.arange((cfg.num_rx_ue_sel + 2) * 2), (cfg.num_rx_ue_sel + 2, -1))

        cfg.precoding_method = "ZF"
        rst_bd = sim_mu_mimo_all(cfg)
        ber[ue_arr_idx] = rst_bd[0]
        ldpc_ber[ue_arr_idx] = rst_bd[1]
        goodput[ue_arr_idx] = rst_bd[2]
        throughput[ue_arr_idx] = rst_bd[3]
        bitrate[ue_arr_idx] = rst_bd[4]
        
        nodewise_goodput.append(rst_bd[5])
        nodewise_throughput.append(rst_bd[6])
        nodewise_bitrate.append(rst_bd[7])
        ranks.append(rst_bd[8])
        uncoded_ber_list.append(rst_bd[9])
        ldpc_ber_list.append(rst_bd[10])
        sinr_dB.append(np.concatenate(rst_bd[11]))

        if cfg.csi_prediction:
            np.savez("results/{}/mu_mimo_results_UE_{}_pred.npz".format(folder_name, rx_ues_arr[ue_arr_idx]),
                    ber=ber, ldpc_ber=ldpc_ber, goodput=goodput, throughput=throughput, bitrate=bitrate, nodewise_goodput=rst_bd[5],
                    nodewise_throughput=rst_bd[6], nodewise_bitrate=rst_bd[7], ranks=rst_bd[8], uncoded_ber_list=rst_bd[9],
                    ldpc_ber_list=rst_bd[10], sinr_dB=rst_bd[11])
        else:
            np.savez("results/{}/mu_mimo_results_UE_{}.npz".format(folder_name, rx_ues_arr[ue_arr_idx]),
                    ber=ber, ldpc_ber=ldpc_ber, goodput=goodput, throughput=throughput, bitrate=bitrate, nodewise_goodput=rst_bd[5],
                    nodewise_throughput=rst_bd[6], nodewise_bitrate=rst_bd[7], ranks=rst_bd[8], uncoded_ber_list=rst_bd[9],
                    ldpc_ber_list=rst_bd[10], sinr_dB=rst_bd[11])
    
    #############################################
    # Test for beamforming gain
    #############################################

    # cfg.rank_adapt = False
    # cfg.link_adapt = False

    # cfg.ue_ranks = [1]  # same rank for all UEs
    # cfg.code_rate = 0.5
    # cfg.modulation_order = 2

    # num_tx_streams = (cfg.num_tx_ue_sel+2) * cfg.ue_ranks[0]
    # cfg.num_tx_streams = num_tx_streams
    # cfg.num_rx_ue_sel = (num_tx_streams - 4) // 2  # TODO consolidate params
    # cfg.ue_indices = np.reshape(np.arange((cfg.num_rx_ue_sel + 2) * 2), (cfg.num_rx_ue_sel + 2, -1))
    

    # cfg.precoding_method = "None"
    # rst_bd = sim_mu_mimo_all(cfg)
    # ber[1] = rst_bd[0]
    # ldpc_ber[1] = rst_bd[1]
    # goodput[1] = rst_bd[2]
    # throughput[1] = rst_bd[3]

    #############################################
    # Testing without rank and link adaptation
    #############################################
    
    # # Test 1 parameters
    # num_tx_streams = 6
    # cfg.num_tx_streams = num_tx_streams
    # cfg.num_rx_ue_sel = (num_tx_streams - 4) // 2  # TODO consolidate params
    # cfg.ue_indices = np.reshape(np.arange((cfg.num_rx_ue_sel + 2) * 2), (cfg.num_rx_ue_sel + 2, -1))
    # cfg.ue_ranks = [2]  # same rank for all UEs
    # cfg.code_rate = 0.5
    # cfg.modulation_order = 2

    # cfg.precoding_method = "ZF"
    # rst_svd = sim_mu_mimo_all(cfg)
    # ber[2] = rst_svd[0]
    # ldpc_ber[2] = rst_svd[1]
    # goodput[2] = rst_svd[2]
    # throughput[2] = rst_svd[3]

    # # Test 2 parameters
    # num_tx_streams = 12
    # cfg.num_tx_streams = num_tx_streams
    # cfg.num_rx_ue_sel = (num_tx_streams - 4) // 2  # TODO consolidate params
    # cfg.ue_indices = np.reshape(np.arange((cfg.num_rx_ue_sel + 2) * 2), (cfg.num_rx_ue_sel + 2, -1))
    # cfg.ue_ranks = [2]  # same rank for all UEs
    # cfg.code_rate = 0.5
    # cfg.modulation_order = 4

    # cfg.precoding_method = "ZF"
    # rst_svd = sim_mu_mimo_all(cfg)
    # ber[3] = rst_svd[0]
    # ldpc_ber[3] = rst_svd[1]
    # goodput[3] = rst_svd[2]
    # throughput[3] = rst_svd[3]

