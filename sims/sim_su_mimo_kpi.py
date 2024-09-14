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
from dmimo.su_mimo_kpi import sim_su_mimo_all

gpu_num = 1  # Use "" to use the CPU, Use 0 to select first GPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Main function
if __name__ == "__main__":

    # Simulation settings
    cfg = SimConfig()
    cfg.total_slots = 100        # total number of slots in ns-3 channels
    cfg.start_slot_idx = 75     # starting slots (must be greater than csi_delay + 5)
    cfg.csi_delay = 9           # feedback delay in number of subframe
    cfg.cfo_sigma = 0.0         # in Hz
    cfg.sto_sigma = 0.0         # in nanosecond
    mobility = 'medium_mobility'
    cfg.ns3_folder = "ns3/channels_" + mobility + '/'

    folder_name = os.path.basename(os.path.abspath(cfg.ns3_folder))
    os.makedirs(os.path.join("results", folder_name), exist_ok=True)
    print("\n Using channels in {}".format(folder_name))

    # ber = np.zeros(3)
    # ldpc_ber = np.zeros(3)
    # goodput = np.zeros(3)
    # throughput = np.zeros(3)

    #############################################
    # Testing with rank and link adaptation
    #############################################

    cfg.rank_adapt = True
    cfg.link_adapt = True
    cfg.csi_prediction = True
    
    cfg.precoding_method = "ZF"
    rst_zf = sim_su_mimo_all(cfg)
    ber = rst_zf[0]
    ldpc_ber = rst_zf[1]
    goodput = rst_zf[2]
    throughput = rst_zf[3]
    bitrate = rst_zf[4]

    ranks = rst_zf[5]
    ldpc_ber_list = rst_zf[6]
    uncoded_ber_list = rst_zf[7]

    if cfg.csi_prediction:
        np.savez("results/{}/su_mimo_results_pred.npz".format(folder_name),
                    ber=ber, ldpc_ber=ldpc_ber, goodput=goodput, throughput=throughput, bitrate=bitrate, ranks=ranks, uncoded_ber_list=uncoded_ber_list,
                    ldpc_ber_list=ldpc_ber_list)
    else:
        np.savez("results/{}/su_mimo_results.npz".format(folder_name),
                    ber=ber, ldpc_ber=ldpc_ber, goodput=goodput, throughput=throughput, bitrate=bitrate, ranks=ranks, uncoded_ber_list=uncoded_ber_list,
                    ldpc_ber_list=ldpc_ber_list)


    #############################################
    # Testing without rank and link adaptation
    #############################################
    
    # cfg.rank_adapt = False
    # cfg.link_adapt = False
    
    # # Test 1 parameters
    # cfg.num_tx_streams = 2
    # cfg.modulation_order = 2
    # cfg.code_rate = 0.5

    # cfg.precoding_method = "ZF"
    # rst_svd = sim_su_mimo_all(cfg)
    # ber[1] = rst_svd[0]
    # ldpc_ber[1] = rst_svd[1]
    # goodput[1] = rst_svd[2]
    # throughput[1] = rst_svd[3]

    # # Test 2 parameters
    # cfg.num_tx_streams = 6
    # cfg.modulation_order = 6
    # cfg.code_rate = 0.5

    # cfg.precoding_method = "ZF"
    # rst_svd = sim_su_mimo_all(cfg)
    # ber[2] = rst_svd[0]
    # ldpc_ber[2] = rst_svd[1]
    # goodput[2] = rst_svd[2]
    # throughput[2] = rst_svd[3]

    # np.savez("../results/{}/su_mimo_results_s{}.npz".format(folder_name, cfg.num_tx_streams),
    #             ber=ber, ldpc_ber=ldpc_ber, goodput=goodput, throughput=throughput)
