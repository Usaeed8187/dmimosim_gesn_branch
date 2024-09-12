"""
Simulation of SU-MIMO scenario with ns-3 channels

This scripts should be called from the "sims" folder
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

gpu_num = 0  # Use "" to use the CPU, Use 0 to select first GPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['DRJIT_LIBLLVM_PATH'] = '/usr/lib/llvm/16/lib64/libLLVM.so'

# Configure to use only a single GPU and allocate only as much memory as needed
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel('ERROR')

# add system folder for the dmimo library
sys.path.append(os.path.join('..'))

from dmimo.config import SimConfig
from dmimo.su_mimo_chanpred import sim_su_mimo_chanpred_all


# Main function
if __name__ == "__main__":

    # Simulation settings
    cfg = SimConfig()
    cfg.total_slots = 90        # total number of slots in ns-3 channels
    cfg.start_slot_idx = 70     # starting slots (must be greater than csi_delay + 5)
    cfg.csi_delay = 8           # feedback delay in number of subframe
    cfg.cfo_sigma = 0.0         # in Hz
    cfg.sto_sigma = 0.0         # in nanosecond
    cfg.ns3_folder = "../ns3/channels_medium_mobility/"

    folder_name = os.path.basename(os.path.abspath(cfg.ns3_folder))
    os.makedirs(os.path.join("../results", folder_name), exist_ok=True)
    print("Using channels in {}".format(folder_name))

    for num_tx_streams in [4, 5, 6, 7, 8]:
        # 4/5/7/8 equal to total number of streams
        cfg.num_tx_streams = num_tx_streams

        # Modulation order: 2/4/6 for QPSK/16QAM/64QAM
        modulation_orders = [2, 4, 6]
        num_modulations = len(modulation_orders)
        ber = np.zeros((2, num_modulations))
        ldpc_ber = np.zeros((2, num_modulations))
        goodput = np.zeros((2, num_modulations))
        throughput = np.zeros((2, num_modulations))
        bitrate = np.zeros((2, num_modulations))

        for k in range(num_modulations):
            cfg.modulation_order = modulation_orders[k]

            cfg.csi_prediction = True
            cfg.precoding_method = "SVD"
            rst_svd = sim_su_mimo_chanpred_all(cfg)
            ber[0, k] = rst_svd[0]
            ldpc_ber[0, k] = rst_svd[1]
            goodput[0, k] = rst_svd[2]
            throughput[0, k] = rst_svd[3]
            bitrate[0, k] = rst_svd[4]

            cfg.csi_prediction = True
            cfg.precoding_method = "ZF"
            rst_zf = sim_su_mimo_chanpred_all(cfg)
            ber[1, k] = rst_zf[0]
            ldpc_ber[1, k] = rst_zf[1]
            goodput[1, k] = rst_zf[2]
            throughput[1, k] = rst_zf[3]
            bitrate[1, k] = rst_zf[4]

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
        ax[2].plot(modulation_orders, bitrate.transpose(), '*-')
        ax[2].legend(['Goodput-SVD', 'Goodput-ZF', 'Throughput-SVD', 'Throughput-ZF', 'Bitrate-SVD', 'Bitrate-ZF'])

        plt.savefig("../results/{}/su_mimo_results_chanpred_s{}.png".format(folder_name, cfg.num_tx_streams))

        np.savez("../results/{}/su_mimo_results_chanpred_s{}.npz".format(folder_name, cfg.num_tx_streams),
                 ber=ber, ldpc_ber=ldpc_ber, goodput=goodput, throughput=throughput)

