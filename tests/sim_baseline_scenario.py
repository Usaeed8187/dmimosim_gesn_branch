"""
Simulation of dMIMO baseline scenario with ns-3 channels

Note: this scripts should be called from the project root folder
"""

# add system folder for the dmimo library
import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from dmimo import sim_baseline

# Main function
if __name__ == "__main__":

    ns3_folder = "./ns3/channels/"
    modulations = ["QPSK", "16QAM", "64QAM"]
    modulation_orders = [2, 4, 6]
    num_modulations = len(modulation_orders)
    csi_delay = 2  # feedback delay in number of subframe

    ber = np.zeros((2, num_modulations))
    ldpc_ber = np.zeros((2, num_modulations))
    goodput = np.zeros((2, num_modulations))
    for k in range(num_modulations):
        rst_svd, xh_svd = sim_baseline(precoding_method="SVD", csi_delay=csi_delay, num_bits_per_symbol=modulation_orders[k], ns3_folder=ns3_folder)
        ber[0, k] = rst_svd[0]
        ldpc_ber[0, k] = rst_svd[1]
        goodput[0, k] = rst_svd[2]
        rst_zf, xh_zf = sim_baseline(precoding_method="ZF", csi_delay=csi_delay, num_bits_per_symbol=modulation_orders[k], ns3_folder=ns3_folder)
        ber[1, k] = rst_zf[0]
        ldpc_ber[1, k] = rst_zf[1]
        goodput[1, k] = rst_zf[2]

        print("Results for SVD with " + modulations[k])
        print("  Uncoded BER: ", ber[0, k])
        print("  LDPC BER: ", ldpc_ber[0, k])
        print("  Goodput: ", goodput[0, k])
        print("Results for ZF with " + modulations[k])
        print("  Uncoded BER: ", ber[1, k])
        print("  LDPC BER: ", ldpc_ber[1, k])
        print("  Goodput: ", goodput[1, k])
