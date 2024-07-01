"""
Simulation of dMIMO SU-MIMO scenario with ns-3 channels

Note: this scripts should be called from the project root folder
"""

# add system folder for the dmimo library
import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from dmimo import sim_su_mimo

# Main function
if __name__ == "__main__":
    ns3_folder = "./ns3/channels/"

    ber_QPSK = np.zeros((2, 4))
    ldpc_ber_QPSK = np.zeros((2, 4))
    goodput_QPSK = np.zeros((2, 4))
    for csi_delay in range(4):
        rst_svd, xh_svd = sim_su_mimo(precoding_method="SVD", num_bits_per_symbol=2, first_slot_idx=10, csi_delay=csi_delay, ns3_folder=ns3_folder)
        ber_QPSK[0, csi_delay] = rst_svd[0]
        ldpc_ber_QPSK[0, csi_delay] = rst_svd[1]
        goodput_QPSK[0, csi_delay] = rst_svd[2]
        rst_zf, xh_zf = sim_su_mimo(precoding_method="ZF", num_bits_per_symbol=2, first_slot_idx=10, csi_delay=csi_delay, ns3_folder=ns3_folder)
        ber_QPSK[1, csi_delay] = rst_zf[0]
        ldpc_ber_QPSK[1, csi_delay] = rst_zf[1]
        goodput_QPSK[1, csi_delay] = rst_zf[2]

        print("Results for SVD with QPSK (csi_delay={})".format(csi_delay))
        print("  Uncoded BER: ", ber_QPSK[0, csi_delay])
        print("  LDPC BER: ", ldpc_ber_QPSK[0, csi_delay])
        print("  Goodput: ", goodput_QPSK[0, csi_delay])
        print("Results for ZF with QPSK (csi_delay={})".format(csi_delay))
        print("  Uncoded BER: ", ber_QPSK[1, csi_delay])
        print("  LDPC BER: ", ldpc_ber_QPSK[1, csi_delay])
        print("  Goodput: ", goodput_QPSK[1, csi_delay])

    ber_16QAM = np.zeros((2, 4))
    ldpc_ber_16QAM = np.zeros((2, 4))
    goodput_16QAM = np.zeros((2, 4))
    for csi_delay in range(4):
        rst_svd, xh_svd = sim_su_mimo(precoding_method="SVD", num_bits_per_symbol=4, first_slot_idx=5, csi_delay=csi_delay, ns3_folder=ns3_folder)
        ber_16QAM[0, csi_delay] = rst_svd[0]
        ldpc_ber_16QAM[0, csi_delay] = rst_svd[1]
        goodput_16QAM[0, csi_delay] = rst_svd[2]
        rst_zf, xh_zf = sim_su_mimo(precoding_method="ZF", num_bits_per_symbol=4, first_slot_idx=5, csi_delay=csi_delay, ns3_folder=ns3_folder)
        ber_16QAM[1, csi_delay] = rst_zf[0]
        ldpc_ber_16QAM[1, csi_delay] = rst_zf[1]
        goodput_16QAM[1, csi_delay] = rst_zf[2]

        print("Results for SVD with 16QAM (csi_delay={})".format(csi_delay))
        print("  Uncoded BER: ", ber_16QAM[0, csi_delay])
        print("  LDPC BER: ", ldpc_ber_16QAM[0, csi_delay])
        print("  Goodput: ", goodput_16QAM[0, csi_delay])
        print("Results for ZF with 16QAM (csi_delay={})".format(csi_delay))
        print("  Uncoded BER: ", ber_16QAM[1, csi_delay])
        print("  LDPC BER: ", ldpc_ber_16QAM[1, csi_delay])
        print("  Goodput: ", goodput_16QAM[1, csi_delay])

    ber_64QAM = np.zeros((2, 4))
    ldpc_ber_64QAM = np.zeros((2, 4))
    goodput_64QAM = np.zeros((2, 4))
    for csi_delay in range(4):
        rst_svd, xh_svd = sim_su_mimo(precoding_method="SVD", num_bits_per_symbol=6, first_slot_idx=5, csi_delay=csi_delay, ns3_folder=ns3_folder)
        ber_64QAM[0, csi_delay] = rst_svd[0]
        ldpc_ber_64QAM[0, csi_delay] = rst_svd[1]
        goodput_64QAM[0, csi_delay] = rst_svd[2]
        rst_zf, xh_zf = sim_su_mimo(precoding_method="ZF", num_bits_per_symbol=6, first_slot_idx=5, csi_delay=csi_delay, ns3_folder=ns3_folder)
        ber_64QAM[1, csi_delay] = rst_zf[0]
        ldpc_ber_64QAM[1, csi_delay] = rst_zf[1]
        goodput_64QAM[1, csi_delay] = rst_zf[2]

        print("Results for SVD with 64QAM (csi_delay={})".format(csi_delay))
        print("  Uncoded BER: ", ber_64QAM[0, csi_delay])
        print("  LDPC BER: ", ldpc_ber_64QAM[0, csi_delay])
        print("  Goodput: ", goodput_64QAM[0, csi_delay])
        print("Results for ZF with 64QAM (csi_delay={})".format(csi_delay))
        print("  Uncoded BER: ", ber_64QAM[1, csi_delay])
        print("  LDPC BER: ", ldpc_ber_64QAM[1, csi_delay])
        print("  Goodput: ", goodput_64QAM[1, csi_delay])
