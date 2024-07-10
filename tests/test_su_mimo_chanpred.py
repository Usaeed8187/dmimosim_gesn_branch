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

from dmimo.config import SimConfig, RCConfig
from dmimo.su_mimo_chanpred import sim_su_mimo_chanpred


# Main function
if __name__ == "__main__":

    # Simulation settings
    cfg = SimConfig()
    cfg.total_slots = 50        # total number of slots in ns-3 channels
    cfg.start_slot_idx = 20     # starting slots (must be greater than csi_delay + 5)
    cfg.csi_delay = 6           # feedback delay in number of subframe
    cfg.num_tx_streams = 6      # 4/6 equal to total number of streams
    cfg.cfo_sigma = 0.0         # in Hz
    cfg.sto_sigma = 0.0         # in nanosecond
    cfg.ns3_folder = "../ns3/channels4pred/"

    cfg.csi_prediction = True
    cfg.first_slot_idx = 16 # 12 # cfg.start_slot_idx # RCConfig().history_len * cfg.csi_delay

    bers, bits, x_hat = sim_su_mimo_chanpred(cfg, precoding_method="ZF")

    print("Channel prediction: ", cfg.csi_prediction)
    print(bers)
    print(bits)


