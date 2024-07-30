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
from dmimo.su_mimo_chanpred import sim_su_mimo_chanpred


# Main function
if __name__ == "__main__":

    # Simulation settings
    cfg = SimConfig()
    cfg.total_slots = 90        # total number of slots in ns-3 channels
    cfg.start_slot_idx = 70     # starting slots (must be greater than csi_delay + 5)
    cfg.csi_delay = 8           # feedback delay in number of subframe
    cfg.num_tx_streams = 6      # 4/6 equal to total number of streams
    cfg.cfo_sigma = 0.0         # in Hz
    cfg.sto_sigma = 0.0         # in nanosecond
    cfg.ns3_folder = "../ns3/channels_s3/"

    avg_ber = 0.0
    avg_ber_pred = 0.0
    avg_tput = 0.0
    avg_tput_pred = 0.0
    total_runs = 0
    for first_slot_idx in np.arange(cfg.start_slot_idx, cfg.total_slots, cfg.num_slots_p1+cfg.num_slots_p2):
        total_runs += 1
        print("------ Run {} -----".format(total_runs))
        cfg.first_slot_idx = first_slot_idx
        cfg.csi_prediction = False
        cfg.precoding_method = "ZF"
        bers, bits, x_hat = sim_su_mimo_chanpred(cfg)
        avg_ber += bers[0]
        avg_tput += bits[0]
        print("Channel prediction: ", cfg.csi_prediction)
        print("BER: ", bers)
        print("Goodbits: ", bits)

        cfg.csi_prediction = True
        cfg.precoding_method = "ZF"
        bers, bits, x_hat = sim_su_mimo_chanpred(cfg)
        avg_ber_pred += bers[0]
        avg_tput_pred += bits[0]
        print("Channel prediction: ", cfg.csi_prediction)
        print("BER: ", bers)
        print("Goodbits: ", bits)

    avg_ber /= total_runs
    avg_ber_pred /= total_runs
    avg_tput /= total_runs
    avg_tput_pred /= total_runs

    overhead = cfg.num_slots_p2 / (cfg.num_slots_p1 + cfg.num_slots_p2)
    avg_tput *= overhead/(cfg.slot_duration * 1e6)
    avg_tput_pred *= overhead/(cfg.slot_duration * 1e6)

    print("")
    print("Average BER: {:3f}".format(avg_ber), "Average Throughput: {:.2f}".format(avg_tput))
    print("Average BER with prediction: {:3f}".format(avg_ber_pred), "Average Throughput: {:.2f}".format(avg_tput_pred))


