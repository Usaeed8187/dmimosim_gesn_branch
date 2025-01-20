"""
Simulation of MU-MIMO scenario with ns-3 channels

This scripts should be called from the "tests" folder
"""

# add system folder for the dmimo library
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dmimo.config import SimConfig
from dmimo.mu_mimo_gesn_test import sim_mu_mimo_all


sys.path.append(os.path.join('..'))
source_dir = '/home/data/ns3_channels_q4/'
destination_dir = 'ns3/'
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)
for root, dirs, files in os.walk(source_dir):
    # Construct the relative path to replicate the directory structure
    relative_path = os.path.relpath(root, source_dir)
    destination_subdir = os.path.join(destination_dir, relative_path)

    # Create the subdirectory in the destination if it doesn't exist
    if not os.path.exists(destination_subdir):
        os.makedirs(destination_subdir)
    
    # Create symlinks for each file in the current directory
    for file in files:
        source_file = os.path.join(root, file)
        destination_file = os.path.join(destination_subdir, file)

        # If the symlink already exists, remove it
        if os.path.exists(destination_file):
            os.remove(destination_file)

        # Create the symlink
        os.symlink(source_file, destination_file)
        # print(f"Symlink created for {source_file} -> {destination_file}")


script_name = sys.argv[0]
arguments = sys.argv[1:]

print(f"Script Name: {script_name}")
print(f"Arguments: {arguments}")

if len(arguments) > 0:
    mobility = arguments[0]
    drop_idx = arguments[1]
    rx_ues_arr = arguments[2:]
    rx_ues_arr = np.array(rx_ues_arr, dtype=int)
    
    print("Current mobility: {} \n Current drop: {} \n".format(mobility, drop_idx))
    print("rx_ues_arr: ", rx_ues_arr)
    print("rx_ues_arr[0]: ", rx_ues_arr[0])

# Main function
if __name__ == "__main__":

    # Simulation settings
    cfg = SimConfig()
    cfg.total_slots = 46        # total number of slots in ns-3 channels
    cfg.start_slot_idx = 30     # starting slots (must be greater than csi_delay + 5)
    cfg.csi_delay = 4           # feedback delay in number of subframe
    cfg.cfo_sigma = 0.0         # in Hz
    cfg.sto_sigma = 0.0         # in nanosecond
    cfg.num_tx_ue_sel = 1
    if arguments == []:
        mobility = 'high_mobility'
        drop_idx = '1'
        # rx_ues_arr = [1,2,4,6]
        rx_ues_arr = [2]
    cfg.ns3_folder = "ns3/channels_" + mobility + '_' + drop_idx + '/'

    folder_name = os.path.basename(os.path.abspath(cfg.ns3_folder))
    os.makedirs(os.path.join("results", folder_name), exist_ok=True)
    print("Using channels in {}".format(folder_name))

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

    #############################################
    # Testing with rank and link adaptation
    #############################################

    cfg.rank_adapt = True
    cfg.link_adapt = True
    cfg.csi_prediction = True
    cfg.predictor = 'gesn'

    for ue_arr_idx in range(np.size(rx_ues_arr)):

        cfg.num_rx_ue_sel = rx_ues_arr[ue_arr_idx]
        cfg.ue_indices = np.reshape(np.arange((cfg.num_rx_ue_sel + 2) * 2), (cfg.num_rx_ue_sel + 2, -1))

        cfg.precoding_method = "ZF"
        pred_nmse_gesn_model_based, pred_nmse_gesn_grad_descent, pred_nmse_vanilla = sim_mu_mimo_all(cfg)

        folder_path = "results/channels_multiple_mu_mimo/results/{}".format(folder_name)
        os.makedirs(folder_path, exist_ok=True)
        np.savez("{}/mu_mimo_results_UE_{}_pred.npz".format(folder_path, rx_ues_arr[ue_arr_idx]),
                pred_nmse_gesn_model_based=pred_nmse_gesn_model_based, pred_nmse_gesn_grad_descent=pred_nmse_gesn_grad_descent,
                pred_nmse_vanilla=pred_nmse_vanilla)