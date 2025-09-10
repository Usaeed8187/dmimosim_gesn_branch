"""
Simulation of MU-MIMO scenario with ns-3 channels

This scripts should be called from the "tests" folder
"""

# add system folder for the dmimo library
import sys
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")          # silence Python-side TF logs

from dmimo.config import SimConfig, RCConfig
from dmimo.dmimo_chanpred import sim_mu_mimo_all

os.environ['PYTHONHASHSEED'] = '10'
tf.random.set_seed(10)

sys.path.append(os.path.join('..'))
src = Path('/home/data/ns3_channels_q4/').expanduser()
dst = Path('ns3')

if not src.is_dir():
    raise FileNotFoundError(f"Source dir not found: {src}")

for f in src.rglob('*'):
    if not f.is_file():
        continue
    rel = f.relative_to(src)
    out = dst / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    # remove existing file or symlink at destination
    if out.exists() or out.is_symlink():
        out.unlink()
    # make symlink using absolute real path (avoids broken links)
    out.symlink_to(f.resolve())

script_name = sys.argv[0]
arguments = sys.argv[1:]

print(f"Script Name: {script_name}")
print(f"Arguments: {arguments}")

if len(arguments) > 0:
    mobility = arguments[0]
    drop_idx = arguments[1]
    vector_inputs = arguments[2]
    rx_ues_arr = arguments[3:]
    rx_ues_arr = np.array(rx_ues_arr, dtype=int)
    
    print("Current mobility: {} \n Current drop: {} \n Current Vector Inputs: {} \n".format(mobility, drop_idx, vector_inputs))
    print("rx_ues_arr: ", rx_ues_arr)
    print("rx_ues_arr[0]: ", rx_ues_arr[0])

# Main function
if __name__ == "__main__":

    # Simulation settings
    rc_config = RCConfig()
    cfg = SimConfig()
    cfg.total_slots = 90        # total number of slots in ns-3 channels
    cfg.start_slot_idx = 80     # starting slots (must be greater than csi_delay + 5)
    cfg.csi_delay = 4           # feedback delay in number of subframe
    cfg.cfo_sigma = 0.0         # in Hz
    cfg.sto_sigma = 0.0         # in nanosecond
    cfg.num_tx_ue_sel = 6
    if arguments == []:
        mobility = 'high_mobility'
        drop_idx = '2'
        rx_ues_arr = [3]
        vector_inputs = 'tx_ants' # tx_ants, rx_ants, none, all
        csi_delays = [2, 3, 4, 5, 6, 7, 8]
    cfg.ns3_folder = "ns3/channels_" + mobility + '_' + drop_idx + '/'
    rc_config.lr = 0.01
    rc_config.num_epochs = 50
    rc_config.enable_window = True
    rc_config.window_length = 1
    rc_config.num_neurons = 16
    rc_config.vector_inputs = vector_inputs
    rc_config.weight_initialization = 'model_based_freq_corr' # "model_based_aoa_aod", "model_based_freq_corr", "model_based_delays" #TODO: find a better edge update method than grad descent. maybe attention based mechanism
    rc_config.mobility = mobility
    rc_config.drop_idx = drop_idx
    rc_config.history_len = 8
    cfg.graph_formulation = 'per_antenna_pair' # "per_node_pair", "per_antenna_pair", "supergraph"
    cfg.num_tx_streams = 1
    cfg.modulation_order = 2

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

        for csi_delay_idx in range(np.size(csi_delays)):

            cfg.num_rx_ue_sel = rx_ues_arr[ue_arr_idx]
            cfg.csi_delay = csi_delays[csi_delay_idx]

            cfg.ue_indices = np.reshape(np.arange((cfg.num_rx_ue_sel + 2) * 2), (cfg.num_rx_ue_sel + 2, -1))

            cfg.precoding_method = "ZF"
            (pred_nmse_pred_nmse_outdated,
            pred_nmse_wesn,
            pred_nmse_wgesn_per_antenna_pair,
            pred_nmse_kalman,
            uncoded_ber_outdated,
            uncoded_ber_wesn,
            uncoded_ber_wgesn,
            uncoded_ber_kalman) = sim_mu_mimo_all(cfg, rc_config)

            folder_path = "results/dmimo_chanpred/channels_multiple_mu_mimo_vector_inputs_{}/results_{}_epochs_{}_lr_{}_window_length_{}_weight_initialization_{}/{}".format(rc_config.vector_inputs, cfg.graph_formulation, 
                                                                        rc_config.num_epochs, rc_config.lr, rc_config.window_length, 
                                                                        rc_config.weight_initialization, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            np.savez(
                "{}/mu_mimo_results_num_UEs_{}_pred.npz".format(folder_path, rx_ues_arr[ue_arr_idx]),
                pred_nmse_pred_nmse_outdated=pred_nmse_pred_nmse_outdated,
                pred_nmse_wesn=pred_nmse_wesn,
                pred_nmse_wgesn_per_antenna_pair=pred_nmse_wgesn_per_antenna_pair,
                pred_nmse_kalman=pred_nmse_kalman,
                uncoded_ber_outdated=uncoded_ber_outdated,
                uncoded_ber_wesn=uncoded_ber_wesn,
                uncoded_ber_wgesn=uncoded_ber_wgesn,
                uncoded_ber_kalman=uncoded_ber_kalman,
            )
