import numpy as np
import tensorflow as tf
from utilities_weiner_filter_pred import *
from dmimo.config import Ns3Config, SimConfig
from dmimo.channel import dMIMOChannels

################################################################################
# Simulation Settings
################################################################################

start_slot_idx = 95
csi_delay = 4
history_len = 16
batch_size = 1
num_ue_ant = 2
num_bs_ant = 4
syms_per_subframe = 14
num_subcarriers = 512
mobility = 'high_mobility'
drop_idx = 1
add_correlation = False
rho_target = 0.9

num_rx_nodes = 4
num_tx_nodes = 7

N_r = (num_rx_nodes-1)*num_ue_ant+num_bs_ant
N_t = (num_tx_nodes-1)*num_ue_ant+num_bs_ant

################################################################################
# Loading Channel Estimates
################################################################################

first_csi_history_idx = start_slot_idx - (csi_delay * history_len)
channel_history_slots = tf.range(first_csi_history_idx, start_slot_idx, csi_delay)

h_freq_csi_history = tf.zeros((tf.size(channel_history_slots), batch_size, 1, (num_rx_nodes-1)*num_ue_ant+num_bs_ant,
                                1, (num_tx_nodes-1)*num_ue_ant+num_bs_ant, syms_per_subframe, num_subcarriers), dtype=tf.complex64)

for loop_idx, slot_idx in enumerate(channel_history_slots):
    # h_freq_csi has shape [batch_size, num_rx, num_rx_ant, num_tx, num_txs_ant, num_ofdm_sym, fft_size]
    folder_path = "ns3/channel_estimates_{}_{}_rx_{}_tx_{}".format(mobility, drop_idx, N_r, N_t)
    file_path = "{}/dmimochans_{}".format(folder_path, slot_idx)
    try:
        data = np.load("{}.npz".format(file_path))
        h_freq_csi = data['h_freq_csi']
    except:
        # h_freq_csi, _ = lmmse_channel_estimation(dmimo_chans, rg_csi, slot_idx=slot_idx)
        # os.makedirs(folder_path, exist_ok=True)
        # np.savez(file_path, h_freq_csi=h_freq_csi)
        raise Exception("Channel estimates not found")
    indices = tf.constant([[loop_idx]])
    updates = tf.expand_dims(h_freq_csi, axis=0)
    h_freq_csi_history = tf.tensor_scatter_nd_update(h_freq_csi_history, indices, updates)

h_freq_csi_history = np.asarray(h_freq_csi_history)
h_freq_csi_history_list = []

for tx_node_idx in range(num_tx_nodes):
    for rx_node_idx in range(num_rx_nodes):
        if tx_node_idx == 0:
            tx_ant_idx = np.arange(0,num_bs_ant)
        else:
            tx_ant_idx = np.arange(num_bs_ant + (tx_node_idx-1)*num_ue_ant,num_bs_ant + (tx_node_idx)*num_ue_ant)
        TxAnt = len(tx_ant_idx)

        if rx_node_idx == 0:
            rx_ant_idx = np.arange(0,num_bs_ant)
        else:
            rx_ant_idx = np.arange(num_bs_ant + (rx_node_idx-1)*num_ue_ant,num_bs_ant + (rx_node_idx)*num_ue_ant)
        RxAnt = len(rx_ant_idx)

        curr_h_freq_csi_history = h_freq_csi_history[:,:,:,rx_ant_idx,:,...]
        curr_h_freq_csi_history = curr_h_freq_csi_history[:,:,:,:,:,tx_ant_idx,...]

        h_freq_csi_history_list.append(np.squeeze(curr_h_freq_csi_history))

hold = 1

################################################################################
# Adding correlation
################################################################################


cands = evaluate_pairwise_wiener_candidates(h_freq_csi_history_list, lam=0.0, max_pairs=10)
for r in cands:
    i, j, t = r["i"], r["j"], r["target_link"]
    print(f"(i,j)=({i},{j}) → predict {t} | test NMSE={r['nmse_test']:.4e} | train NMSE={r['nmse_train']:.4e} | shape={r['shape']}")

candidate_to_use = 3

h_freq_csi_history_list_truncated = [h_freq_csi_history_list[cands[candidate_to_use]["i"]], h_freq_csi_history_list[cands[candidate_to_use]["j"]]]

# Measure empirical complex correlation before
corr_before = complex_corr_coeff(h_freq_csi_history_list_truncated[0], h_freq_csi_history_list_truncated[1])
print(f"\nEmpirical complex corr before: {np.abs(corr_before):.4f}")

if add_correlation:    

    h_freq_csi_history_list_truncated[1] = impose_crosslink_correlation(h_freq_csi_history_list_truncated[0], 
                                                                        h_freq_csi_history_list_truncated[1], 
                                                                        rho_target=rho_target, 
                                                                        rng=np.random.default_rng(2025))

    # Measure empirical complex correlation after
    corr_after  = complex_corr_coeff(h_freq_csi_history_list_truncated[0], h_freq_csi_history_list_truncated[1])

    print(f"Empirical complex corr after : {np.abs(corr_after):.4f} (target ~ {rho_target})")

h_freq_csi_history_list_corr = h_freq_csi_history_list_truncated

hold = 1

################################################################################
# Weiner Filter Based Prediction
################################################################################

link_to_predict = 0 if cands[candidate_to_use]["i"] == cands[candidate_to_use]["target_link"] else 1

y1_hat_last, w1, info1 = wiener_predict_TEST_last_subframe_single_link(
    h_freq_csi_history_list_corr, lam=1e-3, link_index=link_to_predict
)
pretty_print_wiener_info(info1, title="Prediction using ONE link")

y_hat_last, w_filter, info = wiener_predict_TEST_last_subframe_from_both_links(
    h_freq_csi_history_list_corr, lam=1e-3, target_link=link_to_predict
)
pretty_print_wiener_info(info, title="Prediction using BOTH links")

hold = 1