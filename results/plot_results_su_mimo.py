import sys
import os
sys.path.append(os.path.join('..'))

import matplotlib.pyplot as plt
import numpy as np



#############################################
# Settings
#############################################

rx_ues_arr = [1,2,4,6]

# Define positions for the box plots
positions_all_scenarios = []
positions_scenario_1 = np.arange(1,np.size(rx_ues_arr)+1)
positions_all_scenarios.append(positions_scenario_1)
positions_all_scenarios.append(positions_scenario_1 + 5) # Shifted right for the second scenario
positions_all_scenarios.append(positions_scenario_1 + 10) # Shifted further right for the third scenario

mobilities = ['low_mobility', 'medium_mobility', 'high_mobility']



#############################################
# KPI Handling
#############################################

ber = []
ldpc_ber = []
goodput = []
throughput = []
bitrate = []
ranks = []
uncoded_ber_list = []
ldpc_ber_list = []

baseline_ber = []
baseline_ldpc_ber = []
baseline_goodput = []
baseline_throughput = []
baseline_bitrate = []
baseline_ranks = []
baseline_uncoded_ber_list = []
baseline_ldpc_ber_list = []

for mobility_idx in range(np.size(mobilities)):

    curr_mobility = mobilities[mobility_idx]

    file_path = "results/channels_{}/su_mimo_results.npz".format(curr_mobility)
    data = np.load(file_path)
    
    ber.append(data['ber'])
    ldpc_ber.append(data['ldpc_ber'])
    uncoded_ber_list.append(data['uncoded_ber_list'])
    ldpc_ber_list.append(data['ldpc_ber_list'])
    goodput.append(data['goodput'])
    throughput.append(data['throughput'])
    bitrate.append(data['bitrate'])
    ranks.append(ranks)

    baseline_file_path = "results/channels_{}/baseline_results.npz".format(curr_mobility)
    baseline_data = np.load(baseline_file_path)

    baseline_ranks.append(baseline_data['ranks'])
    baseline_uncoded_ber_list.append(baseline_data['uncoded_ber_list'])
    baseline_ldpc_ber_list.append(baseline_data['ldpc_ber_list'])
    baseline_ber.append(baseline_data['ber'])
    baseline_ldpc_ber.append(baseline_data['ldpc_ber'])
    baseline_goodput.append(baseline_data['goodput'])
    baseline_throughput.append(baseline_data['throughput'])
    baseline_bitrate.append(baseline_data['bitrate'])

#############################################
# Plots
#############################################

############################### End-to-end average BER ######################################
# Method 1 for end-to-end average BER (mobility on x-axis)
plt.figure()
x_labels = ['Low mobility', 'Medium mobility', 'High mobility']
x_values = np.arange(1,np.size(mobilities)+1)
plt.semilogy(x_values, np.asarray(ber), marker='o', label='SU MIMO')
plt.semilogy(x_values, np.asarray(baseline_ber), marker='*', label='Baseline')
plt.grid(True)
plt.xticks(x_values, x_labels)
plt.ylabel('BER')
plt.title('Uncoded BER')
plt.legend()
plt.savefig("results/plots/BER_SU_MIMO")

# # Method 2 for end-to-end average BER (number of UEs on x-axis)
# plt.figure()
# plt.semilogy(rx_ues_arr, ber[0], marker='o', label='Low Mobility')
# plt.semilogy(rx_ues_arr, ber[1], marker='s', label='Medium Mobility')
# plt.semilogy(rx_ues_arr, ber[2], marker='^', label='High Mobility')
# plt.grid(True)
# plt.xlabel('Number of UEs')
# plt.ylabel('BER')
# plt.title('Uncoded BER')
# plt.legend()
# plt.savefig("results/plots/BER")

############################### End-to-end average BLER ######################################
# Method 1 for end-to-end average BER (mobility on x-axis)
plt.figure()
x_labels = ['Low mobility', 'Medium mobility', 'High mobility']
x_values = np.arange(1,np.size(mobilities)+1)
plt.semilogy(x_values, np.asarray(ldpc_ber), marker='o', label='SU MIMO')
plt.semilogy(x_values, np.asarray(baseline_ldpc_ber), marker='*', label='Baseline')
plt.grid(True)
plt.xticks(x_values, x_labels)
plt.ylabel('BLER')
plt.title('BLER')
plt.legend()
plt.savefig("results/plots/BLER_SU_MIMO")


# # Method 2 for end-to-end average BLER (number of UEs on x-axis)
# plt.figure()
# plt.semilogy(rx_ues_arr, ldpc_ber[0], marker='o', label='Low Mobility')
# plt.semilogy(rx_ues_arr, ldpc_ber[1], marker='s', label='Medium Mobility')
# plt.semilogy(rx_ues_arr, ldpc_ber[2], marker='^', label='High Mobility')
# plt.grid(True)
# plt.xlabel('Number of UEs')
# plt.ylabel('BLER')
# plt.title('BLER')
# plt.legend()
# plt.savefig("results/plots/BLER")


hold = 1