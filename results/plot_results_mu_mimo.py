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
nodewise_goodput = []
nodewise_throughput = []
nodewise_bitrate = []
ranks = []
uncoded_ber_list = []
ldpc_ber_list = []
sinr_dB = []

for mobility_idx in range(np.size(mobilities)):

    curr_mobility = mobilities[mobility_idx]

    temp_nodewise_goodput = []
    temp_nodewise_throughput = []
    temp_nodewise_bitrate = []
    temp_ranks = []
    temp_uncoded_ber_list = []
    temp_ldpc_ber_list = []
    temp_sinr_dB = []

    for ue_arr_idx in range(np.size(rx_ues_arr)):

        file_path = "results/channels_{}/mu_mimo_results_UE_{}.npz".format(curr_mobility, rx_ues_arr[ue_arr_idx])
        data = np.load(file_path)

        temp_nodewise_goodput.append(data['nodewise_goodput'])
        temp_nodewise_throughput.append(data['nodewise_throughput'])
        temp_nodewise_bitrate.append(data['nodewise_bitrate'])
        temp_ranks.append(data['ranks'])
        temp_uncoded_ber_list.append(data['uncoded_ber_list'])
        temp_ldpc_ber_list.append(data['ldpc_ber_list'])
        temp_sinr_dB.append(np.concatenate(data['sinr_dB']))
    
    ber.append(data['ber'])
    ldpc_ber.append(data['ldpc_ber'])
    goodput.append(data['goodput'])
    throughput.append(data['throughput'])
    bitrate.append(data['bitrate'])
    nodewise_goodput.append(temp_nodewise_goodput)
    nodewise_throughput.append(temp_nodewise_throughput)
    nodewise_bitrate.append(temp_nodewise_bitrate)
    ranks.append(temp_ranks)
    uncoded_ber_list.append(temp_uncoded_ber_list)
    ldpc_ber_list.append(temp_ldpc_ber_list)
    sinr_dB.append(temp_sinr_dB)

#############################################
# Plots
#############################################



############################### SINR Distributions (rx nodes in phase 2) ######################################
plt.figure()
colors = ['red', 'green', 'purple']
for mobility_idx in range(np.size(mobilities)):
    plt.boxplot(sinr_dB[mobility_idx], positions=positions_all_scenarios[mobility_idx], widths=0.6, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor=colors[mobility_idx], color=colors[mobility_idx]),
                medianprops=dict(color='black'))
plt.xticks(np.concatenate(positions_all_scenarios), rx_ues_arr * 3)
legend_elements = [plt.Line2D([0], [0], color='red', lw=4, label='Scenario 1'),
                   plt.Line2D([0], [0], color='green', lw=4, label='Scenario 2'),
                   plt.Line2D([0], [0], color='purple', lw=4, label='Scenario 3')]
plt.legend(handles=legend_elements, title='Scenarios', loc='upper right')
plt.grid(True)
plt.xlabel('Number of UEs')
plt.ylabel('SINR')
plt.title('SINR (dB)')
plt.savefig("results/plots/SINR")

############################### End-to-end average BER ######################################
plt.figure()
plt.semilogy(rx_ues_arr, ber[0], marker='o', label='Low Mobility')
plt.semilogy(rx_ues_arr, ber[1], marker='s', label='Medium Mobility')
plt.semilogy(rx_ues_arr, ber[2], marker='^', label='High Mobility')
plt.grid(True)
plt.xlabel('Number of UEs')
plt.ylabel('BER')
plt.title('Uncoded BER')
plt.legend()
plt.savefig("results/plots/BER")

############################### End-to-end average BLER ######################################
plt.figure()
plt.semilogy(rx_ues_arr, ldpc_ber[0], marker='o', label='Low Mobility')
plt.semilogy(rx_ues_arr, ldpc_ber[1], marker='s', label='Medium Mobility')
plt.semilogy(rx_ues_arr, ldpc_ber[2], marker='^', label='High Mobility')
plt.grid(True)
plt.xlabel('Number of UEs')
plt.ylabel('BLER')
plt.title('BLER')
plt.legend()
plt.savefig("results/plots/BLER")


############################### Probability of Outage (rx nodes in phase 2) - graph version 1 ######################################
threshold = 0.15
outage_probability = []
for mobility_idx in range(np.size(mobilities)):
    temp_outage_probability = []
    for ue_idx in range(np.size(rx_ues_arr)):
        prob = np.sum(uncoded_ber_list[mobility_idx][ue_idx] > threshold) / np.size(uncoded_ber_list[0][0]) * 100
        temp_outage_probability.append(prob)
    outage_probability.append(temp_outage_probability)
plt.figure()
plt.plot(rx_ues_arr, outage_probability[0], marker='o', label='Low Mobility')
plt.plot(rx_ues_arr, outage_probability[1], marker='s', label='Medium Mobility')
plt.plot(rx_ues_arr, outage_probability[2], marker='^', label='High Mobility')
plt.grid(True)
plt.xlabel('Number of UEs')
plt.ylabel('Probability (%)')
plt.title('Probability of Outage')
plt.legend()
plt.savefig("results/plots/prob_outage")



hold = 1