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

# Prediction off

prediction_results = False
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

    temp_nodewise_goodput = []
    temp_nodewise_throughput = []
    temp_nodewise_bitrate = []
    temp_ranks = []
    temp_uncoded_ber_list = []
    temp_ldpc_ber_list = []
    temp_sinr_dB = []

    for ue_arr_idx in range(np.size(rx_ues_arr)):
        
        if prediction_results:
            file_path = "results/channels_{}/mu_mimo_results_UE_{}_pred.npz".format(curr_mobility, rx_ues_arr[ue_arr_idx])
        else:
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

    if prediction_results:
        baseline_file_path = "results/channels_{}/baseline_results_pred.npz".format(curr_mobility)
    else:
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

# Prediction on

prediction_results = True
ber_pred = []
ldpc_ber_pred = []
goodput_pred = []
throughput_pred = []
bitrate_pred = []
nodewise_goodput_pred = []
nodewise_throughput_pred = []
nodewise_bitrate_pred = []
ranks_pred = []
uncoded_ber_list_pred = []
ldpc_ber_list_pred = []
sinr_dB_pred = []

baseline_ber_pred = []
baseline_ldpc_ber_pred = []
baseline_goodput_pred = []
baseline_throughput_pred = []
baseline_bitrate_pred = []
baseline_ranks_pred = []
baseline_uncoded_ber_list_pred = []
baseline_ldpc_ber_list_pred = []

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
        
        if prediction_results:
            file_path = "results/channels_{}/mu_mimo_results_UE_{}_pred.npz".format(curr_mobility, rx_ues_arr[ue_arr_idx])
        else:
            file_path = "results/channels_{}/mu_mimo_results_UE_{}.npz".format(curr_mobility, rx_ues_arr[ue_arr_idx])
        data = np.load(file_path)

        temp_nodewise_goodput.append(data['nodewise_goodput'])
        temp_nodewise_throughput.append(data['nodewise_throughput'])
        temp_nodewise_bitrate.append(data['nodewise_bitrate'])
        temp_ranks.append(data['ranks'])
        temp_uncoded_ber_list.append(data['uncoded_ber_list'])
        temp_ldpc_ber_list.append(data['ldpc_ber_list'])
        temp_sinr_dB.append(np.concatenate(data['sinr_dB']))
    
    ber_pred.append(data['ber'])
    ldpc_ber_pred.append(data['ldpc_ber'])
    goodput_pred.append(data['goodput'])
    throughput_pred.append(data['throughput'])
    bitrate_pred.append(data['bitrate'])
    nodewise_goodput_pred.append(temp_nodewise_goodput)
    nodewise_throughput_pred.append(temp_nodewise_throughput)
    nodewise_bitrate_pred.append(temp_nodewise_bitrate)
    ranks_pred.append(temp_ranks)
    uncoded_ber_list_pred.append(temp_uncoded_ber_list)
    ldpc_ber_list_pred.append(temp_ldpc_ber_list)
    sinr_dB_pred.append(temp_sinr_dB)

    if prediction_results:
        baseline_file_path = "results/channels_{}/baseline_results_pred.npz".format(curr_mobility)
    else:
        baseline_file_path = "results/channels_{}/baseline_results.npz".format(curr_mobility)
    baseline_data = np.load(baseline_file_path)

    baseline_ranks_pred.append(baseline_data['ranks'])
    baseline_uncoded_ber_list_pred.append(baseline_data['uncoded_ber_list'])
    baseline_ldpc_ber_list_pred.append(baseline_data['ldpc_ber_list'])
    baseline_ber_pred.append(baseline_data['ber'])
    baseline_ldpc_ber_pred.append(baseline_data['ldpc_ber'])
    baseline_goodput_pred.append(baseline_data['goodput'])
    baseline_throughput_pred.append(baseline_data['throughput'])
    baseline_bitrate_pred.append(baseline_data['bitrate'])

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
plt.ylim(-10, 15)
plt.savefig("results/plots/SINR_MU_MIMO")

############################### End-to-end average BER ######################################
# # Method 1 for end-to-end average BER (mobility on x-axis)
# plt.figure()
# x_labels = ['Low mobility', 'Medium mobility', 'High mobility']
# x_values = np.arange(1,np.size(mobilities)+1)
# plt.semilogy(x_values, np.asarray(ber)[:,0], marker='o', label='1 UE')
# plt.semilogy(x_values, np.asarray(ber)[:,1], marker='s', label='2 UEs')
# plt.semilogy(x_values, np.asarray(ber)[:,2], marker='v', label='4 UEs')
# plt.semilogy(x_values, np.asarray(ber)[:,3], marker='^', label='6 UEs')
# plt.semilogy(x_values, np.asarray(baseline_ber), marker='*', label='Baseline')
# plt.grid(True)
# plt.xticks(x_values, x_labels)
# plt.ylabel('BER')
# plt.title('Uncoded BER')
# plt.legend()
# plt.savefig("results/plots/BER_MU_MIMO")

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

# # Method 3 for end-to-end average BER (bar plot with mobility on x-axis)
# num_categories = len(mobilities)
# x = np.arange(num_categories)
# bar_width = 0.15
# x_labels = ['Scenario 1', 'Scenario 2', 'Scenario 3']
# plt.figure()
# plt.bar(x - 2 * bar_width, np.asarray(baseline_ber), width=bar_width, label='Baseline', color='#4F81BD')
# plt.bar(x - bar_width, np.asarray(ber)[:,0], width=bar_width, label='1 UE', color='#C0504D')
# plt.bar(x, np.asarray(ber)[:,1], width=bar_width, label='2 UEs', color='#9BBB59')
# plt.bar(x + bar_width, np.asarray(ber)[:,2], width=bar_width, label='4 UEs', color='#8064A2')
# plt.bar(x + 2 * bar_width, np.asarray(ber)[:,3], width=bar_width, label='6 UEs', color='#4BACC6')
# plt.yscale('log')
# plt.xticks(x, x_labels)
# plt.grid(True)
# plt.ylabel('BER')
# plt.title('Uncoded BER')
# plt.legend()
# plt.savefig("results/plots/BER_MU_MIMO")


# Method 3 for end-to-end average BER (bar plot with mobility on x-axis), but plotting only the best selection of UEs. also plotting the prediction cases
num_categories = len(mobilities)
x = np.arange(num_categories)
bar_width = 0.25
x_labels = ['Scenario 1', 'Scenario 2', 'Scenario 3']
plt.figure()
plt.bar(x - bar_width, np.asarray(baseline_ber), width=bar_width, label='Baseline', color='#4F81BD')
plt.bar(x, np.asarray(ber)[:,2], width=bar_width, label='MU MIMO', color='#C0504D')
plt.bar(x + bar_width, np.asarray(ber_pred)[:,2], width=bar_width, label='MU MIMO with Channel Prediction', color='#9BBB59')
plt.yscale('log')
plt.xticks(x, x_labels)
plt.grid(True)
plt.ylabel('BER')
plt.title('Uncoded BER')
plt.legend()
plt.savefig("results/plots/BER_MU_MIMO")


############################### End-to-end average BLER ######################################
# # Method 1 for end-to-end average BER (mobility on x-axis)
# plt.figure()
# x_labels = ['Low mobility', 'Medium mobility', 'High mobility']
# x_values = np.arange(1,np.size(mobilities)+1)
# plt.semilogy(x_values, np.asarray(ldpc_ber)[:,0], marker='o', label='1 UE')
# plt.semilogy(x_values, np.asarray(ldpc_ber)[:,1], marker='s', label='2 UEs')
# plt.semilogy(x_values, np.asarray(ldpc_ber)[:,2], marker='v', label='4 UEs')
# plt.semilogy(x_values, np.asarray(ldpc_ber)[:,3], marker='^', label='6 UEs')
# plt.semilogy(x_values, np.asarray(baseline_ldpc_ber), marker='*', label='Baseline')
# plt.grid(True)
# plt.xticks(x_values, x_labels)
# plt.ylabel('BLER')
# plt.title('BLER')
# plt.legend()
# plt.savefig("results/plots/BLER_MU_MIMO")


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

# # Method 3 for end-to-end average BLER (bar plot with mobility on x-axis)
# num_categories = len(mobilities)
# x = np.arange(num_categories)
# bar_width = 0.15
# x_labels = ['Scenario 1', 'Scenario 2', 'Scenario 3']
# plt.figure()
# plt.bar(x - 2 * bar_width, np.asarray(baseline_ldpc_ber), width=bar_width, label='Baseline', color='#4F81BD')
# plt.bar(x - bar_width, np.asarray(ldpc_ber)[:,0], width=bar_width, label='1 UE', color='#C0504D')
# plt.bar(x, np.asarray(ldpc_ber)[:,1], width=bar_width, label='2 UEs', color='#9BBB59')
# plt.bar(x + bar_width, np.asarray(ldpc_ber)[:,2], width=bar_width, label='4 UEs', color='#8064A2')
# plt.bar(x + 2 * bar_width, np.asarray(ldpc_ber)[:,3], width=bar_width, label='6 UEs', color='#4BACC6')
# plt.yscale('log')
# plt.xticks(x, x_labels)
# plt.grid(True)
# plt.ylabel('BLER')
# plt.title('BLER')
# plt.legend()
# plt.savefig("results/plots/BLER_MU_MIMO")

# # Method 3 for end-to-end average BLER (bar plot with mobility on x-axis), but plotting only the best selection of UEs
# num_categories = len(mobilities)
# x = np.arange(num_categories)
# bar_width = 0.35
# x_labels = ['Scenario 1', 'Scenario 2', 'Scenario 3']
# plt.figure()
# plt.bar(x - bar_width/2, np.asarray(baseline_ldpc_ber), width=bar_width, label='Baseline', color='#4F81BD')
# plt.bar(x + bar_width/2, np.asarray(ldpc_ber)[:,2], width=bar_width, label='MU MIMO', color='#9BBB59')
# plt.yscale('log')
# plt.xticks(x, x_labels)
# plt.grid(True)
# plt.ylabel('BER')
# plt.title('Uncoded BER')
# plt.legend()
# plt.savefig("results/plots/BLER_MU_MIMO")

# Method 3 for end-to-end average BLER (bar plot with mobility on x-axis), but plotting only the best selection of UEs
num_categories = len(mobilities)
x = np.arange(num_categories)
bar_width = 0.35
x_labels = ['Scenario 1', 'Scenario 2', 'Scenario 3']
plt.figure()
plt.bar(x - bar_width, np.asarray(baseline_ldpc_ber_pred), width=bar_width, label='Baseline', color='#4F81BD')
plt.bar(x, np.asarray(ldpc_ber)[:,2], width=bar_width, label='MU MIMO', color='#C0504D')
plt.bar(x + bar_width, np.asarray(ldpc_ber_pred)[:,2], width=bar_width, label='MU MIMO with Channel Prediction', color='#9BBB59')
plt.yscale('log')
plt.xticks(x, x_labels)
plt.grid(True)
plt.ylabel('BER')
plt.title('Uncoded BER')
plt.legend()
plt.savefig("results/plots/BLER_MU_MIMO")


############################### Probability of Outage (rx nodes in phase 2) ######################################
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
plt.savefig("results/plots/prob_outage_MU_MIMO")


hold = 1