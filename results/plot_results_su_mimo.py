import sys
import os
sys.path.append(os.path.join('..'))

import matplotlib.pyplot as plt
import numpy as np



#############################################
# Settings
#############################################

mobilities = ['low_mobility', 'medium_mobility', 'high_mobility']

prediction_results = False

rx_ues_arr = [1,2,4,6]

# Define positions for the box plots
positions_all_scenarios = []
positions_scenario_1 = np.arange(1,np.size(rx_ues_arr)+1)
positions_all_scenarios.append(positions_scenario_1)
positions_all_scenarios.append(positions_scenario_1 + 5) # Shifted right for the second scenario
positions_all_scenarios.append(positions_scenario_1 + 10) # Shifted further right for the third scenario

evaluate_multiple_UEs = True

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

    if evaluate_multiple_UEs:

        temp_ldpc_ber = []
        temp_throughput = []
        temp_goodput = []
        temp_bitrate = []

        for ue_arr_idx in range(np.size(rx_ues_arr)):

            if prediction_results:
                file_path = "results/channels_{}/su_mimo_results_UE_{}_pred.npz".format(curr_mobility, rx_ues_arr[ue_arr_idx])
            else:
                file_path = "results/channels_{}/su_mimo_results_UE_{}.npz".format(curr_mobility, rx_ues_arr[ue_arr_idx])
            data = np.load(file_path)

            temp_ldpc_ber.append(data['ldpc_ber'])
            temp_throughput.append(data['throughput'])
            temp_goodput.append(data['goodput'])
            temp_bitrate.append(data['bitrate'])
        
        throughput.append(temp_throughput)
        goodput.append(temp_goodput)
        bitrate.append(temp_bitrate)

        ldpc_ber.append(temp_ldpc_ber)
    
    else:

        if prediction_results:
            file_path = "results/channels_{}/su_mimo_results_pred.npz".format(curr_mobility)
        else:
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

#############################################
# Plots
#############################################

############################### End-to-end average BER ######################################
# Method 1 for end-to-end average BER (mobility on x-axis)
if not evaluate_multiple_UEs:
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

# Method 3 for end-to-end average BER (bar plot with mobility on x-axis)
if not evaluate_multiple_UEs:
    num_categories = len(mobilities)
    x = np.arange(num_categories)
    bar_width = 0.35
    x_labels = ['Scenario 1', 'Scenario 2', 'Scenario 3']
    plt.figure()
    plt.bar(x - bar_width/2, np.asarray(baseline_ber), width=bar_width, label='Baseline', color='#4F81BD')
    plt.bar(x + bar_width/2, np.asarray(ber), width=bar_width, label='SU MIMO', color='#C0504D')
    plt.yscale('log')
    plt.xticks(x, x_labels)
    plt.grid(True)
    plt.ylabel('BER')
    plt.title('Uncoded BER')
    plt.legend()
    plt.savefig("results/plots/BER_SU_MIMO")


############################### End-to-end average BLER ######################################
# Method 1 for end-to-end average BLER (mobility on x-axis)
# plt.figure()
# x_labels = ['Low mobility', 'Medium mobility', 'High mobility']
# x_values = np.arange(1,np.size(mobilities)+1)
# plt.semilogy(x_values, np.asarray(ldpc_ber), marker='o', label='SU MIMO')
# plt.semilogy(x_values, np.asarray(baseline_ldpc_ber), marker='*', label='Baseline')
# plt.grid(True)
# plt.xticks(x_values, x_labels)
# plt.ylabel('BLER')
# plt.title('BLER')
# plt.legend()
# plt.savefig("results/plots/BLER_SU_MIMO")


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

# Method 3 for end-to-end average BLER (bar plot with mobility on x-axis)
if not evaluate_multiple_UEs:
    num_categories = len(mobilities)
    x = np.arange(num_categories)
    bar_width = 0.35
    x_labels = ['Scenario 1', 'Scenario 2', 'Scenario 3']
    plt.figure()
    plt.bar(x - bar_width/2, np.asarray(baseline_ldpc_ber), width=bar_width, label='Baseline', color='#4F81BD')
    plt.bar(x + bar_width/2, np.asarray(ldpc_ber), width=bar_width, label='SU MIMO', color='#C0504D')
    plt.yscale('log')
    plt.xticks(x, x_labels)
    plt.grid(True)
    plt.ylabel('BLER')
    plt.title('BLER')
    plt.legend()
    plt.savefig("results/plots/BLER_SU_MIMO")


############################### Throughput and bitrate ######################################

if evaluate_multiple_UEs:
    print('Throughput: \n',  np.asarray(throughput))
else:
    print('Throughput: ',  throughput)
    print('Bitrate: ',  bitrate)

hold = 1