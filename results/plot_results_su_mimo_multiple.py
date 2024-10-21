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

num_drops = 24

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

for mobility_idx in range(np.size(mobilities)):

    curr_mobility = mobilities[mobility_idx]

    if evaluate_multiple_UEs:

        temp_ldpc_ber = []
        temp_throughput = []
        temp_goodput = []
        temp_bitrate = []

        for ue_arr_idx in range(np.size(rx_ues_arr)):

            throughput_all_drops = []

            for drop_idx in np.arange(1, num_drops+1):

                if prediction_results:
                    file_path = "results/channels_multiple_su_mimo/results/channels_{}_{}/su_mimo_results_UE_{}_pred.npz".format(curr_mobility, drop_idx, rx_ues_arr[ue_arr_idx])
                else:
                    file_path = "results/channels_multiple_su_mimo/results/channels_{}_{}/su_mimo_results_UE_{}.npz".format(curr_mobility, drop_idx, rx_ues_arr[ue_arr_idx])
                data = np.load(file_path)

                throughput_all_drops.append(data['throughput'])
            
            temp_throughput.append(np.mean(throughput_all_drops, axis=0))

        throughput.append(temp_throughput)


############################### Throughput ######################################

if evaluate_multiple_UEs:
    print('Throughput: \n',  np.asarray(throughput))


hold = 1