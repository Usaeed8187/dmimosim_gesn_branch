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

    temp_ldpc_ber = []
    temp_goodput = []
    temp_bitrate = []

    throughput_all_drops = []
    ldpc_ber_list_all_drops = []

    for drop_idx in np.arange(1, num_drops+1):
        
        try:
            if prediction_results:
                file_path = "results/channels_multiple_baseline/ZF/channels_{}_{}/baseline_results_pred.npz".format(curr_mobility, drop_idx)
            else:
                file_path = "results/channels_multiple_baseline/ZF/channels_{}_{}/baseline_results.npz".format(curr_mobility, drop_idx)
            data = np.load(file_path)
        except:
            continue

        throughput_all_drops.append(data['throughput'])
        ldpc_ber_list_all_drops.append(data['ldpc_ber_list'])

    throughput.append(np.mean(throughput_all_drops, axis=0))
    ldpc_ber_list.append(np.mean(ldpc_ber_list_all_drops))


############################### Throughput ######################################

print('Throughput: \n',  np.asarray(throughput))
print('Number of drops evaluated:', len(throughput_all_drops))

############################### BLER ######################################

print('BLER: \n',  np.asarray(ldpc_ber_list))
print('Number of drops evaluated:', len(ldpc_ber_list_all_drops))


hold = 1