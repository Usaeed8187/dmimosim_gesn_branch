#!/bin/bash

# Array of arguments
declare -a mobilities=("high_mobility")
declare -a drop_idx=("1" "2" "3")
declare -a rx_ues_arr=([0]="2 4 6" [1]="6" [2]="4 6")
# declare -a optional_args=("opt1" "opt2" "opt3")

# Loop through the arrays
for i in ${!mobilities[@]}; do
    for j in ${!drop_idx[@]}; do
        
        echo "Mobility: ${mobilities[$i]}, Drop idx: ${drop_idx[$j]}"
        python sims/sim_mu_mimo_kpi.py "${mobilities[$i]}" "${drop_idx[$j]}" ${rx_ues_arr[$j]}

    done
done