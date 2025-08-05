#!/bin/bash

# Array of arguments
declare -a mobilities=("high_mobility")
declare -a drop_idx=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
declare -a rx_ues_arr=("4" "6" "8" "10")
declare -a vector_inputs=("tx_ants")

# Loop through the arrays
for i in ${!mobilities[@]}; do
    for j in ${!drop_idx[@]}; do
        for k in ${!vector_inputs[@]}; do
            echo "Mobility: ${mobilities[$i]}, Drop idx: ${drop_idx[$j]}, Vector input: ${vector_inputs[$k]}"
            python sims/sim_mu_mimo_mass.py "${mobilities[$i]}" "${drop_idx[$j]}" "${vector_inputs[$k]}" ${rx_ues_arr} 
        done
    done
done