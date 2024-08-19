# dMIMO Simulator

## Description
System simulator for the dMIMO project, using generated channels 
from [ns-3 simulator](https://www.nsnam.org/) 
and components from [Sionna](https://nvlabs.github.io/sionna/). 

The core simulator library is located in the "dmimo" folder,
the "sims" folder contains the top-level simulation scripts 
for baseline and SU-MIMO scenarios. Channel coefficients generated
from ns-3 simulator are stored in the "ns3" folder, and simulation 
results are saved in the "results" folder.

Additional documentation can be found in the "docs" folder, including
description of core component modules and instruction for setting up
the Sionna simulator on Linux systems.

## Getting started

Setup Git SSH command from Linux terminal (see https://code.vt.edu/help/user/ssh)
```
export GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519"
```
Clone the **main** branch of this repository
```
cd <workspace_dir>
git clone git@code.vt.edu:yiliang/dmimosim
cd dmimosim
```
Convert the ns-3 channel data to the optimized format.
See the section below for generating the ns-3 channel data.
```
cd ns3
python convert_ns3_channels.py <ns3_output_folder> channels
```
Activate the Conda environment
```
conda activate sionna
```
Run the simulation scripts
```
cd ../sims
python sim_baseline.py
python sim_mu_mimo.py
```


## Channel data generation

Build the ns-3 system simulator, 
see [the instructions](https://code.vt.edu/dmimo/ns3-system-simulation/-/blob/end-to-end/README.md)
from the ns-3 System Simulator.
```
git clone git@code.vt.edu:dmimo/ns3-system-simulation.git VT_dmimo_ns3
cd VT_dmimo_ns3
git checkout end-to-end
mkdir build
cd build/
cmake -DCMAKE_BUILD_TYPE=Release -DNS3_WARNINGS_AS_ERRORS=OFF ..
make -j4 scratch_dMIMO_channel_extraction_main
```

Generate the dMIMO channel data including MIMO channel coefficients and 
propagation losses.

```
cd VT_dmimo_ns3/scratch/dMIMO_channel_extraction/
python main.py --seed 3007 --scenario V2V-Urban --small_scale_fading --num_subframes 50 \
 --squad1_speed_km_h=3.0 --squad2_speed_km_h=3.0 --intra_sq1_rw_speed_km_h=0.3 \
 --intra_sq2_rw_speed_km_h=0.3 --buildings_file 1narrow.txt
```

## Development status

The codes are currently unstable in the initial development phase.





