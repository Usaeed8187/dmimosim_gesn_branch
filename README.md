# dMIMO Simulator

## Description
System simulator based on Sionna for the dMIMMO project.

## Getting started

(Optional) setup Git SSH command from Linux terminal (see https://code.vt.edu/help/user/ssh)
```
export GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519"
```
Clone the main branch of this repository
```
cd <workspace_dir>
git clone git@code.vt.edu:yiliang/dmimosim
cd dmimosim
```
Copy or generate the dMIMO channel data for simulation
```
mkdir -p ns3/channels
cp -a <ns3_data_folder/*.npz> ./ns3/channels
```
Activate Conda environment, and run the simulation scripts for SU-MIMO
```
conda activate sionna
```
Open Jupyter Notebooks for demo scripts
```
jupyter notebook -p 8080
```

## Development status

The codes are currently unstable in the initial development phase.





