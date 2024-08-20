import os
import sys
import numpy as np


def convert_ipc_channels(ipc_data_folder, ns3_chans_folder):

    ### Load scenario configuration
    config = np.load(os.path.join(ipc_data_folder, '00_config.npz'), allow_pickle=True)['config'].item()
    
    ### TxSquad channels, shape is [num_txue * num_ue_ant, num_bs_ant, num_ofdm_sym, num_subcarrier]
    Hts = np.zeros((
        config['numSquad1UEs'] * config['numUEAnt'],
        config['numBSAnt'],
        config['numSymsPerSubframe'],
        config['numSCs']),
        dtype=np.complex64)
    Hts[:, :, :, :] = np.nan
    ### RxSquad channels, shape is [num_bs_ant, num_rxue * num_ue_ant, num_ofdm_sym, num_subcarrier]
    Hrs = np.zeros((
        config['numBSAnt'],
        config['numSquad2UEs'] * config['numUEAnt'],
        config['numSymsPerSubframe'],
        config['numSCs']),
        dtype=np.complex64)
    Hrs[:, :, :, :] = np.nan
    ### dMIMO channels, shape is [num_rxs_ant, num_txs_ant, num_ofdm_sym, num_subcarrier]
    Hdm = np.zeros((
        config['numSquad2UEs'] * config['numUEAnt'] + config['numBSAnt'],
        config['numSquad1UEs'] * config['numUEAnt'] + config['numBSAnt'],
        config['numSymsPerSubframe'],
        config['numSCs']),
        dtype=np.complex64)
    Hdm[:, :, :, :] = np.nan
    ### TxSquad pathloss in dB, shape is [num_txue, num_ofdm_sym]
    Lts = np.zeros((
        config['numSquad1UEs'],
        config['numSymsPerSubframe']),
        dtype=np.double)
    Lts[:, :] = np.nan
    ### RxSquad pathloss in dB, shape is [num_rxue, num_ofdm_sym]
    Lrs = np.zeros((
        config['numSquad2UEs'],
        config['numSymsPerSubframe']),
        dtype=np.double)
    Lrs[:, :] = np.nan
    ### dMIMO pathloss in dB, shape is [num_rxue+1, num_txue+1, num_ofdm_sym]
    Ldm = np.zeros((
        config['numSquad2UEs'] + 1,
        config['numSquad1UEs'] + 1,
        config['numSymsPerSubframe']),
        dtype=np.double)

    for slot_idx in range(config['numSubframes']):
        # Load channel data for current slot
        tBS_id = config['node_ids']['BS1_id']
        tUE_ids = config['node_ids']['Squad1UE_ids']
        rBS_id = config['node_ids']['BS2_id']
        rUE_ids = config['node_ids']['Squad2UE_ids']
        ueAnts = config['numUEAnt']
        bsAnts = config['numBSAnt']
        for sym_in_sf in range(config['numSymsPerSubframe']):
            file_t = slot_idx * config['numSymsPerSubframe'] + sym_in_sf
            with np.load(os.path.join(ipc_data_folder, f"ch_t_{file_t}.npz"), allow_pickle=True) as chfile:
                propLosses = chfile['propagationLossesDb'].item()
                Hmats = chfile['Hmats'].item()
                ## Load Hts and Lts
                for i, tUEid in enumerate(tUE_ids):
                    start = i * ueAnts
                    end = (i + 1) * ueAnts
                    Hts[start:end, :, sym_in_sf, :] = Hmats[(tBS_id, tUEid)]
                    Lts[i, sym_in_sf] = propLosses[(tBS_id, tUEid)]
                ## Load Hrs and Lrs
                for i, rUEid in enumerate(rUE_ids):
                    start = i * ueAnts
                    end = (i + 1) * ueAnts
                    Hrs[:, start:end, sym_in_sf, :] = Hmats[(rUEid, rBS_id)]
                    Lrs[i, sym_in_sf] = propLosses[(rUEid, rBS_id)]
                ## Load Hdm and Ldm
                for i, tNodeId in enumerate([tBS_id] + tUE_ids):
                    if i == 0:
                        start = 0
                        end = bsAnts
                    else:
                        start = bsAnts + (i - 1) * ueAnts
                        end = bsAnts + i * ueAnts
                    for j, rNodeId in enumerate([rBS_id] + rUE_ids):
                        if j == 0:
                            start2 = 0
                            end2 = bsAnts
                        else:
                            start2 = bsAnts + (j - 1) * ueAnts
                            end2 = bsAnts + j * ueAnts
                        Hdm[start:end, start2:end2, sym_in_sf, :] = Hmats[(rNodeId, tNodeId)]
                        Ldm[j, i, sym_in_sf] = propLosses[(rNodeId, tNodeId)]

        # save channel for current subframe/slot
        output_file = os.path.join(ns3_chans_folder, "dmimochans_{}.npz".format(slot_idx))
        np.savez_compressed(output_file, Hdm=Hdm, Hrs=Hrs, Hts=Hts, Ldm=Ldm, Lrs=Lrs, Lts=Lts)

    return config['numSubframes'] * config['numSymsPerSubframe']


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: convert_ipc_channels <ipc_data_folder> <ns3_chans_folder>")
        quit()

    ipc_data_folder = sys.argv[1]
    ns3_chans_folder = sys.argv[2]
    if not os.path.exists(os.path.join(ipc_data_folder, "00_config.npz")):
        print("Error: The directory {} does not contain the channel files!".format(ipc_data_folder))
        quit()

    if os.path.exists(ns3_chans_folder):
        print("Error: ns-3 channel folder exist!")
        quit()
    else:
        os.makedirs(ns3_chans_folder)

    # Convert ns-3 channel for each subframe/slot
    num_ofdm_syms = convert_ipc_channels(ipc_data_folder, ns3_chans_folder)
    print("\rFinish converting channels ({} snapshots)\n".format(num_ofdm_syms))

