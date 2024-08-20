import os
import sys
import shutil
import glob

import numpy as np


def read_ns3_channels(ns3_sim_folder, ns3_chans_folder, num_ofdm_syms, num_bs=1, num_ue=10, num_bs_ant=4, num_ue_ant=2, fft_size=512):

    slot_len = 14  # OFDM symbols per slot
    
    for slotidx in range(num_ofdm_syms // slot_len):
        
        Lts = np.zeros((num_ue, slot_len))
        Lrs = np.zeros((num_ue, slot_len))
        Ldm = np.zeros((num_ue + num_bs, num_ue + num_bs, slot_len))  
        
        Hts = np.zeros((num_ue*num_ue_ant, num_bs_ant, slot_len, fft_size), dtype=np.complex128)
        Hdm = np.zeros((num_ue*num_ue_ant+num_bs_ant, num_ue*num_ue_ant+num_bs_ant, slot_len, fft_size), dtype=np.complex128)
        Hrs = np.zeros((num_bs_ant, num_ue*num_ue_ant, slot_len, fft_size), dtype=np.complex128)
              
        for symidx in range(slot_len):
            time_idx = slotidx*slot_len + symidx
            foldername = os.path.join(ns3_sim_folder, "time_idx_{}/".format(time_idx))
            sys.stdout.write("\rReading folder {}".format(foldername))
            sys.stdout.flush()
            
            # Tx Squad (node 2 - 11)
            for k in range(num_ue):
                offset = 2*num_bs
                filename = foldername + "pl_0_{}.npy".format(k + offset)
                Lts[k,symidx] = np.load(filename)[0]

            # Rx Squad (node 12 - 21)
            for k in range(num_ue):
                offset = 2*num_bs + num_ue
                filename = foldername + "pl_{}_1.npy".format(k + offset)
                Lrs[k,symidx] = np.load(filename)[0]

            # D-MIMO links
            # node 0 is TxBB, node 1 is RxBB
            Ldm[0,0,symidx] = np.load(foldername + "pl_0_1.npy")[0]
            for m in range(num_ue): # RxUE
                offset = 2*num_bs + num_ue
                filename = foldername + "pl_0_{}.npy".format(m + offset)
                Ldm[m+1,0,symidx] = np.load(filename)[0]
                for n in range(num_ue): # TxUE
                    filename = foldername + "pl_{}_1.npy".format(n + 2*num_bs)
                    Ldm[0,n+1,symidx] = np.load(filename)[0]
                    filename = foldername + "pl_{}_{}.npy".format(n + 2*num_bs, m + 2*num_bs + num_ue);
                    Ldm[m+1,n+1,symidx] = np.load(filename)[0]
            
            # read TxSqud chanels
            for m in range(num_ue):
                offset = 2*num_bs
                filename = foldername + "hmat_0_{}.npy".format(m + offset)
                H = np.load(filename)
                assert  H.shape == (num_ue_ant, num_bs_ant, fft_size)
                Hts[m*num_ue_ant:(m+1)*num_ue_ant, :, symidx, :] = H

            # read RxSqud chanels
            for m in range(num_ue):
                offset = 2*num_bs + num_ue
                filename = foldername + "hmat_{}_1.npy".format(m + offset)
                H = np.load(filename)
                assert H.shape == (num_bs_ant, num_ue_ant, fft_size)
                Hrs[:,m*num_ue_ant:(m+1)*num_ue_ant, symidx, :] = H

            # read DMIMO chanels
            H = np.load(foldername + "hmat_0_1.npy")  # direct BS-BS channel
            Hdm[0:num_bs_ant,0:num_bs_ant,symidx,:] = H
            offset = 2*num_bs + num_ue
            for m in range(num_ue):  # RxSqud UE
                rxidx_low, rxidx_high = (num_bs_ant + m*num_ue_ant, num_bs_ant + (m+1)*num_ue_ant)
                # TxBS -> RxUE
                filename = foldername + "hmat_0_{}.npy".format(m + offset)
                H = np.load(filename)
                assert H.shape == (num_ue_ant, num_bs_ant, fft_size)
                Hdm[rxidx_low:rxidx_high, 0:num_bs_ant, symidx, :] = H
                for n in range(num_ue):  # TxSqud UE
                    txidx_low, txidx_high = (num_bs_ant + n*num_ue_ant, num_bs_ant + (n+1)*num_ue_ant)
                    # TxUE -> RxUE
                    filename = foldername + "hmat_{}_{}.npy".format(n + 2*num_bs, m + offset)
                    H = np.load(filename)
                    assert H.shape == (num_ue_ant, num_ue_ant, fft_size)
                    Hdm[rxidx_low:rxidx_high, txidx_low:txidx_high, symidx, :] = H
            for n in range(num_ue):
                txid_low, txid_high = (num_bs_ant + n*num_ue_ant, num_bs_ant + (n+1)*num_ue_ant)
                # TxUE -> RxBS
                filename = foldername + "hmat_{}_1.npy".format(m + offset)
                H = np.load(filename)
                assert H.shape == (num_bs_ant, num_ue_ant, fft_size)
                Hdm[0:num_bs_ant, txid_low:txid_high, symidx, :] = H
           
            # remove tempory time_idx_<symidx> subfolders
            # shutil.rmtree(foldername, ignore_errors=True)
            
        # save channel for current subframe/slot
        output_file = os.path.join(ns3_chans_folder, "dmimochans_{}.npz".format(slotidx))
        np.savez_compressed(output_file, Hdm=Hdm, Hrs=Hrs, Hts=Hts, Ldm=Ldm, Lrs=Lrs, Lts=Lts)


if __name__ == "__main__":
        
    if len(sys.argv) < 3:
        print("Usage: convert_ns3_channels <ns3_sim_folder> <ns3_chans_folder>")
        quit()

    ns3_sim_folder = sys.argv[1]
    ns3_chans_folder = sys.argv[2]
    if not os.path.isdir(ns3_sim_folder):
        print("Invalid ns-3 simulator data folder: {}".format(ns3_sim_folder))
        quit()

    if os.path.exists(ns3_chans_folder):
        print("Error: ns-3 channel folder exist!")
        quit()
    else:
        os.makedirs(ns3_chans_folder)

    ns3_outputs = glob.glob(ns3_sim_folder + "/time_idx_*/")
    if len(ns3_outputs) == 0:
        print("No ns-3 channels found in {}".format(ns3_sim_folder))
        quit()
    
    # Convert ns-3 channel for each subframe/slot
    # Each subfolder time_idx_<symidx> store channel for one OFDM symbol
    num_ofdm_syms = len(ns3_outputs)
    read_ns3_channels(ns3_sim_folder, ns3_chans_folder, num_ofdm_syms)
    
    print("\rFinish converting channels ({} snapshots)\n".format(num_ofdm_syms))
    
