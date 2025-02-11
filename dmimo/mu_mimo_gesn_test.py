import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
import matplotlib.pyplot as plt
import time

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.mimo import StreamManagement

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.fec.interleaving import RowColumnInterleaver, Deinterleaver

from sionna.mapping import Mapper, Demapper
from sionna.utils import BinarySource
from sionna.utils.metrics import compute_ber, compute_bler

from dmimo.config import Ns3Config, SimConfig, NetworkConfig, RCConfig
from dmimo.channel import dMIMOChannels, lmmse_channel_estimation, standard_rc_pred_freq_mimo, gesn_pred_freq_mimo
from dmimo.mimo import BDPrecoder, BDEqualizer, ZFPrecoder, rankAdaptation, linkAdaptation
from dmimo.mimo import update_node_selection
from dmimo.utils import add_frequency_offset, add_timing_offset, cfo_val, sto_val, compute_UE_wise_BER, compute_UE_wise_SER

from .txs_mimo import TxSquad
from .rxs_mimo import RxSquad


class MU_MIMO(Model):

    def __init__(self, cfg: SimConfig, rc_config: RCConfig, **kwargs):
        """
        Create MU-MIMO simulation object

        :param cfg: simulation settings
        """
        super().__init__(trainable=False, **kwargs)

        self.cfg = cfg
        self.rc_config = rc_config
        self.batch_size = cfg.num_slots_p2  # batch processing for all slots in phase 2

        # CFO and STO settings
        self.sto_sigma = sto_val(cfg, cfg.sto_sigma)
        self.cfo_sigma = cfo_val(cfg, cfg.cfo_sigma)

        # To use sionna-compatible interface, regard TxSquad as one BS transmitter
        # A 4-antennas basestation is regarded as the combination of two 2-antenna UEs
        self.num_streams_per_tx = cfg.num_tx_streams

        self.num_txs_ant = 2 * cfg.num_tx_ue_sel + 4  # gNB always present with 4 antennas
        self.num_ue_ant = 2  # assuming 2 antennas per UE for reshaping data/channels
        if cfg.ue_indices is None:
            # no rank/link adaptation
            self.num_rxs_ant = self.num_streams_per_tx
            self.num_rx_ue = self.num_rxs_ant // self.num_ue_ant
        else:
            # rank adaptation support
            self.num_rxs_ant = np.sum([len(val) for val in cfg.ue_indices])
            self.num_rx_ue = self.num_rxs_ant // self.num_ue_ant
            if cfg.ue_ranks is None:
                cfg.ue_ranks = self.num_ue_ant  # no rank adaptation

        # Create an RX-TX association matrix
        # rx_tx_association[i,j]=1 means that receiver i gets at least one stream from transmitter j.
        rx_tx_association = np.ones((self.num_rx_ue, 1))

        # Instantiate a StreamManagement object
        # This determines which data streams are determined for which receiver.
        sm = StreamManagement(rx_tx_association, self.num_streams_per_tx)

        # Adjust guard subcarriers for channel estimation grid
        csi_effective_subcarriers = (cfg.fft_size // self.num_txs_ant) * self.num_txs_ant
        csi_guard_carriers_1 = (cfg.fft_size - csi_effective_subcarriers) // 2
        csi_guard_carriers_2 = (cfg.fft_size - csi_effective_subcarriers) - csi_guard_carriers_1

        # Resource grid for channel estimation
        self.rg_csi = ResourceGrid(num_ofdm_symbols=14,
                                   fft_size=cfg.fft_size,
                                   subcarrier_spacing=cfg.subcarrier_spacing,
                                   num_tx=1,
                                   num_streams_per_tx=self.num_txs_ant,
                                   cyclic_prefix_length=cfg.cyclic_prefix_len,
                                   num_guard_carriers=[csi_guard_carriers_1, csi_guard_carriers_2],
                                   dc_null=False,
                                   pilot_pattern="kronecker",
                                   pilot_ofdm_symbol_indices=[2, 11])

        # Adjust guard subcarriers for different number of streams
        effective_subcarriers = (csi_effective_subcarriers // self.num_streams_per_tx) * self.num_streams_per_tx
        guard_carriers_1 = (csi_effective_subcarriers - effective_subcarriers) // 2
        guard_carriers_2 = (csi_effective_subcarriers - effective_subcarriers) - guard_carriers_1
        guard_carriers_1 += csi_guard_carriers_1
        guard_carriers_2 += csi_guard_carriers_2

        # OFDM resource grid (RG) for normal transmission
        self.rg = ResourceGrid(num_ofdm_symbols=14,
                               fft_size=cfg.fft_size,
                               subcarrier_spacing=cfg.subcarrier_spacing,
                               num_tx=1,
                               num_streams_per_tx=self.num_streams_per_tx,
                               cyclic_prefix_length=64,
                               num_guard_carriers=[guard_carriers_1, guard_carriers_2],
                               dc_null=False,
                               pilot_pattern="kronecker",
                               pilot_ofdm_symbol_indices=[2, 11])

        # Update number of data bits and LDPC params
        cfg.ldpc_n = int(2 * self.rg.num_data_symbols)  # Number of coded bits
        cfg.ldpc_k = int(cfg.ldpc_n * cfg.code_rate)  # Number of information bits
        self.num_codewords = cfg.modulation_order // 2  # number of codewords per frame
        self.num_bits_per_frame = cfg.ldpc_k * self.num_codewords * self.num_streams_per_tx
        self.num_uncoded_bits_per_frame = cfg.ldpc_n * self.num_codewords * self.num_streams_per_tx

        # The encoder maps information bits to coded bits
        self.encoder = LDPC5GEncoder(cfg.ldpc_k, cfg.ldpc_n)

        # LDPC interleaver
        self.intlvr = RowColumnInterleaver(3072, axis=-1)  # fixed design for current RG config
        self.dintlvr = Deinterleaver(interleaver=self.intlvr)

        # The mapper maps blocks of information bits to constellation symbols
        self.mapper = Mapper("qam", cfg.modulation_order)

        # The resource grid mapper maps symbols onto an OFDM resource grid
        self.rg_mapper = ResourceGridMapper(self.rg)

        # The zero forcing and block diagonalization precoder
        self.bd_precoder = BDPrecoder(self.rg, sm, return_effective_channel=True)
        self.zf_precoder = ZFPrecoder(self.rg, sm, return_effective_channel=True)
        self.bd_equalizer = BDEqualizer(self.rg, sm)

        # The LS channel estimator will provide channel estimates and error variances
        self.ls_estimator = LSChannelEstimator(self.rg, interpolation_type="lin")

        # The LMMSE equalizer will provide soft symbols together with noise variance estimates
        self.lmmse_equ = LMMSEEqualizer(self.rg, sm)

        # The demapper produces LLR for all coded bits
        self.demapper = Demapper("maxlog", "qam", cfg.modulation_order)

        # The decoder provides hard-decisions on the information bits
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)

    def call(self, dmimo_chans: dMIMOChannels, info_bits=None):
        """
        Signal processing for one MU-MIMO transmission cycle (P2)

        :param dmimo_chans: dMIMO channels
        :param info_bits: information bits
        :return: decoded bits, uncoded BER, demodulated QAM symbols (for debugging purpose)
        """

        if self.cfg.perfect_csi is True:
            # Perfect channel estimation
            h_freq_csi, rx_snr_db = dmimo_chans.load_channel(slot_idx=self.cfg.first_slot_idx - self.cfg.csi_delay,
                                                             batch_size=self.batch_size)
        elif self.cfg.csi_prediction is True:
            if self.cfg.predictor == 'standard_rc':
                rc_predictor = standard_rc_pred_freq_mimo('MU_MIMO', num_rx_ant = 4 + self.cfg.num_rx_ue_sel*2)
            elif self.cfg.predictor == 'gesn':
                rc_predictor = gesn_pred_freq_mimo('MU_MIMO', self.rc_config, num_rx_ant = 4 + self.cfg.num_rx_ue_sel*2, 
                                                    num_tx_ant=self.cfg.num_tx_ue_sel*2 + 4, max_adjacency='all', method='per_node_pair', 
                                                    num_neurons=16, edge_weighting_method='model_based') # edge_weighting_method: 'model_based', 'grad_descent'
            
            # Get CSI history
            # TODO: optimize channel estimation and optimization procedures (currently very slow)
            h_freq_csi_history = rc_predictor.get_csi_history(self.cfg.first_slot_idx, self.cfg.csi_delay,
                                                                self.rg_csi, dmimo_chans)
            # Do channel prediction

            # Get Vanilla RC NMSE for comparison
            self.rc_config.enable_window = True
            rc_predictor_vanilla = standard_rc_pred_freq_mimo('MU_MIMO', self.rc_config, num_rx_ant = 4 + self.cfg.num_rx_ue_sel*2, 
                                                    num_neurons=self.rc_config.num_neurons)
            h_freq_csi_true, rx_snr_db = dmimo_chans.load_channel(slot_idx=self.cfg.first_slot_idx,
                                    batch_size=self.batch_size)
            h_freq_csi_true = rc_predictor_vanilla.rb_mapper(h_freq_csi_true)
            h_freq_csi_vanilla = rc_predictor_vanilla.predict(h_freq_csi_history)
            pred_nmse_wesn = rc_predictor_vanilla.cal_nmse(h_freq_csi_true[0,...], h_freq_csi_vanilla[0,...])

            self.rc_config.enable_window = False
            rc_predictor_vanilla = standard_rc_pred_freq_mimo('MU_MIMO', self.rc_config, num_rx_ant = 4 + self.cfg.num_rx_ue_sel*2, num_neurons=self.rc_config.num_neurons)
            h_freq_csi_vanilla = rc_predictor_vanilla.predict(h_freq_csi_history)
            pred_nmse_esn = rc_predictor_vanilla.cal_nmse(h_freq_csi_true[0,...], h_freq_csi_vanilla[0,...])
            
            # Compare with gradient descent GESN
            h_freq_csi_true = np.squeeze(h_freq_csi_true).transpose([0,1,2,4,3])
            self.rc_config.enable_window = False
            rc_predictor = gesn_pred_freq_mimo('MU_MIMO', self.rc_config, num_rx_ant = 4 + self.cfg.num_rx_ue_sel*2, 
                                                    num_tx_ant=self.cfg.num_tx_ue_sel*2 + 4, max_adjacency='all', method=self.cfg.graph_formulation, 
                                                    num_neurons=16, edge_weighting_method='grad_descent') # edge_weighting_method: 'model_based', 'grad_descent'
            h_freq_csi_grad_descent = rc_predictor.predict(h_freq_csi_history, h_freq_csi_true[0,...])

            pred_nmse_gesn_per_antenna_pair = rc_predictor.cal_nmse(h_freq_csi_true[0,...], h_freq_csi_grad_descent)

            self.rc_config.enable_window = True
            rc_predictor = gesn_pred_freq_mimo('MU_MIMO', self.rc_config, num_rx_ant = 4 + self.cfg.num_rx_ue_sel*2, 
                                                    num_tx_ant=self.cfg.num_tx_ue_sel*2 + 4, max_adjacency='all', method=self.cfg.graph_formulation, 
                                                    num_neurons=16, edge_weighting_method='grad_descent') # edge_weighting_method: 'model_based', 'grad_descent'
            h_freq_csi_grad_descent = rc_predictor.predict(h_freq_csi_history, h_freq_csi_true[0,...])

            pred_nmse_wgesn_per_antenna_pair = rc_predictor.cal_nmse(h_freq_csi_true[0,...], h_freq_csi_grad_descent)

            
            # Print out all results
            print("\nESN: ", pred_nmse_esn)
            print("WESN : ", pred_nmse_wesn)
            print("GESN (per_antenna_pair): ", pred_nmse_gesn_per_antenna_pair)
            print("WGESN (per_antenna_pair): ", pred_nmse_wgesn_per_antenna_pair)
            print("Window size:", rc_predictor_vanilla.window_length)

            # Test plots
            plot = False
            if plot:
                h_freq_csi_true, rx_snr_db = dmimo_chans.load_channel(slot_idx=self.cfg.first_slot_idx,
                                                             batch_size=self.batch_size)
                h_freq_csi_true = np.squeeze(h_freq_csi_true).transpose([0,1,2,4,3])
                h_freq_csi_true = rc_predictor.rb_mapper(h_freq_csi_true)

                h_freq_csi_outdated = np.squeeze(h_freq_csi_history).transpose([0,1,2,4,3])
                h_freq_csi_outdated = rc_predictor.rb_mapper(h_freq_csi_outdated)

                h_freq_csi_predicted_gesn = h_freq_csi
                if not rc_predictor_vanilla.rb_granularity:
                    h_freq_csi_predicted_vanilla = rc_predictor.rb_mapper(tf.transpose(h_freq_csi_vanilla[:,0,:,0,...], perm=[0,1,2,4,3]))
                else:
                    h_freq_csi_predicted_vanilla = tf.transpose(h_freq_csi_vanilla[:,0,:,0,...], perm=[0,1,2,4,3])

                debug_rx_ant = 0
                debug_tx_ant = 0
                debug_ofdm_sym = 1
                
                plt.figure()
                plt.plot(np.real(h_freq_csi_predicted_gesn[debug_rx_ant, debug_tx_ant, :, debug_ofdm_sym]), label="GESN Predicted Channel")
                plt.plot(np.real(h_freq_csi_predicted_vanilla[0, debug_rx_ant, debug_tx_ant, :, debug_ofdm_sym]), label="Vanilla ESN Predicted Channel")
                # plt.plot(np.real(h_freq_csi_outdated[-1, debug_rx_ant, debug_tx_ant, :, debug_ofdm_sym]), label="Outdated Channel")
                plt.plot(np.real(h_freq_csi_true[0, debug_rx_ant, debug_tx_ant, :, debug_ofdm_sym]), label="Ground Truth Channel")
                plt.legend()
                plt.savefig('prediction_comparison')
                # plt.show()
            plot = False

        else:
            # LMMSE channel estimation
            h_freq_csi, err_var_csi = lmmse_channel_estimation(dmimo_chans, self.rg_csi,
                                                               slot_idx=self.cfg.first_slot_idx - self.cfg.csi_delay,
                                                               cfo_sigma=self.cfo_sigma, sto_sigma=self.sto_sigma)
            _, rx_snr_db = dmimo_chans.load_channel(slot_idx=self.cfg.first_slot_idx - self.cfg.csi_delay, batch_size=self.batch_size)
                

        return [pred_nmse_esn, pred_nmse_wesn, pred_nmse_gesn_per_antenna_pair, pred_nmse_wgesn_per_antenna_pair]


def sim_mu_mimo(cfg: SimConfig, rc_config:RCConfig):
    """
    Simulation of MU-MIMO scenarios using different settings

    :param cfg: simulation settings
    :return: [uncoded_ber, coded_ber], [goodbits, userbits, ratedbits]
    """

    # dMIMO channels from ns-3 simulator
    ns3cfg = Ns3Config(data_folder=cfg.ns3_folder, total_slots=cfg.total_slots)
    dmimo_chans = dMIMOChannels(ns3cfg, "dMIMO", add_noise=True)
    cfg.bs_txpwr_dbm = ns3cfg.bs_txpwr_dbm

    # UE selection
    if cfg.enable_ue_selection is True:
        tx_ue_mask, rx_ue_mask = update_node_selection(cfg)
        ns3cfg.update_ue_mask(tx_ue_mask, rx_ue_mask)

    # Create MU-MIMO simulation
    mu_mimo = MU_MIMO(cfg, rc_config)

    # The binary source will create batches of information bits
    binary_source = BinarySource()
    info_bits = binary_source([cfg.num_slots_p2, mu_mimo.num_bits_per_frame])

    # MU-MIMO transmission (P2)
    [pred_nmse_esn, pred_nmse_wesn, pred_nmse_gesn_per_antenna_pair, pred_nmse_wgesn_per_antenna_pair] = mu_mimo(dmimo_chans, info_bits)

    return pred_nmse_esn, pred_nmse_wesn, pred_nmse_gesn_per_antenna_pair, pred_nmse_wgesn_per_antenna_pair


def sim_mu_mimo_all(cfg: SimConfig, rc_config:RCConfig):
    """"
    Simulation of MU-MIMO scenario according to the frame structure
    """

    total_cycles = 0
    
    pred_nmse_esn = []
    pred_nmse_wesn = []
    pred_nmse_gesn_per_antenna_pair = []
    pred_nmse_wgesn_per_antenna_pair = []
    for first_slot_idx in np.arange(cfg.start_slot_idx, cfg.total_slots, cfg.num_slots_p1 + cfg.num_slots_p2):

        print("first_slot_idx: ", first_slot_idx, "\n")

        total_cycles += 1
        cfg.first_slot_idx = first_slot_idx
        # try:
        #     curr_pred_nmse_gesn, curr_pred_nmse_vanilla = sim_mu_mimo(cfg)

        #     pred_nmse_gesn.append(curr_pred_nmse_gesn)
        #     pred_nmse_vanilla.append(curr_pred_nmse_vanilla)

        # except:
        #     print("Continued \n")
        #     continue

        curr_pred_nmse_esn, curr_pred_nmse_wesn, curr_pred_nmse_gesn_per_antenna_pair, curr_pred_nmse_wgesn_per_antenna_pair = sim_mu_mimo(cfg, rc_config)

        pred_nmse_esn.append(curr_pred_nmse_esn)
        pred_nmse_wesn.append(curr_pred_nmse_wesn)
        pred_nmse_gesn_per_antenna_pair.append(curr_pred_nmse_gesn_per_antenna_pair)
        pred_nmse_wgesn_per_antenna_pair.append(curr_pred_nmse_wgesn_per_antenna_pair)


    return pred_nmse_esn, pred_nmse_wesn, pred_nmse_gesn_per_antenna_pair, pred_nmse_wgesn_per_antenna_pair