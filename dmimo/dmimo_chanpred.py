import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
import matplotlib.pyplot as plt
import time

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, RemoveNulledSubcarriers
from sionna.mimo import StreamManagement

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.fec.interleaving import RowColumnInterleaver, Deinterleaver

from sionna.mapping import Mapper, Demapper
from sionna.utils import BinarySource
from sionna.utils.metrics import compute_ber, compute_bler
from sionna.channel import ApplyOFDMChannel
from sionna.utils import expand_to_rank, complex_normal, flatten_last_dims

from dmimo.config import Ns3Config, SimConfig, NetworkConfig, RCConfig
from dmimo.channel import dMIMOChannels, lmmse_channel_estimation, standard_rc_pred_freq_mimo, gesn_pred_freq_dmimo
from dmimo.channel import multimode_esn_pred
from dmimo.channel import twomode_wesn_pred
from dmimo.channel import twomode_graph_wesn_pred, twomode_graph_wesn_pred_v2
from dmimo.channel.wesn_pred import WESN
from dmimo.channel.kalman_pred_freq_dmimo import kalman_pred_freq_dmimo
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

        self.num_txs_ant = 2 * cfg.num_tx_ue_sel + 4  # gNB always present with 4 antennas
        self.num_ue_ant = 2  # assuming 2 antennas per UE for reshaping data/channels
        self.num_rx_ue = cfg.num_rx_ue_sel + 2
        if cfg.ue_ranks is None:
            cfg.ue_ranks = 1  # no rank adaptation

        # To use sionna-compatible interface, regard TxSquad as one BS transmitter
        # A 4-antennas basestation is regarded as the combination of two 2-antenna UEs
        self.num_streams_per_tx = cfg.num_tx_streams * self.num_rx_ue

        assert self.cfg.num_tx_ue_sel*2 + 4 >= self.num_streams_per_tx, "TxSquad should have antennas >= transmit streams"

        # Create an RX-TX association matrix
        # rx_tx_association[i,j]=1 means that receiver i gets at least one stream from transmitter j.
        rx_tx_association = np.ones((self.num_rx_ue, 1), dtype=int)

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

        self.apply_channel = ApplyOFDMChannel(add_awgn=False, dtype=tf.as_dtype(tf.complex64))

        self.removed_nulled_scs = RemoveNulledSubcarriers(self.rg)

    def call(self, dmimo_chans: dMIMOChannels, info_bits=None):
        """
        Signal processing for one MU-MIMO transmission cycle (P2)

        :param dmimo_chans: dMIMO channels
        :param info_bits: information bits
        :return: decoded bits, uncoded BER, demodulated QAM symbols (for debugging purpose)
        """

        if not self.cfg.return_estimated_channel:
            # LDPC encoder processing
            info_bits = tf.reshape(info_bits, [self.batch_size, 1, self.rg.num_streams_per_tx,
                                            self.num_codewords, self.encoder.k])
            c = self.encoder(info_bits)
            c = tf.reshape(c, [self.batch_size, 1, self.rg.num_streams_per_tx, self.num_codewords * self.encoder.n])

            # Interleaving for coded bits
            d = self.intlvr(c)

            # QAM mapping for the OFDM grid
            x = self.mapper(d)
            x_rg = self.rg_mapper(x)

        if self.cfg.perfect_csi is True:
            # Perfect channel estimation
            h_freq_csi, rx_snr_db = dmimo_chans.load_channel(slot_idx=self.cfg.first_slot_idx - self.cfg.csi_delay,
                                                             batch_size=self.batch_size)
        elif self.cfg.csi_prediction is True:
            if self.cfg.predictor == 'standard_rc':
                rc_predictor = standard_rc_pred_freq_mimo('MU_MIMO', num_rx_ant = 4 + self.cfg.num_rx_ue_sel*2)
            elif self.cfg.predictor == 'gesn':
                rc_predictor = gesn_pred_freq_dmimo('MU_MIMO', self.rc_config, num_rx_ant = 4 + self.cfg.num_rx_ue_sel*2, 
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
            
            h_freq_csi_vanilla = rc_predictor_vanilla.predict(h_freq_csi_history[-3:,...])
            pred_nmse_wesn = self.nmse(h_freq_csi_true[0,...], h_freq_csi_vanilla[0,...])
            print("Vanilla WESN NMSE: ", pred_nmse_wesn)



            # Put channel feedback history into RB granularity
            h_freq_csi_history = rc_predictor_vanilla.rb_mapper(h_freq_csi_history)
            T, _, _, RxAnt, _, TxAnt, num_syms, RB = h_freq_csi_history.shape












            # Use 2-mode ESN to do WESN based prediction (all together)
            twomode_predictor = twomode_wesn_pred(rc_config=self.rc_config, 
                                                 num_freq_re=RB, 
                                                 num_rx_ant=RxAnt, 
                                                 num_tx_ant=TxAnt
                                                 )
            h_freq_csi_twomode_all = twomode_predictor.predict(h_freq_csi_history)
            pred_nmse_twomode_wesn_all = self.nmse(h_freq_csi_true[0,...], h_freq_csi_twomode_all[0,...])
            print("Two-Mode (All) WESN NMSE: ", pred_nmse_twomode_wesn_all)
















            # Use 2-mode ESN to do WESN based prediction (per_tx_rx_node_pair)

            h_freq_csi_twomode_per_link = np.zeros(h_freq_csi_history[0,...].shape, dtype=h_freq_csi_history.dtype)
            num_bs_ant = 4
            num_ue_ant = 2

            for tx_node_idx in range(self.cfg.num_tx_ue_sel+1):
                for rx_node_idx in range(self.cfg.num_rx_ue_sel+1):
                    if tx_node_idx == 0:
                        tx_ant_idx = np.arange(0,num_bs_ant)
                    else:
                        tx_ant_idx = np.arange(num_bs_ant + (tx_node_idx-1)*num_ue_ant,num_bs_ant + (tx_node_idx)*num_ue_ant)
                    TxAnt = len(tx_ant_idx)

                    if rx_node_idx == 0:
                        rx_ant_idx = np.arange(0,num_bs_ant)
                    else:
                        rx_ant_idx = np.arange(num_bs_ant + (rx_node_idx-1)*num_ue_ant,num_bs_ant + (rx_node_idx)*num_ue_ant)
                    RxAnt = len(rx_ant_idx)

                    curr_h_freq_csi_history = h_freq_csi_history[:,:,:,rx_ant_idx,:,...]
                    curr_h_freq_csi_history = curr_h_freq_csi_history[:,:,:,:,:,tx_ant_idx,...]
                    
                    twomode_predictor = twomode_wesn_pred(rc_config=self.rc_config, 
                                                num_freq_re=RB, 
                                                num_rx_ant=RxAnt, 
                                                num_tx_ant=TxAnt
                                                )
                    rx_idx, tx_idx = np.ix_(rx_ant_idx, tx_ant_idx)
                    tmp = np.asarray(twomode_predictor.predict(curr_h_freq_csi_history))
                    h_freq_csi_twomode_per_link[:, :, rx_idx, :, tx_idx, :, :] = tmp.transpose(2, 4, 0, 1, 3, 5, 6)

            pred_nmse_twomode_wesn_per_link = self.nmse(h_freq_csi_true[0,...], h_freq_csi_twomode_per_link[0,...])
            print("Two-Mode (Per-Link) WESN NMSE: ", pred_nmse_twomode_wesn_per_link)

            # plt.figure()
            # plt.plot(np.real(h_freq_csi_true[0,0,0,0,0,0,:]), label='true future channel')
            # plt.plot(np.real(h_freq_csi_twomode_all[0,0,0,0,0,0,:]), label='Twomode (All)')
            # plt.plot(np.real(h_freq_csi_twomode_per_link[0,0,0,0,0,0,:]), label='Twomode (Per-Link)')
            # plt.legend()
            # plt.savefig('b')















            # Use 2-mode Graph ESN to do WESN based prediction
            # T, _, _, RxAnt, _, TxAnt, num_syms, RB = h_freq_csi_history.shape
            # graph_twomode_predictor = twomode_graph_wesn_pred(rc_config=self.rc_config, 
            #                                      num_freq_re=RB, 
            #                                      num_rx_ant=RxAnt, 
            #                                      num_tx_ant=TxAnt
            #                                     )
            # h_freq_csi_graph_twomode = graph_twomode_predictor.predict(h_freq_csi_history)
            # pred_nmse_graph_twomode_wesn = self.nmse(h_freq_csi_true[0,...], h_freq_csi_graph_twomode[0,...])
            # print("Two-Mode Graph WESN (V1) NMSE: ", pred_nmse_graph_twomode_wesn)

            # plt.figure()
            # plt.plot(np.real(h_freq_csi_true[0,0,0,0,0,0,:]), label='true future channel')
            # plt.plot(np.real(h_freq_csi_vanilla[0,0,0,0,0,0,:]), label='Vanilla')
            # plt.plot(np.real(h_freq_csi_twomode[0,0,0,0,0,0,:]), label='Twomode')
            # plt.plot(np.real(h_freq_csi_graph_twomode[0,0,0,0,0,0,:]), label='Graph Twomode')
            # plt.legend()
            # plt.savefig('a')














            # Use 2-mode Graph ESN to do WESN based prediction
            T, _, _, RxAnt, _, TxAnt, num_syms, RB = h_freq_csi_history.shape
            graph_twomode_predictor = twomode_graph_wesn_pred_v2(rc_config=self.rc_config, 
                                                 num_freq_re=RB, 
                                                 num_rx_ant=RxAnt, 
                                                 num_tx_ant=TxAnt,
                                                #  adjacency_method='eigenmode_cov'
                                                )
            h_freq_csi_graph_twomode = graph_twomode_predictor.predict(h_freq_csi_history)
            pred_nmse_graph_twomode_wesn = self.nmse(h_freq_csi_true[0,...], h_freq_csi_graph_twomode[0,...])
            print("Two-Mode Graph WESN (V2) NMSE: ", pred_nmse_graph_twomode_wesn)


            plt.figure()
            rx_ant = 2
            tx_ant = 3
            ofdm_sym = 5
            plt.plot(np.real(h_freq_csi_true[0,0,rx_ant,0,tx_ant,ofdm_sym,:]), label='true future channel')
            plt.plot(np.real(h_freq_csi_twomode_per_link[0,0,rx_ant,0,tx_ant,ofdm_sym,:]), label='Twomode (Per_link)')
            plt.plot(np.real(h_freq_csi_graph_twomode[0,0,rx_ant,0,tx_ant,ofdm_sym,:]), label='Graph Twomode')
            plt.legend()
            plt.savefig('a')

            hold = 1















            # Use 3-mode ESN to do WESN based prediction
            # h_freq_csi_history = rc_predictor_vanilla.rb_mapper(h_freq_csi_history)
            # T, _, _, RxAnt, _, TxAnt, num_syms, RB = h_freq_csi_history.shape
            # multimode_predictor = multimode_esn_pred(
            #     N_f=RB, N_t=TxAnt, N_r=RxAnt,
            #     d_f=64, d_t=4, d_r=4,
            #     window_len=self.rc_config.window_length,
            #     debug=False,          # <— turn on
            #     safe_solve=False      # <— try safer linear algebra first
            # )
            # h_freq_csi_multimode_wesn = multimode_predictor.predict(
            #     h_freq_csi_history,
            #     washout=0,
            #     lambdas=(1.0, 1.0, 1.0),   # start a bit stronger if you see big cond(A)
            #     iters=2
            # )

            # multimode_predictor = multimode_esn_pred(   N_f=RB,
            #                                             N_t=TxAnt,
            #                                             N_r=RxAnt,
            #                                             d_f=64,
            #                                             d_t=4,
            #                                             d_r=4,
            #                                             window_len=self.rc_config.window_length)
            # h_freq_csi_multimode_wesn = multimode_predictor.predict(h_freq_csi_history)
            # pred_nmse_multimode_wesn = self.nmse(h_freq_csi_true[0,...], h_freq_csi_multimode_wesn)
            # print("Multimode ESN NMSE: ", pred_nmse_multimode_wesn)


            # norm = np.mean(np.real(h_freq_csi_true[0,0,0,0,0,0,:])) / np.mean(np.real(h_freq_csi_multimode_wesn[0,0,0,0,0,:]))
            # plt.figure()
            # plt.plot(np.real(h_freq_csi_true[0,0,0,0,0,0,:]), label='true future channel')
            # plt.plot(np.real(h_freq_csi_history[-1,0,0,0,0,0,0,:]), label='outdated channel')
            # plt.plot(norm * np.real(h_freq_csi_multimode_wesn[0,0,0,0,0,:]), label='predicted channel')
            # plt.legend()
            # plt.savefig('a')

            # hold = 1















            # # Use the standardized code to do WESN based prediction
            # h_freq_csi_history = rc_predictor_vanilla.rb_mapper(h_freq_csi_history)
            # T, _, _, RxAnt, _, TxAnt, num_syms, RB = h_freq_csi_history.shape
            # Din_raw = int(RB)
            # h_freq_csi_standardized_wesn = np.zeros((1, RxAnt, 1, TxAnt, num_syms, RB), dtype=np.complex64)

            # for rx_ant_idx in range(RxAnt):
            #     for tx_ant_idx in range(TxAnt):
            #         # h_freq_csi_history = h_freq_csi_history[:,:,:,RxAnt:RxAnt+1,:,TxAnt:TxAnt+1,...]
            #         X_seqs, Y_seqs = [], []
            #         for t in range(T-1):
            #             # Inputs at time t: [10,16,14,43] → [14,10,16,43] → [14, Din_raw]
            #             x_t = tf.transpose(h_freq_csi_history[t,   0, 0, rx_ant_idx:rx_ant_idx+1, 0, tx_ant_idx:tx_ant_idx+1, :, :], perm=[2, 0, 1, 3])
            #             x_t = tf.reshape(x_t, [num_syms, Din_raw])
            #             # Targets at time t+1, aligned per symbol i
            #             y_tp1 = tf.transpose(h_freq_csi_history[t+1, 0, 0, rx_ant_idx:rx_ant_idx+1, 0, tx_ant_idx:tx_ant_idx+1, :, :], perm=[2, 0, 1, 3])
            #             y_tp1 = tf.reshape(y_tp1, [num_syms, Din_raw])
            #             X_seqs.append(x_t)
            #             Y_seqs.append(y_tp1)
                    
            #         X = tf.stack(X_seqs, axis=0)   # [batch=2, timesteps=14, Din_raw]
            #         Y = tf.stack(Y_seqs, axis=0)   # [batch=2, timesteps=14, Din_raw]

            #         # if X.dtype.is_complex:
            #         #     X = tf.concat([tf.math.real(X), tf.math.imag(X)], axis=-1)
            #         #     Y = tf.concat([tf.math.real(Y), tf.math.imag(Y)], axis=-1)

            #         Din = int(X.shape[-1])
            #         Dout = int(Y.shape[-1])

            #         # --- Build and train WESN ---
            #         # win_len=0 lets the RNN carry context; set >0 if you also want explicit input lags per step
            #         layer = WESN(units=self.rc_config.num_neurons,
            #                      win_len=self.rc_config.window_length,
            #                      readout_units=Dout,
            #                      connectivity=self.rc_config.W_tran_sparsity,
            #                      spectral_radius=self.rc_config.W_tran_radius,
            #                      input_scale=self.rc_config.input_scale,
            #                      inv_regularization=self.rc_config.regularization)
            #         inp = tf.keras.Input(shape=(num_syms, Din))   # (timesteps=14 fixed; you could also use None)
            #         out = layer(inp)
            #         model = tf.keras.Model(inp, out)
            #         layer.ls_initialize(X, Y)
            #         model.compile(optimizer="adam", loss="mse")
            #         model.fit(X, Y, epochs=0, verbose=1)     # small dataset; consider more epochs or weight decay

            #         # --- Inference: predict subframe at t=3 from subframe at t=2 ---
            #         X_last = tf.transpose(h_freq_csi_history[-1, 0, 0, rx_ant_idx:rx_ant_idx+1, 0, tx_ant_idx:tx_ant_idx+1, :, :], perm=[2, 0, 1, 3])   # [14,10,16,43]
            #         X_last = tf.reshape(X_last, [1, num_syms, Din_raw])                       # [1,14,Din_raw]
            #         # If the model was trained on stacked real/imaginary parts the
            #         # expected input dimension will be twice the raw dimension.
            #         # In that case, stack here as well.  Otherwise, feed the complex
            #         # tensor directly.
            #         if X_last.dtype.is_complex and Din == 2 * Din_raw:
            #             X_last = tf.concat([tf.math.real(X_last),
            #                                 tf.math.imag(X_last)], axis=-1)

            #         Y_pred = model.predict(X_last, verbose=0)[0]   # [14, Dout]

            #         # If the network output stacks real and imaginary parts, convert
            #         # back to a complex tensor and original shape [14,10,16,43].
            #         if Dout == 2 * Din_raw:
            #             Dhalf = Dout // 2
            #             Y_pred = tf.complex(Y_pred[:, :Dhalf],
            #                                  Y_pred[:, Dhalf:])
                    
            #         Y_pred_c = tf.reshape(Y_pred, [num_syms, 1, 1, RB])  # [14,10,16,43]
                    
            #         tmp = tf.transpose(Y_pred_c, perm=[1,2,0,3])
            #         tmp = tmp[tf.newaxis, :, tf.newaxis, :, :, :]

            #         h_freq_csi_standardized_wesn[:, rx_ant_idx:rx_ant_idx+1, :, tx_ant_idx:tx_ant_idx+1, :, :] = np.asarray(tmp)
            #         hold = 1




            # h_freq_csi_standardized_wesn = tf.transpose(Y_pred_c, perm=[1,2,0,3])
            # h_freq_csi_standardized_wesn = h_freq_csi_standardized_wesn[tf.newaxis, :, tf.newaxis, :, :, :]
            # h_freq_csi_standardized_wesn = tf.convert_to_tensor(h_freq_csi_standardized_wesn)

            # pred_nmse_standardized_wesn = self.nmse(h_freq_csi_true[0,...], h_freq_csi_standardized_wesn)


            # hold = 1














            # Kalman filter baseline prediction
            # rc_predictor_kf = kalman_pred_freq_dmimo('MU_MIMO', self.rc_config,
            #                                          num_rx_ant=self.num_rx_ue,
            #                                          num_tx_ant=self.cfg.num_tx_ue_sel*2 + 4)
            # h_freq_csi_kalman = rc_predictor_kf.predict(h_freq_csi_history)
            # h_freq_csi_kalman = tf.transpose(h_freq_csi_kalman, perm=[0,1,2,3,4,6,5])
            # pred_nmse_kalman = self.nmse(h_freq_csi_true[0,...], h_freq_csi_kalman[0,...])
            
            # Compare with gradient descent GESN
            # self.rc_config.enable_window = True
            # rc_predictor = gesn_pred_freq_dmimo('MU_MIMO', self.rc_config, num_rx_ant = self.num_rx_ue, 
            #                                         num_tx_ant=self.cfg.num_tx_ue_sel*2 + 4, max_adjacency='all', method=self.cfg.graph_formulation, 
            #                                         num_neurons=16, edge_weighting_method='grad_descent') # edge_weighting_method: 'model_based', 'grad_descent'
            # h_freq_csi_grad_descent = rc_predictor.predict(h_freq_csi_history, h_freq_csi_true[0,...])
            # h_freq_csi_grad_descent = tf.transpose(h_freq_csi_grad_descent, perm=[0,1,3,2])
            # pred_nmse_wgesn_per_antenna_pair = self.nmse(np.squeeze(h_freq_csi_true[0,...]), h_freq_csi_grad_descent)
            # h_freq_csi_grad_descent = tf.transpose(h_freq_csi_grad_descent, perm=[0,1,3,2])

            h_freq_csi_outdated = np.squeeze(h_freq_csi_history).transpose([0,1,2,4,3])
            h_freq_csi_outdated = h_freq_csi_outdated[-1, ...]
            h_freq_csi_outdated = tf.transpose(h_freq_csi_outdated, perm=[0,1,3,2])
            pred_nmse_outdated = self.nmse(np.squeeze(h_freq_csi_true[0,...]), h_freq_csi_outdated)
            h_freq_csi_outdated = tf.transpose(h_freq_csi_outdated, perm=[0,1,3,2])
            
        else:
            # LMMSE channel estimation
            h_freq_csi, err_var_csi = lmmse_channel_estimation(dmimo_chans, self.rg_csi,
                                                               slot_idx=self.cfg.first_slot_idx - self.cfg.csi_delay,
                                                               cfo_sigma=self.cfo_sigma, sto_sigma=self.sto_sigma)
            _, rx_snr_db = dmimo_chans.load_channel(slot_idx=self.cfg.first_slot_idx - self.cfg.csi_delay, batch_size=self.batch_size)
        
        # [batch_size, num_rx, num_rxs_ant, num_tx, num_txs_ant, num_ofdm_sym, fft_size]
        # h_freq_csi_grad_descent = tf.transpose(h_freq_csi_grad_descent, perm=[0,1,3,2])
        # h_freq_csi_grad_descent = h_freq_csi_grad_descent[tf.newaxis, tf.newaxis, :, tf.newaxis, :, :, :]
        h_freq_csi_outdated = tf.transpose(h_freq_csi_outdated, perm=[0,1,3,2])
        h_freq_csi_outdated = h_freq_csi_outdated[tf.newaxis, tf.newaxis, :, tf.newaxis, :, :, :]
        h_freq_csi_outdated = tf.reshape(h_freq_csi_outdated, (1, self.num_rx_ue, -1, *h_freq_csi_outdated.shape[3:]))
        # h_freq_csi_outdated = tf.transpose(h_freq_csi_outdated, perm=[0, 2, 1, 3, 4, 5, 6])

        # [batch_size, num_rx_ue, num_ue_ant, num_tx, num_txs_ant, num_ofdm_sym, fft_size]
        h_freq_csi_vanilla = tf.reshape(h_freq_csi_vanilla, (1, self.num_rx_ue, -1, *h_freq_csi_vanilla.shape[3:]))
        # h_freq_csi_kalman = tf.reshape(h_freq_csi_kalman, (-1, self.num_rx_ue, 1, *h_freq_csi_kalman.shape[3:]))
        h_freq_csi_twomode_all = tf.reshape(h_freq_csi_twomode_all, (1, self.num_rx_ue, -1, *h_freq_csi_twomode_all.shape[3:]))
        h_freq_csi_twomode_per_link = tf.reshape(h_freq_csi_twomode_per_link, (1, self.num_rx_ue, -1, *h_freq_csi_twomode_per_link.shape[3:]))
        h_freq_csi_graph_twomode = tf.reshape(h_freq_csi_graph_twomode, (1, self.num_rx_ue, -1, *h_freq_csi_graph_twomode.shape[3:]))
        h_freq_csi_true = h_freq_csi_true[0:1,...]
        h_freq_csi_true = tf.reshape(h_freq_csi_true, (1, self.num_rx_ue, -1, *h_freq_csi_true.shape[3:]))


        
        SNR_range = np.arange(0, 22, 2)
        uncoded_bers = 0.5*np.ones((6, SNR_range.shape[0]))
        
        for snr_idx, snr in enumerate(SNR_range):
            
            rx_snr_db = snr

            # apply dMIMO channels to the resource grid in the frequency domain.
            h_freq, _ = dmimo_chans.load_channel(slot_idx=self.cfg.first_slot_idx,
                                                                batch_size=self.batch_size)
            
            for curr_method in range(uncoded_bers.shape[0]):
                
                if curr_method == 0:
                    h_freq_csi = h_freq
                    h_freq_csi = tf.reshape(h_freq_csi, (h_freq_csi.shape[0], self.num_rx_ue, -1, *h_freq_csi.shape[3:]))
                elif curr_method == 1:
                    h_freq_csi = h_freq_csi_outdated
                elif curr_method == 2:
                    h_freq_csi = h_freq_csi_vanilla
                elif curr_method == 3:
                    h_freq_csi = h_freq_csi_twomode_all
                elif curr_method == 4:
                    h_freq_csi = h_freq_csi_twomode_per_link
                elif curr_method == 5:
                    h_freq_csi = h_freq_csi_graph_twomode
                
                if curr_method != 0:
                    h_freq_csi = tf.repeat(h_freq_csi, repeats=12, axis=-1)
                    h_freq_csi = h_freq_csi[..., :512]

                # apply precoding to OFDM grids
                if self.cfg.precoding_method == "ZF":
                    # ue_indices = [[i] for i in range(self.num_rx_ue)]
                    ue_ranks = self.num_streams_per_tx / self.num_rx_ue
                    # x_precoded, h_eff = self.zf_precoder([x_rg, tf.transpose(h_freq, perm=[0, 2, 1, 3, 4, 5, 6]), ue_indices, ue_ranks])
                    x_precoded, h_eff = self.zf_precoder([x_rg, h_freq_csi, self.cfg.ue_indices, ue_ranks])
                else:
                    ValueError("unsupported precoding method for MASS")                
                
                debug = False
                if debug:
                    plt.figure()
                    debug_rx_ant = 3
                    debug_tx_ant = 1
                    debug_ofdm_sym = 10
                    plt.plot(np.real(h_freq[0,0,debug_rx_ant,0,debug_tx_ant,debug_ofdm_sym,:]))
                    plt.plot(np.real(h_freq_csi_history[-1,0,0,debug_rx_ant,0,debug_tx_ant,debug_ofdm_sym,:]))
                    plt.savefig('a')
                debug = False
                y = self.apply_channel([x_precoded, h_freq])
                no = np.power(10.0, rx_snr_db / (-10.0))
                y = self.awgn([y, no])

                # LS channel estimation with linear interpolation
                no = tf.reduce_mean(no)
                no = tf.cast(no, tf.float32)
                h_hat, err_var = self.ls_estimator([y, no])

                x_hat = np.zeros(x_rg.shape, dtype=np.complex64)
                x_hat = x_hat[..., :self.rg.num_effective_subcarriers]
                for rx_node in range(self.cfg.num_rx_ue_sel+1):
                    
                    if rx_node == 0:
                        rx_ant = tf.range(0,4)
                        stream_idx = tf.range(0,2*self.cfg.num_tx_streams)
                    else:
                        rx_ant = tf.range(4 + (rx_node-1)*2, 4 + rx_node*2)
                        start = 2*self.cfg.num_tx_streams + (rx_node-1)*self.cfg.num_tx_streams
                        end = 2*self.cfg.num_tx_streams + rx_node*self.cfg.num_tx_streams
                        stream_idx = tf.range(start,end)

                    curr_y = tf.gather(y, rx_ant, axis=2)
                    curr_y = tf.gather(curr_y, self.rg.effective_subcarrier_ind, axis=-1)
                    curr_y = tf.squeeze(curr_y)

                    curr_h = tf.gather(h_hat, rx_ant, axis=2)
                    curr_h = tf.squeeze(curr_h)
                    curr_h = tf.gather(curr_h, stream_idx, axis=2)

                    curr_x_hat = self.mmse_detect(curr_y, curr_h, snr_db=rx_snr_db)
                    curr_x_hat = curr_x_hat[:, np.newaxis, ...]
                    curr_x_hat = np.asarray(curr_x_hat)

                    x_hat[:,:,stream_idx,:,:] = curr_x_hat
                
                all_symbols = tf.range(self.rg.num_ofdm_symbols)
                pilot_symbols = self.rg._pilot_ofdm_symbol_indices

                # Get the set difference: symbols not used for pilots
                data_symbol_indices = tf.sets.difference(
                    tf.expand_dims(all_symbols, 0), tf.expand_dims(pilot_symbols, 0)
                ).values
                x_hat = x_hat[:,:,:,data_symbol_indices ,:]
                x_hat = tf.reshape(x_hat, (x_hat.shape[0], x_hat.shape[1], x_hat.shape[2], x_hat.shape[3] * x_hat.shape[4]))
                x_hat = tf.convert_to_tensor(x_hat)

                # Soft-output QAM demapper
                llr = self.demapper([x_hat, no])

                # Hard-decision bit error rate
                d_hard = tf.cast(llr > 0, tf.float32) # Shape: [nbatches, 1, number of streams, number of effective subcarriers * number of data OFDM symbols * QAM order]
                uncoded_bers[curr_method, snr_idx] = compute_ber(d, d_hard).numpy()

        plt.figure()
        plt.semilogy(SNR_range, uncoded_bers[0, :], label="Perfect Channel")
        plt.semilogy(SNR_range, uncoded_bers[1, :], label="Outdated Channel")
        plt.semilogy(SNR_range, uncoded_bers[2, :], label="WESN Channel")
        plt.semilogy(SNR_range, uncoded_bers[3, :], label="Twomode (All) WESN Channel")
        plt.semilogy(SNR_range, uncoded_bers[4, :], label="Twomode (Per-Link) WESN Channel")
        plt.semilogy(SNR_range, uncoded_bers[5, :], label="Twomode Graph WESN Channel")
        plt.grid()
        plt.legend()
        plt.savefig('bers_tmp')

        return [pred_nmse_outdated, pred_nmse_wesn, pred_nmse_wgesn_per_antenna_pair, pred_nmse_kalman], uncoded_bers, x_hat
    
    def mmse_detect(self, curr_y, curr_h, snr_db=None, noise_var=None, eps=1e-7):
        """
        curr_y: [B, N_r, S, K]          (received)
        curr_h: [B, N_r, N_t, S, K]     (channel estimate)
        snr_db: scalar SNR in dB for noise variance (optional)
        noise_var: scalar or [B, S, K] noise variance σ² (optional)
        eps: small diagonal loading for numerical stability
        returns: curr_x_hat [B, N_t, S, K]
        """

        B, Nr, S, K = curr_y.shape
        Nt = curr_h.shape[2]

        # Decide noise variance σ²
        if noise_var is None:
            if snr_db is None:
                # Fallback: a modest regularization if no SNR/noise provided
                sigma2 = tf.constant(1e-3, dtype=curr_h.dtype.real_dtype)
            else:
                sigma2 = tf.cast(10.0 ** (-snr_db / 10.0), curr_h.dtype.real_dtype)
        else:
            sigma2 = tf.cast(noise_var, curr_h.dtype.real_dtype)  # can be scalar or [B,S,K]

        # Reorder to [B, S, K, N_r, N_t] and [B, S, K, N_r, 1]
        H = tf.transpose(curr_h, perm=[0, 3, 4, 1, 2])                       # [B,S,K,Nr,Nt]
        y = tf.transpose(curr_y, perm=[0, 2, 3, 1])                           # [B,S,K,Nr]
        y = y[..., tf.newaxis]                                                # [B,S,K,Nr,1]

        # Compute H^H H  and  H^H y
        Hh = tf.linalg.adjoint(H)                                             # [B,S,K,Nt,Nr]
        G  = tf.matmul(Hh, H)                                                 # [B,S,K,Nt,Nt]
        Hy = tf.matmul(Hh, y)                                                 # [B,S,K,Nt,1]

        # Add σ² I (supports scalar σ² or per-[B,S,K] variance)
        I = tf.eye(Nt, dtype=H.dtype)[tf.newaxis, tf.newaxis, tf.newaxis, ...]    # [1,1,1,Nt,Nt]
        if tf.rank(sigma2) == 0:
            A = G + tf.cast(sigma2, G.dtype) * I
        else:
            # reshape σ² to [B,S,K,1,1] for broadcast
            sig = tf.reshape(tf.cast(sigma2, G.dtype), [B, S, K, 1, 1])
            A = G + sig * I
        # Small extra loading
        A = A + tf.cast(eps, A.dtype) * I

        # Solve (H^H H + σ² I) x = H^H y via Cholesky (Hermitian PD)
        L = tf.linalg.cholesky(A)
        X = tf.linalg.cholesky_solve(L, Hy)                                   # [B,S,K,Nt,1]

        # Back to [B, N_t, S, K]
        X = tf.squeeze(X, axis=-1)                                            # [B,S,K,Nt]
        curr_x_hat = tf.transpose(X, perm=[0, 3, 1, 2])                       # [B,Nt,S,K]
        return curr_x_hat

    def nmse(self, H_true, H_pred, standard=True):
        # Promote both inputs to the same backend first
        if isinstance(H_true, np.ndarray) and isinstance(H_pred, np.ndarray):
            backend = np
        else:                              # at least one is a tf.Tensor
            H_true = tf.convert_to_tensor(H_true)          # keep original dtype
            H_pred = tf.cast(H_pred, H_true.dtype)         # <-- safe cast
            backend = tf

        diff = backend.abs(H_true - H_pred) ** 2
        num  = backend.reduce_sum(diff) if backend is tf else backend.sum(diff)

        if standard:
            denom_term = backend.abs(H_true) ** 2
        else:
            denom_term = (backend.abs(H_true) + backend.abs(H_pred)) ** 2

        denom = backend.reduce_sum(denom_term) if backend is tf else backend.sum(denom_term)
        if backend is tf:
            nmse = backend.cast(num / denom, backend.float32)
        else:
            nmse = np.asarray(num / denom, dtype=backend.float64)

        return nmse

    def awgn(self, inputs):

        x, no = inputs

        # Create tensors of real-valued Gaussian noise for each complex dim.
        noise = complex_normal(tf.shape(x), dtype=x.dtype)

        # Add extra dimensions for broadcasting
        no = expand_to_rank(no, tf.rank(x), axis=-1)

        # Apply variance scaling
        noise *= tf.cast(tf.sqrt(no), noise.dtype)

        # Add noise to input
        y = x + noise

        return y


def sim_mu_mimo(cfg: SimConfig, rc_config:RCConfig):
    """
    Simulation of MU-MIMO scenarios using different settings

    :param cfg: simulation settings
    :return: [uncoded_ber, coded_ber], [goodbits, userbits, ratedbits]
    """
    cfg.return_estimated_channel = False

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
    [pred_nmse_outdated, pred_nmse_wesn, pred_nmse_wgesn_per_antenna_pair, pred_nmse_kalman], uncoded_bers, x_hat = mu_mimo(dmimo_chans, info_bits)


    return [pred_nmse_outdated, pred_nmse_wesn, pred_nmse_wgesn_per_antenna_pair, pred_nmse_kalman], uncoded_bers


def sim_mu_mimo_all(cfg: SimConfig, rc_config:RCConfig):
    """"
    Simulation of MU-MIMO scenario according to the frame structure
    """

    total_cycles = 0
    
    pred_nmse_pred_nmse_outdated = []
    pred_nmse_wesn = []
    pred_nmse_wgesn_per_antenna_pair = []
    pred_nmse_kalman = []
    
    uncoded_ber_outdated = []
    uncoded_ber_wesn = []
    uncoded_ber_wgesn = []
    uncoded_ber_kalman = []    

    for first_slot_idx in np.arange(cfg.start_slot_idx, cfg.total_slots, cfg.num_slots_p1 + cfg.num_slots_p2):

        print("first_slot_idx: ", first_slot_idx, "\n")

        total_cycles += 1
        cfg.first_slot_idx = first_slot_idx

        [curr_pred_nmse_outdated, curr_pred_nmse_wesn, curr_pred_nmse_wgesn_per_antenna_pair, curr_pred_nmse_kalman], curr_uncoded_bers = sim_mu_mimo(cfg, rc_config)

        pred_nmse_pred_nmse_outdated.append(curr_pred_nmse_outdated)
        pred_nmse_wesn.append(curr_pred_nmse_wesn)
        pred_nmse_wgesn_per_antenna_pair.append(curr_pred_nmse_wgesn_per_antenna_pair)
        pred_nmse_kalman.append(curr_pred_nmse_kalman)

        uncoded_ber_outdated.append(curr_uncoded_bers[0])
        uncoded_ber_wesn.append(curr_uncoded_bers[1])
        uncoded_ber_wgesn.append(curr_uncoded_bers[2])
        uncoded_ber_kalman.append(curr_uncoded_bers[3])

    return pred_nmse_pred_nmse_outdated, pred_nmse_wesn, pred_nmse_wgesn_per_antenna_pair, pred_nmse_kalman, uncoded_ber_outdated, uncoded_ber_wesn, uncoded_ber_wgesn, uncoded_ber_kalman