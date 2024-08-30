import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.ofdm import ZFPrecoder
from sionna.mimo import StreamManagement
from sionna.channel import AWGN

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.fec.interleaving import RowColumnInterleaver, Deinterleaver

from sionna.mapping import Mapper, Demapper
from sionna.utils import BinarySource, ebnodb2no
from sionna.utils.metrics import compute_ber, compute_bler

from dmimo.config import Ns3Config, SimConfig, NetworkConfig
from dmimo.channel import dMIMOChannels, lmmse_channel_estimation
from dmimo.mimo import BDPrecoder, BDEqualizer, ZFPrecoder, rankAdaptation, linkAdaptation
from dmimo.mimo import update_node_selection
from dmimo.mu_mimo import MU_MIMO
from dmimo.utils import add_frequency_offset, add_timing_offset, cfo_val, sto_val

class MU_MIMO(Model):

    def __init__(self, cfg: SimConfig, dmimo_chans: dMIMOChannels, **kwargs):
        """
        Create Baseline simulation object

        :param cfg: simulation settings
        """
        super().__init__(kwargs)

        self.cfg = cfg
        self.dmimo_chans = dmimo_chans

        # dMIMO configuration for MU-MIMO
        # To use sionna-compatible interface, regard TxSquad as one BS transmitter
        self.num_txs_ant = 2 * cfg.num_tx_ue_sel + 4  # total number of Tx squad antennas
        self.num_ue_ant = dmimo_chans.ns3_config.num_ue_ant
        if cfg.ue_indices is None:
            self.num_ue = cfg.num_tx_ue_sel + 2 # counting the BS as 2 UEs
            self.num_rxs_ant = num_ue * num_ue_ant
        else:
            self.num_rxs_ant = np.sum([len(val) for val in cfg.ue_indices])
            self.num_ue = self.num_rxs_ant // self.num_ue_ant
            if cfg.ue_ranks is None:
                # by default no rank adaptation
                cfg.ue_ranks = self.num_ue_ant

        # CFO and STO settings
        self.sto_sigma = sto_val(cfg, cfg.sto_sigma)
        self.cfo_sigma = cfo_val(cfg, cfg.cfo_sigma)

        # Use 4 UEs with BS (12 antennas)
        # A 4-antennas basestation is regarded as the combination of two 2-antenna UEs
        self.num_streams_per_tx = self.num_ue * cfg.ue_ranks  # num_ue * num_ue_ant

        # batch processing for all slots in phase 2
        self.batch_size = cfg.num_slots_p2

        # Create an RX-TX association matrix
        # rx_tx_association[i,j]=1 means that receiver i gets at least one stream from transmitter j.
        rx_tx_association = np.ones((self.num_ue, 1))

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

        # LDPC params
        self.num_codewords = cfg.modulation_order//2  # number of codewords per frame
        cfg.ldpc_n = int(self.rg.num_data_symbols*cfg.modulation_order/self.num_codewords)  # Number of coded bits
        cfg.ldpc_k = int(cfg.ldpc_n*cfg.code_rate)  # Number of information bits


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

    def call(self, info_bits):
        b = tf.reshape(info_bits, [self.batch_size, 1, self.rg.num_streams_per_tx,
                                   self.num_codewords, self.encoder.k])
        c = self.encoder(b)
        c = tf.reshape(c, [self.batch_size, 1, self.rg.num_streams_per_tx, self.num_codewords * self.encoder.n])
        # Interleaving for coded bits
        d = self.intlvr(c)
        # QAM mapping for the OFDM grid
        x = self.mapper(d)
        x_rg = self.rg_mapper(x)

        if self.cfg.perfect_csi:
            # Perfect channel estimation
            h_freq_csi, rx_snr_db = self.dmimo_chans.load_channel(slot_idx=self.cfg.first_slot_idx - self.cfg.csi_delay, batch_size=self.batch_size)
            # add some noise to simulate channel estimation errors
            chest_noise = AWGN()
            h_freq_csi = chest_noise([h_freq_csi, 2e-3])
        else:
            # LMMSE channel estimation
            h_freq_csi, err_var_csi = lmmse_channel_estimation(self.dmimo_chans, self.rg_csi,
                                                               slot_idx=self.cfg.first_slot_idx - self.cfg.csi_delay,
                                                               cfo_sigma=self.cfo_sigma, sto_sigma=self.sto_sigma)
            _, rx_snr_db = self.dmimo_chans.load_channel(slot_idx=self.cfg.first_slot_idx - self.cfg.csi_delay, batch_size=self.batch_size)

        if self.cfg.return_estimated_channel:
            return h_freq_csi, rx_snr_db

        shape_tmp = h_freq_csi.shape

        # [batch_size, num_rx, num_rxs_ant, num_tx, num_txs_ant, num_ofdm_sym, fft_size]
        h_freq_csi = h_freq_csi[:, :, :self.num_rxs_ant, :, :, :, :]

        # [batch_size, num_rx_ue, num_ue_ant, num_tx, num_txs_ant, num_ofdm_sym, fft_size]
        h_freq_csi = tf.reshape(h_freq_csi, (-1, self.num_ue, self.num_ue_ant, *h_freq_csi.shape[3:]))

        # apply precoding to OFDM grids
        if self.cfg.precoding_method == "ZF":
            x_precoded, g = self.zf_precoder([x_rg, h_freq_csi])
        elif self.cfg.precoding_method == "BD":
            x_precoded, g = self.bd_precoder([x_rg, h_freq_csi, self.cfg.ue_indices, self.cfg.ue_ranks])
        else:
            ValueError("unsupported precoding method")

        # add CFO/STO to simulate synchronization errors
        if self.cfg.sto_sigma > 0:
            x_precoded = add_timing_offset(x_precoded, self.sto_sigma)
        if self.cfg.cfo_sigma > 0:
            x_precoded = add_frequency_offset(x_precoded, self.cfo_sigma)

        # apply dMIMO channels to the resource grid in the frequency domain.
        y = self.dmimo_chans([x_precoded, self.cfg.first_slot_idx])

        # make proper shape
        y = y[:, :, :self.num_rxs_ant, :, :]
        y = tf.reshape(y, (self.batch_size, self.num_ue, self.num_ue_ant, 14, -1))

        if self.cfg.precoding_method == "BD":
            y = self.bd_equalizer([y, h_freq_csi, self.cfg.ue_indices, self.cfg.ue_ranks])

        # LS channel estimation with linear interpolation
        no = 0.1  # initial noise estimation (tunable param)
        h_hat, err_var = self.ls_estimator([y, no])

        # LMMSE equalization
        x_hat, no_eff = self.lmmse_equ([y, h_hat, err_var, no])

        # Soft-output QAM demapper
        llr = self.demapper([x_hat, no_eff])

        # Hard-decision for uncoded bits
        x_hard = tf.cast(llr > 0, tf.float32)
        uncoded_ber = compute_ber(d, x_hard).numpy()

        # LLR deinterleaver for LDPC decoding
        llr = self.dintlvr(llr)
        llr = tf.reshape(llr, [self.batch_size, 1, self.rg.num_streams_per_tx, self.num_codewords, self.encoder.n])

        # LDPC decoding and BER calculation
        dec_bits = self.decoder(llr)

        if self.cfg.rank_adapt and self.cfg.link_adapt:
            h_freq_csi_reshaped = tf.reshape(h_freq_csi, shape_tmp)
            do_rank_link_adaptation(self.cfg, self.dmimo_chans, h_freq_csi_reshaped, rx_snr_db, self.cfg.first_slot_idx)

        return dec_bits, uncoded_ber, x_hat


def mu_mimo_transmission(cfg: SimConfig, dmimo_chans: dMIMOChannels):
    """
    Signal processing for one MU-MIMO transmission cycle (P2)

    :param cfg: simulation settings
    :param dmimo_chans: dMIMO channels
    :return: [uncoded BER, LDPC BER], [goodput, throughput], demodulated QAM symbols (for debugging purpose)
    """

    # The binary source will create batches of information bits
    binary_source = BinarySource()
    # Transmitter processing
    #info_bits = binary_source([batch_size, 1, rg.num_streams_per_tx, num_codewords, encoder.k])

    mu_mimo = MU_MIMO(cfg, dmimo_chans)
    num_bits_per_frame = cfg.ldpc_k * mu_mimo.num_codewords * mu_mimo.rg.num_streams_per_tx
    info_bits = binary_source([cfg.num_slots_p2, num_bits_per_frame])

    # Initial rank and link adaptation
    if cfg.rank_adapt and cfg.link_adapt and cfg.first_slot_idx == cfg.start_slot_idx:
        do_rank_link_adaptation(cfg, dmimo_chans=dmimo_chans, mu_mimo=mu_mimo, info_bits=info_bits)

    # Baseline transmission
    dec_bits, uncoded_ber, x_hat = mu_mimo(info_bits)

    info_bits = tf.reshape(info_bits, dec_bits.shape)

    ber = compute_ber(dec_bits, info_bits).numpy()
    bler = compute_bler(dec_bits, info_bits).numpy()

    print (ber, bler)

    # Goodput and throughput estimation

    goodbits = (1.0 - ber) * num_bits_per_frame
    userbits = (1.0 - bler) * num_bits_per_frame

    return [uncoded_ber, ber], [goodbits, userbits], x_hat.numpy()


def sim_mu_mimo(cfg: SimConfig):
    """
    Simulation of MU-MIMO scenarios using different settings

    :param cfg: simulation settings
    :return: [uncoded BER, LDPC BER], [goodput, throughput], demodulated QAM symbols (for debugging purpose)
    """

    # dMIMO channels from ns-3 simulator
    ns3cfg = Ns3Config(data_folder=cfg.ns3_folder, total_slots=cfg.total_slots)
    dmimo_chans = dMIMOChannels(ns3cfg, "dMIMO", add_noise=True)

    # UE selection
    if cfg.enable_ue_selection:
        tx_ue_mask, rx_ue_mask = update_node_selection(cfg)
        ns3cfg.update_ue_mask(tx_ue_mask, rx_ue_mask)

    # if cfg.rank_adapt and cfg.link_adapt:
    #     do_rank_link_adaptation(cfg, dmimo_chans)

    # SU-MIMO transmission
    return mu_mimo_transmission(cfg, dmimo_chans)



def do_rank_link_adaptation(cfg, dmimo_chans, h_est=None, rx_snr_db=None, start_slot_idx=None, mu_mimo=None, info_bits=None):

    assert cfg.start_slot_idx >= cfg.csi_delay

    if start_slot_idx == None:
        cfg.first_slot_idx = cfg.start_slot_idx
    else:
        cfg.first_slot_idx = start_slot_idx

    if np.any(h_est == None) or np.any(rx_snr_db == None):
        
        cfg.return_estimated_channel = True
        h_est, rx_snr_db = mu_mimo(info_bits)
        cfg.return_estimated_channel = False

    # Rank adaptation
    rank_adaptation = rankAdaptation(dmimo_chans.ns3_config.num_bs_ant, dmimo_chans.ns3_config.num_ue_ant, architecture='MU-MIMO',
                                        snrdb=rx_snr_db, fft_size=cfg.fft_size, precoder='BD', ue_indices=cfg.ue_indices)

    rank_feedback_report = rank_adaptation(h_est, channel_type='dMIMO')

    if rank_adaptation.use_mmse_eesm_method:
        rank = rank_feedback_report[0]
        rate = rank_feedback_report[1]

        cfg.num_tx_streams = int(rank)*(cfg.num_rx_ue_sel+1)
        cfg.ue_ranks = rank
        
        print("\n", "rank per user (MU-MIMO) = ", rank, "\n")
        print("\n", "rate per user (MU-MIMO) = ", rate, "\n")

    else:
        rank = rank_feedback_report
        rate = []

        cfg.num_tx_streams = int(rank)

        print("\n", "rank per user (MU-MIMO) = ", rank, "\n")

    # Link adaptation
    data_sym_position = np.arange(0, 14)
    link_adaptation = linkAdaptation(dmimo_chans.ns3_config.num_bs_ant, dmimo_chans.ns3_config.num_ue_ant, architecture='MU-MIMO',
                                        snrdb=rx_snr_db, nfft=cfg.fft_size, N_s=rank, data_sym_position=data_sym_position, lookup_table_size='short')
    
    mcs_feedback_report = link_adaptation(h_est, channel_type='dMIMO')

    if link_adaptation.use_mmse_eesm_method:
        qam_order_arr = mcs_feedback_report[0]
        code_rate_arr = mcs_feedback_report[1]

        # Majority vote for MCS selection for now
        values, counts = np.unique(qam_order_arr, return_counts=True)
        most_frequent_value = values[np.argmax(counts)]
        cfg.modulation_order = int(most_frequent_value)

        values, counts = np.unique(code_rate_arr, return_counts=True)
        most_frequent_value = values[np.argmax(counts)]
        cfg.code_rate = most_frequent_value

        print("\n", "Bits per stream per user (MU-MIMO) = ", cfg.modulation_order, "\n")
        print("\n", "Code-rate per stream per user (MU-MIMO) = ", cfg.code_rate, "\n")
    else:
        qam_order_arr = mcs_feedback_report[0]
        code_rate_arr = []

        cfg.modulation_order = int(np.min(qam_order_arr))

        print("\n", "Bits per stream per user (MU-MIMO) = ", cfg.modulation_order, "\n")
    
    return rank, rate, qam_order_arr, code_rate_arr


def sim_mu_mimo_all(cfg: SimConfig):
    """"
    Simulation of SU-MIMO transmission phases according to the frame structure
    """

    total_cycles = 0
    uncoded_ber, ldpc_ber, goodput, throughput = 0, 0, 0, 0
    for first_slot_idx in np.arange(cfg.start_slot_idx, cfg.total_slots, cfg.num_slots_p1 + cfg.num_slots_p2):
        total_cycles += 1
        cfg.first_slot_idx = first_slot_idx
        bers, bits, x_hat = sim_mu_mimo(cfg)
        uncoded_ber += bers[0]
        ldpc_ber += bers[1]
        goodput += bits[0]
        throughput += bits[1]

    slot_time = cfg.slot_duration  # default 1ms subframe/slot duration
    overhead = cfg.num_slots_p2/(cfg.num_slots_p1 + cfg.num_slots_p2)
    goodput = goodput / (total_cycles * slot_time * 1e6) * overhead  # Mbps
    throughput = throughput / (total_cycles * slot_time * 1e6) * overhead  # Mbps

    return [uncoded_ber/total_cycles, ldpc_ber/total_cycles, goodput, throughput]
