import numpy as np
import tensorflow as tf

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
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
from dmimo.mimo import SVDPrecoder, SVDEqualizer, rankAdaptation, linkAdaptation
from dmimo.mimo import ZFPrecoder
from dmimo.utils import add_frequency_offset, add_timing_offset, cfo_val, sto_val


def sim_baseline(cfg: SimConfig):
    """
    Simulation of baseline scenarios using 4x4 MIMO channels

    :param cfg: simulation settings
    :return: [uncoded BER, LDPC BER, Goodput], demodulated QAM symbols (for debugging purpose)
    """

    # dMIMO configuration
    num_bs_ant = 4  # Tx squad BB
    num_ue_ant = 4  # Rx squad BB

    # CFO and STO settings
    sto_sigma = sto_val(cfg, cfg.sto_sigma)
    cfo_sigma = cfo_val(cfg, cfg.cfo_sigma)

    # The number of transmitted streams is equal to the number of UE antennas
    assert cfg.num_tx_streams <= num_ue_ant
    num_streams_per_tx = cfg.num_tx_streams

    # batch processing for all slots in phase 2
    batch_size = cfg.num_slots_p2

    # Create an RX-TX association matrix
    # rx_tx_association[i,j]=1 means that receiver i gets at least one stream from transmitter j.
    rx_tx_association = np.array([[1]])  # 1-Tx 1-RX for SU-MIMO

    # Instantiate a StreamManagement object
    # This determines which data streams are determined for which receiver.
    sm = StreamManagement(rx_tx_association, num_streams_per_tx)

    # Adjust guard subcarriers for channel estimation grid
    csi_effective_subcarriers = (cfg.fft_size // num_bs_ant) * num_bs_ant
    csi_guard_carriers_1 = (cfg.fft_size - csi_effective_subcarriers) // 2
    csi_guard_carriers_2 = (cfg.fft_size - csi_effective_subcarriers) - csi_guard_carriers_1

    # Resource grid for channel estimation
    rg_csi = ResourceGrid(num_ofdm_symbols=14,
                          fft_size=cfg.fft_size,
                          subcarrier_spacing=cfg.subcarrier_spacing,
                          num_tx=1,
                          num_streams_per_tx=num_bs_ant,
                          cyclic_prefix_length=cfg.cyclic_prefix_len,
                          num_guard_carriers=[csi_guard_carriers_1, csi_guard_carriers_2],
                          dc_null=False,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2, 11])

    # Adjust guard subcarriers for different number of streams
    effective_subcarriers = (csi_effective_subcarriers // num_streams_per_tx) * num_streams_per_tx
    guard_carriers_1 = (csi_effective_subcarriers - effective_subcarriers) // 2
    guard_carriers_2 = (csi_effective_subcarriers - effective_subcarriers) - guard_carriers_1
    guard_carriers_1 += csi_guard_carriers_1
    guard_carriers_2 += csi_guard_carriers_2

    # OFDM resource grid (RG) for normal transmission
    rg = ResourceGrid(num_ofdm_symbols=14,
                      fft_size=cfg.fft_size,
                      subcarrier_spacing=cfg.subcarrier_spacing,
                      num_tx=1,
                      num_streams_per_tx=num_streams_per_tx,
                      cyclic_prefix_length=64,
                      num_guard_carriers=[guard_carriers_1, guard_carriers_2],
                      dc_null=False,
                      pilot_pattern="kronecker",
                      pilot_ofdm_symbol_indices=[2, 11])

    # LDPC params
    num_codewords = cfg.modulation_order//2  # number of codewords per frame
    ldpc_n = int(rg.num_data_symbols*cfg.modulation_order/num_codewords)  # Number of coded bits
    ldpc_k = int(ldpc_n*cfg.code_rate)  # Number of information bits

    # The binary source will create batches of information bits
    binary_source = BinarySource()

    # The encoder maps information bits to coded bits
    encoder = LDPC5GEncoder(ldpc_k, ldpc_n)

    # LDPC interleaver
    intlvr = RowColumnInterleaver(3072, axis=-1)  # fixed design for current RG config
    dintlvr = Deinterleaver(interleaver=intlvr)

    # The mapper maps blocks of information bits to constellation symbols
    mapper = Mapper("qam", cfg.modulation_order)

    # The resource grid mapper maps symbols onto an OFDM resource grid
    rg_mapper = ResourceGridMapper(rg)

    # The zero forcing precoder
    zf_precoder = ZFPrecoder(rg, sm, return_effective_channel=True)

    # SVD-based precoder and equalizer
    svd_precoder = SVDPrecoder(rg, sm, return_effective_channel=True)
    svd_equalizer = SVDEqualizer(rg, sm)

    # The LS channel estimator will provide channel estimates and error variances
    ls_estimator = LSChannelEstimator(rg, interpolation_type="lin")

    # The LMMSE equalizer will provide soft symbols together with noise variance estimates
    lmmse_equ = LMMSEEqualizer(rg, sm)

    # The demapper produces LLR for all coded bits
    demapper = Demapper("maxlog", "qam", cfg.modulation_order)

    # The decoder provides hard-decisions on the information bits
    decoder = LDPC5GDecoder(encoder, hard_out=True)

    # dMIMO channels from ns-3 simulator
    ns3_config = Ns3Config(data_folder=cfg.ns3_folder, total_slots=cfg.total_slots)
    dmimo_chans = dMIMOChannels(ns3_config, "Baseline", add_noise=True)

    # Transmitter processing
    b = binary_source([batch_size, 1, rg.num_streams_per_tx, num_codewords, encoder.k])
    c = encoder(b)
    c = tf.reshape(c, [batch_size, 1, rg.num_streams_per_tx, num_codewords * encoder.n])
    d = intlvr(c)
    x = mapper(d)
    x_rg = rg_mapper(x)

    if cfg.perfect_csi is True:
        # Perfect channel estimation
        h_freq_csi, rx_snr_db = dmimo_chans.load_channel(slot_idx=cfg.first_slot_idx - cfg.csi_delay, batch_size=batch_size)
        # add some noise to simulate channel estimation errors
        chest_noise = AWGN()
        h_freq_csi = chest_noise([h_freq_csi, 2e-3])
    else:
        # LMMSE channel estimation
        h_freq_csi, err_var_csi = lmmse_channel_estimation(dmimo_chans, rg_csi,
                                                           slot_idx=cfg.first_slot_idx - cfg.csi_delay,
                                                           cfo_sigma=cfo_sigma, sto_sigma=sto_sigma)
        _, rx_snr_db = dmimo_chans.load_channel(slot_idx=cfg.first_slot_idx - cfg.csi_delay, batch_size=batch_size)

    if cfg.return_estimated_channel:
        return h_freq_csi, rx_snr_db

    # apply precoding to OFDM grids
    if cfg.precoding_method == "ZF":
        x_precoded, g = zf_precoder([x_rg, h_freq_csi])
    elif cfg.precoding_method == "SVD":
        x_precoded, g = svd_precoder([x_rg, h_freq_csi])
    else:
        ValueError("unsupported precoding method")

    # add CFO/STO to simulate synchronization errors
    if cfg.sto_sigma > 0:
        x_precoded = add_timing_offset(x_precoded, sto_sigma)
    if cfg.cfo_sigma > 0:
        x_precoded = add_frequency_offset(x_precoded, cfo_sigma)

    # apply dMIMO channels to the resource grid in the frequency domain.
    y = dmimo_chans([x_precoded, cfg.first_slot_idx])

    # SVD equalization
    if cfg.precoding_method == "SVD":
        y = svd_equalizer([y, h_freq_csi, num_streams_per_tx])

    # LS channel estimation with linear interpolation
    no = 0.1  # initial noise estimation (tunable param)
    h_hat, err_var = ls_estimator([y, no])

    # LMMSE equalization
    x_hat, no_eff = lmmse_equ([y, h_hat, err_var, no])

    # Soft-output QAM demapper
    llr = demapper([x_hat, no_eff])

    # Hard-decision for uncoded bits
    x_hard = tf.cast(llr > 0, tf.float32)
    uncoded_ber = compute_ber(d, x_hard).numpy()

    # LLR deinterleaver for LDPC decoding
    llr = dintlvr(llr)
    llr = tf.reshape(llr, [batch_size, 1, rg.num_streams_per_tx, num_codewords, encoder.n])

    # LDPC decoding and BER calculation
    b_hat = decoder(llr)
    ber = compute_ber(b, b_hat).numpy()
    bler = compute_bler(b, b_hat).numpy()

    # Goodput and throughput estimation
    num_bits_per_frame = ldpc_k * num_codewords * rg.num_streams_per_tx
    goodbits = (1.0 - ber) * num_bits_per_frame
    userbits = (1.0 - bler) * num_bits_per_frame

    if cfg.rank_adapt and cfg.link_adapt:
        do_rank_link_adaptation(cfg, h_freq_csi, rx_snr_db, cfg.first_slot_idx)

    return [uncoded_ber, ber], [goodbits, userbits], x_hat.numpy()


def do_rank_link_adaptation(cfg, h_est=None, rx_snr_db=None, start_slot_idx=None):

    assert cfg.start_slot_idx >= cfg.csi_delay

    if start_slot_idx == None:
        cfg.first_slot_idx = cfg.start_slot_idx
    else:
        cfg.first_slot_idx = start_slot_idx

    if np.any(h_est == None) or np.any(rx_snr_db == None):
        
        cfg.return_estimated_channel = True
        h_est, rx_snr_db = sim_baseline(cfg)
        cfg.return_estimated_channel = False

    network_config = NetworkConfig()

    # Rank adaptation test
    rank_adaptation = rankAdaptation(network_config.num_bs_ant, network_config.num_ue_ant, architecture='SU-MIMO',
                                        snrdb=rx_snr_db, fft_size=cfg.fft_size, precoder='SVD')

    rank_feedback_report = rank_adaptation(h_est, channel_type='dMIMO')

    if rank_adaptation.use_mmse_eesm_method:
        rank = rank_feedback_report[0]
        rate = rank_feedback_report[1]

        cfg.num_tx_streams = int(rank)
        
        print("\n", "rank (baseline) = ", rank, "\n")
        print("\n", "rate (baseline) = ", rate, "\n")

    else:
        rank = rank_feedback_report
        rate = []

        cfg.num_tx_streams = int(rank)

        print("\n", "rank (baseline) = ", rank, "\n")

    # Link adaptation test
    data_sym_position = np.arange(0, 14)
    link_adaptation = linkAdaptation(network_config.num_bs_ant, network_config.num_ue_ant, architecture='SU-MIMO',
                                        snrdb=rx_snr_db, nfft=cfg.fft_size, N_s=rank, data_sym_position=data_sym_position, lookup_table_size='long')
    
    mcs_feedback_report = link_adaptation(h_est, channel_type='dMIMO')

    if link_adaptation.use_mmse_eesm_method:
        qam_order_arr = mcs_feedback_report[0]
        code_rate_arr = mcs_feedback_report[1]

        cfg.modulation_order = int(np.min(qam_order_arr))
        cfg.code_rate = np.min(code_rate_arr)

        print("\n", "Bits per stream (baseline) = ", cfg.modulation_order, "\n")
        print("\n", "Code-rate per stream (baseline) = ", cfg.code_rate, "\n")
    else:
        qam_order_arr = mcs_feedback_report[0]
        code_rate_arr = []

        cfg.modulation_order = int(np.min(qam_order_arr))


        print("\n", "Bits per stream (SU-MIMO) = ", cfg.modulation_order, "\n")
    
    return rank, rate, qam_order_arr, code_rate_arr

def sim_baseline_all(cfg: SimConfig):
    """"
    Simulation of baseline transmission (BS-to-BS)
    """
    if cfg.rank_adapt and cfg.link_adapt:
        do_rank_link_adaptation(cfg)

    total_cycles = 0
    uncoded_ber, ldpc_ber, goodput, throughput = 0, 0, 0, 0
    for first_slot_idx in np.arange(cfg.start_slot_idx, cfg.total_slots, cfg.num_slots_p2):
        total_cycles += 1
        cfg.first_slot_idx = first_slot_idx
        bers, bits, x_hat = sim_baseline(cfg)
        uncoded_ber += bers[0]
        ldpc_ber += bers[1]
        goodput += bits[0]
        throughput += bits[1]

    slot_time = cfg.slot_duration  # default 1ms subframe/slot duration
    goodput = goodput / (total_cycles * slot_time * 1e6)  # Mbps
    throughput = throughput / (total_cycles * slot_time * 1e6)  # Mbps

    return [uncoded_ber/total_cycles, ldpc_ber/total_cycles, goodput, throughput]
