import numpy as np
import tensorflow as tf

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

from dmimo.config import Ns3Config, SimConfig
from dmimo.channel import dMIMOChannels, lmmse_channel_estimation
from dmimo.channel import standard_rc_pred_freq_mimo
from dmimo.mimo import SVDPrecoder, SVDEqualizer
from dmimo.utils import add_frequency_offset, add_timing_offset, cfo_val, sto_val


def sim_su_mimo_chanpred(cfg: SimConfig):
    """
    Simulation of SU-MIMO scenarios using different settings

    Effective channel models for phase 2 (P2) and phase 3 (P3) are used, phase 1 (P1) transmission
    is simulated separately and assumed to always provide enough data bandwidth for P2.

    TODO: add link/rank adaption, UE selection

    :param cfg: simulation settings
    :return: [uncoded BER, LDPC BER], [goodput, throughput], demodulated QAM symbols (for debugging purpose)
    """

    # dMIMO configuration
    num_bs_ant = 24  # total number of Tx squad antennas
    num_ue_ant = 8   # total number of Rx antennas for effective channel

    # Estimated EbNo
    ebno_db = 16.0  # temporary fixed for LMMSE equalization

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
    ns3_config = Ns3Config(data_folder=cfg.ns3_folder, total_slots=21)
    dmimo_chans = dMIMOChannels(ns3_config, "dMIMO-Forward", add_noise=True)
    chest_noise = AWGN()

    if cfg.csi_prediction:
        rc_predictor = standard_rc_pred_freq_mimo('SU_MIMO')

    # Compute the noise power for a given Eb/No value.
    # This takes not only the coderate but also the overheads related pilot
    # transmissions and nulled carriers
    no = ebnodb2no(ebno_db, cfg.modulation_order, cfg.code_rate, rg)

    # Transmitter processing
    b = binary_source([batch_size, 1, rg.num_streams_per_tx, num_codewords, encoder.k])
    c = encoder(b)
    c = tf.reshape(c, [batch_size, 1, rg.num_streams_per_tx, num_codewords * encoder.n])
    d = intlvr(c)
    x = mapper(d)
    x_rg = rg_mapper(x)

    if cfg.perfect_csi:
        # Perfect channel estimation
        h_freq_csi, rx_snr_db = dmimo_chans.load_channel(slot_idx=cfg.first_slot_idx - cfg.csi_delay, batch_size=batch_size)
        # add some noise to simulate channel estimation errors
        h_freq_csi = chest_noise([h_freq_csi, 2e-3])
    elif cfg.csi_prediction:
        # Get CSI history
        # TODO: optimize channel estimation and optimization procedures (currently very slow)
        h_freq_csi_history = rc_predictor.get_csi_history(cfg.first_slot_idx, cfg.csi_delay, rg_csi, dmimo_chans)
        # Do channel prediction
        chan_pred = rc_predictor.rc_siso_predict(h_freq_csi_history)
        h_freq_csi = chan_pred
    else:
        # LMMSE channel estimation
        h_freq_csi, err_var_csi = lmmse_channel_estimation(dmimo_chans, rg_csi,
                                                           slot_idx=cfg.first_slot_idx - cfg.csi_delay,
                                                           cfo_sigma=cfo_sigma, sto_sigma=sto_sigma)

    # ZF assume num_rx_ant = num_stream_per_tx
    if cfg.precoding_method == "ZF":
        h_freq_csi = h_freq_csi[:, :, :num_streams_per_tx]

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
        y = svd_equalizer([y, h_freq_csi])

    # Rank adaptation support (extract only relevant streams)
    # [batch_size, 1, num_streams_per_tx, num_ofdm_sym, fft_size]
    y = y[:, :, :num_streams_per_tx, :, :]

    # LS channel estimation with linear interpolation
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

    return [uncoded_ber, ber], [goodbits, userbits], x_hat.numpy()


def sim_su_mimo_chanpred_all(cfg: SimConfig):
    """"
    Simulation of SU-MIMO transmission phases according to the frame structure
    """

    total_cycles = 0
    uncoded_ber, ldpc_ber, goodput, throughput = 0, 0, 0, 0
    for first_slot_idx in np.arange(cfg.start_slot_idx, cfg.total_slots, cfg.num_slots_p1 + cfg.num_slots_p2):
        total_cycles += 1
        cfg.first_slot_idx = first_slot_idx
        bers, bits, x_hat = sim_su_mimo_chanpred(cfg)
        uncoded_ber += bers[0]
        ldpc_ber += bers[1]
        goodput += bits[0]
        throughput += bits[1]

    slot_time = cfg.slot_duration  # default 1ms subframe/slot duration
    overhead = cfg.num_slots_p2/(cfg.num_slots_p1 + cfg.num_slots_p2)
    goodput = goodput / (total_cycles * slot_time * 1e6) * overhead  # Mbps
    throughput = throughput / (total_cycles * slot_time * 1e6) * overhead  # Mbps

    return [uncoded_ber/total_cycles, ldpc_ber/total_cycles, goodput, throughput]
