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
from sionna.utils.metrics import compute_ber

from dmimo.config import Ns3Config
from dmimo.channel import dMIMOChannels, lmmse_channel_estimation
from dmimo.mimo import SVDPrecoder, SVDEqualizer


def sim_su_mimo(precoding_method="SVD", first_slot_idx=3, csi_delay=1, batch_size=3,
                num_bits_per_symbol=2, coderate=0.5, perfect_csi=False, ns3_folder="../ns3/channels"):
    """
    Simulation of SU-MIMO scenarios using different settings

    Effective channel models for phase 2 (P2) and phase 3 (P3) are used, phase 1 (P1) transmission
    is simulated separately and assumed to always provide enough data bandwidth for P2.

    TODO: add link/rank adaption, UE selection

    :param precoding_method: SVD or ZF
    :param first_slot_idx: first subframe/slot for data transmission
    :param csi_delay: CSI estimation feedback delay (in number of subframes/slots)
    :param batch_size: number of frame/slots in each phase 2 cycle
    :param num_bits_per_symbol: modulation order
    :param coderate: LDPC code rate
    :param perfect_csi: Use perfect CSI for precoding/equalization for testing purpose
    :param ns3_folder: folder for ns-3 channel data
    :return: [uncoded BER, LDPC BER, Goodput], demodulated QAM symbols (for debugging purpose)
    """

    # dMIMO configuration
    num_bs_ant = 24  # total number of Tx squad antennas
    num_ue_ant = 8   # total number of Rx atennas for effective channel

    # Estimated EbNo
    ebno_db = 16.0  # temporary fixed for LMMSE equalization

    # The number of transmitted streams is equal to the number of UE antennas
    num_streams_per_tx = num_ue_ant

    # Create an RX-TX association matrix
    # rx_tx_association[i,j]=1 means that receiver i gets at least one stream from transmitter j.
    rx_tx_association = np.array([[1]])  # 1-Tx 1-RX for SU-MIMO

    # Instantiate a StreamManagement object
    # This determines which data streams are determined for which receiver.
    sm = StreamManagement(rx_tx_association, num_streams_per_tx)

    # OFDM resource grid (RG) for normal transmission
    rg = ResourceGrid(num_ofdm_symbols=14,
                      fft_size=512,
                      subcarrier_spacing=15e3,
                      num_tx=1,
                      num_streams_per_tx=num_streams_per_tx,
                      cyclic_prefix_length=64,
                      num_guard_carriers=[4, 4],
                      dc_null=False,
                      pilot_pattern="kronecker",
                      pilot_ofdm_symbol_indices=[2, 11])

    # Resource grid for dMIMO channel estimation
    rg_csi = ResourceGrid(num_ofdm_symbols=14,
                          fft_size=512,
                          subcarrier_spacing=15e3,
                          num_tx=1,
                          num_streams_per_tx=24,
                          cyclic_prefix_length=64,
                          num_guard_carriers=[4, 4],
                          dc_null=False,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2, 11])

    # LPDC params
    num_codewords = num_bits_per_symbol//2  # number of codewords per frame
    n = int(rg.num_data_symbols*num_bits_per_symbol/num_codewords)  # Number of coded bits
    k = int(n*coderate)  # Number of information bits

    # The binary source will create batches of information bits
    binary_source = BinarySource()

    # The encoder maps information bits to coded bits
    encoder = LDPC5GEncoder(k, n)

    # LDPC interleaver
    intlvr = RowColumnInterleaver(3072, axis=-1)  # fixed design for current RG config
    dintlvr = Deinterleaver(interleaver=intlvr)

    # The mapper maps blocks of information bits to constellation symbols
    mapper = Mapper("qam", num_bits_per_symbol)

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
    demapper = Demapper("maxlog", "qam", num_bits_per_symbol)

    # The decoder provides hard-decisions on the information bits
    decoder = LDPC5GDecoder(encoder, hard_out=True)

    # dMIMO channels from ns-3 simulator
    ns3_config = Ns3Config(data_folder=ns3_folder, total_slots=21)
    dmimo_chans = dMIMOChannels(ns3_config, "dMIMO-Forward", add_noise=True)
    chest_noise = AWGN()

    # Compute the noise power for a given Eb/No value.
    # This takes not only the coderate but also the overheads related pilot
    # transmissions and nulled carriers
    no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)

    # Transmitter processing
    b = binary_source([batch_size, 1, rg.num_streams_per_tx, num_codewords, encoder.k])
    c = encoder(b)
    c = tf.reshape(c, [batch_size, 1, rg.num_streams_per_tx, num_codewords * encoder.n])
    d = intlvr(c)
    x = mapper(d)
    x_rg = rg_mapper(x)

    if perfect_csi:
        # Perfect channel estimation
        h_freq_csi, pl_tmp = dmimo_chans.load_channel(slot_idx=first_slot_idx - csi_delay, batch_size=batch_size)
        # add some noise to simulate channel estimation errors
        h_freq_csi = chest_noise([h_freq_csi, 2e-3])
    else:
        # LMMSE channel estimation
        h_freq_csi, err_var_csi = lmmse_channel_estimation(dmimo_chans, rg_csi, slot_idx=first_slot_idx - csi_delay)

    # apply precoding to OFDM grids
    if precoding_method == "ZF":
        x_rg, g = zf_precoder([x_rg, h_freq_csi])
    elif precoding_method == "SVD":
        x_rg, g = svd_precoder([x_rg, h_freq_csi])
    else:
        ValueError("unsupported precoding method")

    # apply dMIMO channels to the resource grid in the frequency domain.
    y = dmimo_chans([x_rg, first_slot_idx])

    # SVD equalization
    if precoding_method == "SVD":
        y = svd_equalizer([y, h_freq_csi])

    # LS channel estimation with linear interpolation
    h_hat, err_var = ls_estimator([y, no])

    # LMMSE equalization
    x_hat, no_eff = lmmse_equ([y, h_hat, err_var, no])

    # Soft-output QAM demapper
    llr = demapper([x_hat, no_eff])

    # Hard-decision for uncoded bits
    x_hard = tf.cast(llr > 0, tf.float32)
    uncoded_ber = compute_ber(d, x_hard)

    # LLR deinterleaver for LDPC decoding
    llr = dintlvr(llr)
    llr = tf.reshape(llr, [batch_size, 1, rg.num_streams_per_tx, num_codewords, encoder.n])

    # LDPC decoding and BER calculation
    b_hat = decoder(llr)
    ber = compute_ber(b, b_hat)

    # Goodput and throughput estimation
    slot_duration = 1e-3  # 1 ms sub-frame
    goodput = (1.0 - ber) * x.shape[-1] * x.shape[-2] * coderate * num_bits_per_symbol / (2 * slot_duration) / 1e6  # Mbps

    return [uncoded_ber.numpy(), ber.numpy(), goodput], x_hat.numpy()


def sim_su_mimo_all(precoding_method="SVD", total_slots=20, num_slots_p1=1, num_slots_p2=3, start_slot_idx=5, csi_delay=1,
                    num_bits_per_symbol=2, coderate=0.5, perfect_csi=False):
    """"
    Simulation of SU-MIMO transmission phases according to the frame structure
    """

    total_cycles = 0
    uncoded_ber, ber, goodput = 0, 0, 0
    for first_slot_idx in np.arange(start_slot_idx, total_slots, num_slots_p1+num_slots_p2):
        total_cycles += 1
        results, x_hat = sim_su_mimo(precoding_method=precoding_method, first_slot_idx=first_slot_idx, batch_size=num_slots_p2, csi_delay=csi_delay,
                                     num_bits_per_symbol=num_bits_per_symbol, coderate=coderate, perfect_csi=perfect_csi)
        uncoded_ber += results[0]
        ber += results[1]
        goodput += results[2]

        return [uncoded_ber/total_cycles, ber/total_cycles, goodput/total_cycles]
