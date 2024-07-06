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
from dmimo.channel import LoadNs3Channel, dMIMOChannels, lmmse_channel_estimation
from dmimo.mimo import ZFPrecoder


def sim_mu_mimo(precoding_method="BD", first_slot_idx=5, csi_delay=1, batch_size=3,
                num_bits_per_symbol=2, coderate=0.5, perfect_csi=False, ns3_folder="../ns3/channels"):

    # dMIMO configuration for MU-MIMO
    # To use sionna-compatible interface, regard TxSquad as one BS transmitter,
    num_bs = 1
    num_bs_ant = 24
    num_ue = 7
    num_ue_ant = 2

    # Use 4 UEs with BS (12 antennas)
    # A 4-antennas basestation is regarded as the combination of two 2-antenna UEs
    num_streams_per_tx = num_ue * num_ue_ant

    # Create an RX-TX association matrix
    # rx_tx_association[i,j]=1 means that receiver i gets at least one stream from transmitter j.
    rx_tx_association = np.ones((num_ue, num_bs))

    # Instantiate a StreamManagement object
    # This determines which data streams are determined for which receiver.
    sm = StreamManagement(rx_tx_association, num_streams_per_tx)

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

    num_codewords = num_bits_per_symbol//2 # number of codewords per frame
    n = int(rg.num_data_symbols*num_bits_per_symbol/num_codewords) # Number of coded bits
    k = int(n*coderate) # Number of information bits

    # Compute the noise power for a given Eb/No value.
    # This takes not only the coderate but also the overheads related pilot
    # transmissions and nulled carriers
    ebno_db = 10.0  # temporary fixed for LMMSE equalization
    no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)

    # The binary source will create batches of information bits
    binary_source = BinarySource()

    # The encoder maps information bits to coded bits
    encoder = LDPC5GEncoder(k, n)

    # LDPC interleaver
    intlvr = RowColumnInterleaver(3072, axis=-1)
    dintlvr = Deinterleaver(interleaver=intlvr)

    # The mapper maps blocks of information bits to constellation symbols
    mapper = Mapper("qam", num_bits_per_symbol)

    # The resource grid mapper maps symbols onto an OFDM resource grid
    rg_mapper = ResourceGridMapper(rg)

    # The zero forcing precoder precodes the transmitter stream towards the intended antennas
    # bd_precoder = BDPrecoder(rg, sm, return_effective_channel=True)
    zf_precoder = ZFPrecoder(rg, sm, return_effective_channel=True)

    # The LS channel estimator will provide channel estimates and error variances
    ls_est = LSChannelEstimator(rg, interpolation_type="lin")

    # The LMMSE equalizer will provide soft symbols together with noise variance estimates
    lmmse_equ = LMMSEEqualizer(rg, sm)

    # The demapper produces LLR for all coded bits
    demapper = Demapper("maxlog", "qam", num_bits_per_symbol)

    # The decoder provides hard-decisions on the information bits
    decoder = LDPC5GDecoder(encoder, hard_out=True)

    ns3_config = Ns3Config(data_folder=ns3_folder, total_slots=21)
    dmimo_chans = dMIMOChannels(ns3_config, "dMIMO", add_noise=True)
    chest_noise = AWGN()

    b = binary_source([batch_size, 1, rg.num_streams_per_tx, num_codewords, encoder.k])
    c = encoder(b)
    c = tf.reshape(c, [batch_size, 1, rg.num_streams_per_tx, num_codewords * encoder.n])
    d = intlvr(c)
    x = mapper(d)
    x_rg = rg_mapper(x)

    if perfect_csi:
        # Perfect channel estimation
        # [batch_size, num_rx, num_rxs_ant, num_tx, num_txs_ant, num_ofdm_sym, fft_size]
        h_freq_csi, pl_tmp = dmimo_chans.load_channel(slot_idx=first_slot_idx - csi_delay, batch_size=batch_size)
        # add some noise to simulate channel estimation errors
        h_freq_csi = chest_noise([h_freq_csi, 2e-3])
    else:
        # LMMSE channel estimation
        h_freq_csi, err_var_csi = lmmse_channel_estimation(dmimo_chans, rg_csi, slot_idx=first_slot_idx - csi_delay)

    # [batch_size, num_rx, num_rxs_ant, num_tx, num_txs_ant, num_ofdm_sym, fft_size]
    h_freq_csi = h_freq_csi[:, :, :num_streams_per_tx, :, :, :, :]

    # [batch_size, num_rx_ue, num_ue_ant, num_tx, num_txs_ant, num_ofdm_sym, fft_size]
    h_freq_csi = tf.reshape(h_freq_csi, (-1, num_ue, num_ue_ant, *h_freq_csi.shape[3:]))

    # add some noise to channel estimation
    h_freq_csi = chest_noise([h_freq_csi, 2.5e-3])

    # used to simulate perfect CSI at the receiver
    x_rg, g = zf_precoder([x_rg, h_freq_csi])

    # apply dMIMO channels to the resource grid in the frequency domain.
    y = dmimo_chans([x_rg, first_slot_idx])
    # Extract only relevant outputs
    y = y[:, :, :num_streams_per_tx, :, :]

    # make proper shape
    y = tf.reshape(y, (batch_size, num_ue, num_ue_ant, 14, -1))

    # LS channel estimation with linear interpolation
    h_hat, err_var = ls_est([y, no])

    x_hat, no_eff = lmmse_equ([y, h_hat, err_var, no])
    llr = demapper([x_hat, no_eff])

    x_hard = tf.cast(llr > 0, tf.float32)
    uncoded_ber = compute_ber(d, x_hard)

    llr = dintlvr(llr)
    llr = tf.reshape(llr, [batch_size, 1, rg.num_streams_per_tx, num_codewords, encoder.n])
    b_hat = decoder(llr)
    ber = compute_ber(b, b_hat)

    # throughput estimation
    slot_duration = 1e-3  # 1 ms sub-frame
    throughput = (1.0 - ber.numpy()) * x.shape[-1] * x.shape[-2] * coderate * num_bits_per_symbol / (2 * slot_duration) / 1e6  # Mbps

    return [uncoded_ber.numpy(), ber.numpy(), throughput], x_hat.numpy()

