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
from dmimo.channel import LoadNs3Channel, dMIMOChannels
from dmimo.mimo import SVDPrecoder, SVDEqualizer

def sim_baseline(precoding_method="SVD", csi_delay=1, batch_size=8, num_bits_per_symbol=2, coderate=0.5):

    # dMIMO configuration
    num_ut = 1
    num_bs = 1
    num_ut_ant = 4
    num_bs_ant = 4
    first_slot_idx = 2
    ebno_db = 16.0  # temporary fixed for LMMSE equalization

    # The number of transmitted streams is equal to the number of UT antennas
    # in both uplink and downlink
    num_streams_per_tx = num_ut_ant

    # Create an RX-TX association matrix
    # rx_tx_association[i,j]=1 means that receiver i gets at least one stream
    # from transmitter j. Depending on the transmission direction (uplink or downlink),
    # the role of UT and BS can change. However, as we have only a single
    # transmitter and receiver, this does not matter:
    rx_tx_association = np.array([[1]])

    # Instantiate a StreamManagement object
    # This determines which data streams are determined for which receiver.
    # In this simple setup, this is fairly easy. However, it can get more involved
    # for simulations with many transmitters and receivers.
    sm = StreamManagement(rx_tx_association, num_streams_per_tx)

    rg = ResourceGrid(num_ofdm_symbols=14,
                      fft_size=512,
                      subcarrier_spacing=15e3,
                      num_tx=1,
                      num_streams_per_tx=num_streams_per_tx,
                      cyclic_prefix_length=64,
                      num_guard_carriers=[0,0],
                      dc_null=False,
                      pilot_pattern="kronecker",
                      pilot_ofdm_symbol_indices=[2, 11])

    # num_bits_per_symbol = 6 # 2-QPSK, 4-16QAM
    # coderate = 0.5 # Code rate
    num_codewords = num_bits_per_symbol # number of codewords per frame
    n = int(rg.num_data_symbols*num_bits_per_symbol/num_codewords) # Number of coded bits
    k = int(n*coderate)  # Number of information bits

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
    zf_precoder = ZFPrecoder(rg, sm, return_effective_channel=True)
    svd_precoder = SVDPrecoder(rg, sm, return_effective_channel=True)
    svd_equalizer = SVDEqualizer(rg, sm)

    # The LS channel estimator will provide channel estimates and error variances
    ls_est = LSChannelEstimator(rg, interpolation_type="lin")

    # The LMMSE equalizer will provide soft symbols together with noise variance estimates
    lmmse_equ = LMMSEEqualizer(rg, sm)

    # The demapper produces LLR for all coded bits
    demapper = Demapper("maxlog", "qam", num_bits_per_symbol)

    # The decoder provides hard-decisions on the information bits
    decoder = LDPC5GDecoder(encoder, hard_out=True)

    ns3_config = Ns3Config(data_folder="../ns3/channels", total_slots=11)
    dmimo_chans = dMIMOChannels(ns3_config, "Baseline", add_noise=True)
    ns3_channel = LoadNs3Channel(ns3_config)
    chest_noise = AWGN()

    # Compute the noise power for a given Eb/No value.
    # This takes not only the coderate but also the overheads related pilot
    # transmissions and nulled carriers
    no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)
    b = binary_source([batch_size, 1, rg.num_streams_per_tx, num_codewords, encoder.k])
    c = encoder(b)
    c = tf.reshape(c, [batch_size, 1, rg.num_streams_per_tx, num_codewords * encoder.n])
    d = intlvr(c)
    x = mapper(d)
    x_rg = rg_mapper(x)

    # Perfect channel estimation
    h_freq_csi, pl_tmp = ns3_channel("Baseline", slot_idx=first_slot_idx - csi_delay, batch_size=batch_size)
    # h_freq_csi = tf.expand_dims(h_freq_csi, 1)  # [batch_size, num_rx, num_rx_ant, num_tx_ant, num_ofdm_symbols, fft_size]
    # h_freq_csi = tf.expand_dims(h_freq_csi, 3)  # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
    # add some noise to channel estimation
    h_freq_csi = chest_noise([h_freq_csi, 5e-3])

    # used to simulate perfect CSI at the receiver
    if precoding_method == "ZF":
        x_rg, g = zf_precoder([x_rg, h_freq_csi])
    elif precoding_method == "SVD":
        x_rg, g = svd_precoder([x_rg, h_freq_csi])
    else:
        raise "unsupported precoding method"

    # apply dMIMO channels to the resource grid in the frequency domain.
    # y_old = dmimo_chans([x_rg, no])
    y = dmimo_chans([x_rg, first_slot_idx])

    # SVD equalization
    if precoding_method == "SVD":
        y = svd_equalizer([y, h_freq_csi])

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
    throughput = (1.0 - ber) * x.shape[-1] * x.shape[-2] * coderate * num_bits_per_symbol / slot_duration / 1e6  # Mbps

    return [uncoded_ber.numpy(), ber.numpy(), throughput], x_hat.numpy()

