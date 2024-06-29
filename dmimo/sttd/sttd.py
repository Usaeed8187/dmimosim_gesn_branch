"""
Space-Time Transmit Diversity (STTD) for dMIMO scenarios
"""

import tensorflow as tf

from sionna.utils import BinarySource
from sionna.mapping import Mapper, Demapper
from sionna.ofdm import ResourceGrid, ResourceGridMapper

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.fec.interleaving import RowColumnInterleaver, Deinterleaver

from sionna.channel import AWGN
from sionna.utils import ebnodb2no
from sionna.utils import flatten_dims
from sionna.utils.metrics import compute_ber

from dmimo.channel import LoadNs3Channel
from dmimo.config import Ns3Config

from .stbc import stbc_encode, stbc_decode


def extract_sttd_channel(slot_idx, batch_size, num_tx_ue=4, num_rx_ue=4):

    assert num_tx_ue in [4, 0], "Invalid number of TxSquad UE"

    ns3_config = Ns3Config(total_slots=11)
    ns3_channel = LoadNs3Channel(ns3_config)
    chest_noise = AWGN()

    num_tx = num_tx_ue + 2  # BS counts as 2 Tx node
    num_rx = num_rx_ue + 2  # BS count as 2 Tx node
    max_num_tx = 12
    max_num_rx = 12
    num_ofdm_sym = 14
    fft_size = 512

    # Perfect channel estimation or LMMSE channel estimation
    # h_freq_ns3 has shape [batch_size, 1, num_rxs_ant, 1, num_txs_ant, num_ofdm_sym, fft_size]
    h_freq_ns3, snr_ns3 = ns3_channel("dMIMO", slot_idx=slot_idx, batch_size=batch_size)
    h_freq_ns3 = tf.reshape(h_freq_ns3, (batch_size, max_num_rx, 2, max_num_tx, 2, num_ofdm_sym, fft_size))

    # simulate channel estimation errors
    h_freq_ns3 = chest_noise([h_freq_ns3, 5e-3])

    # extract channels for STTD
    data_symbol_indices = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]
    tx_indices = tf.repeat(tf.range(0, num_tx), max_num_tx // num_tx)
    h_selected = [h_freq_ns3[:, :, :, tx_indices[k], :, data_symbol_indices[k]:data_symbol_indices[k]+1, :] for k in range(12)]

    # h_freq_all has shape [batch_size, num_rx, num_rxue_ant, num_txue_ant, num_ofm_sym, fft_size]
    h_freq_all = tf.concat(h_selected, axis=-2)  # assemble all symbols as for only one transmitting node

    # select Rx UE
    h_freq_all = h_freq_all[:, 0:num_rx]  # [batch_size, num_rx, num_ue_ant, num_ue_ant, num_ofm_sym, fft_size]

    return h_freq_all


def sim_sttd_ofdm(num_bits_per_symbol=2, coderate=0.5, batch_size=8, predetect_combining = False):

    rg = ResourceGrid(num_ofdm_symbols=14,
                      fft_size=512,
                      subcarrier_spacing=15e3,
                      num_tx=1,
                      num_streams_per_tx=1,
                      cyclic_prefix_length=64,
                      num_guard_carriers=[0, 0],
                      dc_null=False,
                      pilot_pattern="kronecker",
                      pilot_ofdm_symbol_indices=[2, 11])

    # 2-QPSK, 4-16QAM
    num_codewords = num_bits_per_symbol  # number of codewords per frame
    n = int(rg.num_data_symbols * num_bits_per_symbol / num_codewords)  # Number of coded bits
    k = int(n * coderate)  # Number of information bits

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

    # The LS channel estimator will provide channel estimates and error variances
    # ls_est = LSChannelEstimator(rg, interpolation_type="lin")

    # The LMMSE equalizer will provide soft symbols together with noise variance estimates
    # lmmse_equ = LMMSEEqualizer(rg, sm)

    # The demapper produces LLR for all coded bits
    demapper = Demapper("maxlog", "qam", num_bits_per_symbol)

    # The decoder provides hard-decisions on the information bits
    decoder = LDPC5GDecoder(encoder, hard_out=True)

    ebno_db = 10.0
    nvar = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)

    # Source and LDPC encoder
    b = binary_source([batch_size, 1, rg.num_streams_per_tx, num_codewords, encoder.k])
    c = encoder(b)
    c = tf.reshape(c, [batch_size, 1, rg.num_streams_per_tx, num_codewords*encoder.n])

    # Interleaver and mapper
    d = intlvr(c)
    x = mapper(d)

    # Resource grid
    x_rg = rg_mapper(x)
    x_rg = tf.reshape(x_rg, (batch_size, rg.num_ofdm_symbols, -1))  # [batch_size, num_ofdm_sym, fft_size]
    data_symbol_indices = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]
    x_rg = tf.gather(x_rg, data_symbol_indices, axis=1)  # [batch_size, num_ofdm_sym, fft_size]
    x_rg = tf.transpose(x_rg, [0, 2, 1])  # [batch_size, fft_size, num_ofdm_sym]

    # STBC encoding
    tx = stbc_encode(x_rg)  # [batch_size, fft_size, num_ofdm_sym, num_tx_ant]

    # Extract channel for each tx-rx pairs from ns-3 simulator
    h_freq_ns3 = extract_sttd_channel(slot_idx=2, batch_size=batch_size, num_tx_ue=4, num_rx_ue=4)  # [batch_size, num_rx, num_rxue_ant, num_txue_ant, num_ofdm_sym, fft_size]
    h_freq_ns3 = tf.transpose(h_freq_ns3, [0, 5, 4, 1, 2, 3])  # [batch_size, fft_size, num_ofdm_sym, num_rx, num_rxue_ant, num_txue_ant]
    h_freq_ns3 = flatten_dims(h_freq_ns3, num_dims=2, axis=3)  # [batch_size, fft_size, num_ofdm_sym, num_rx * num_rxue_ant, num_txue_ant]

    # Perfect channel estimation or LMMSE channel estimation for precoding
    h_csi = tf.reshape(h_freq_ns3, (*h_freq_ns3.shape[0:2], 6, 2, -1, 2))  # [batch_size, fft_size, num_ofdm_sym/2, 2, num_rx * num_rxue_ant, num_txue_ant]
    # average over two consecutive symbols
    h_csi = tf.reduce_mean(h_csi, axis=3)  # [batch_size, fft_size, num_ofdm_sym/2, num_rx * num_rxue_ant, num_txue_ant]
    h_csi = tf.reshape(h_csi, (*h_csi.shape[:-2], -1, 2, 2))  # [batch_size, fft_size, num_ofdm_sym/2, num_rx, num_rxue_ant, num_txue_ant]
    h_csi = tf.transpose(h_csi, [0,1,3,2,4,5])  # [batch_size, fft_size, num_rx, num_ofdm_sym/2, num_rxue_ant, num_txue_ant]
    h_csi = flatten_dims(h_csi, num_dims=2, axis=3)  # [batch_size, fft_size, num_rx, num_ofdm_sym, num_txue_ant]

    # apply dMIMO channels to the resource grid in the frequency domain.
    ry = tf.linalg.matmul(h_freq_ns3, tf.expand_dims(tx, -1))  # [batch_size, fft_size, num_ofdm_sym, num_rx * num_rxue_ant, 1]
    ry = tf.reshape(ry, (*ry.shape[:-2], -1, 2))  # [batch_size, fft_size, num_ofdm_sym, num_rx, num_rxue_ant]
    ry = tf.transpose(ry, [0, 1, 3, 2, 4]) # [batch_size, fft_size, num_rx, num_ofdm_sym, num_rxue_ant]

    # STBC decoding
    # assuming perfect CSI
    yd, csi = stbc_decode(ry, h_csi)  # [batch_size, fft_size, num_rx, num_ofm_sym]

    # reshape
    yd = tf.transpose(yd, [0, 2, 3, 1])  # [batch_size, num_rx, num_ofm_sym, fft_size]
    csi = tf.transpose(csi, [0, 2, 3, 1])

    if predetect_combining:
        # Combining all receivers
        yd = tf.reduce_mean(yd, axis=1)  # [batch_size, num_ofm_sym, fft_size]
        csi = tf.reduce_mean(csi, axis=1)
        # Demapping
        yd = yd / tf.cast(csi, tf.complex64)  # CSI scaling
        llr = demapper([yd, nvar / csi])

    else:
        # Demapping
        yd = yd / tf.cast(csi, tf.complex64)  # CSI scaling
        llr = demapper([yd, nvar / csi])
        llr = tf.cast(llr > 0, tf.float32)  # hard decision
        # Combining
        llr = tf.reduce_sum(llr, axis=1)

    llr = tf.reshape(llr, (batch_size, 1, 1, -1))
    x_hard = tf.cast(llr > 0, tf.float32)
    uncoded_ber = compute_ber(d, x_hard)

    llr = dintlvr(llr)
    llr = tf.reshape(llr, [batch_size, 1, rg.num_streams_per_tx, num_codewords, encoder.n])
    b_hat = decoder(llr)
    ber = compute_ber(b, b_hat)

    return uncoded_ber.numpy(), ber.numpy()


def sim_sttd_combining(num_rx=6, modulation_order=2, batch_size=8, code_rate=0.5, quant_bits=1, preldpc_combining=False):

    rg = ResourceGrid(num_ofdm_symbols=14,
                      fft_size=512,
                      subcarrier_spacing=15e3,
                      num_tx=1,
                      num_streams_per_tx=1,
                      cyclic_prefix_length=64,
                      num_guard_carriers=[0, 0],
                      dc_null=False,
                      pilot_pattern="kronecker",
                      pilot_ofdm_symbol_indices=[2, 11])

    num_codewords = modulation_order # number of codewords per frame
    n = int(rg.num_data_symbols*modulation_order/num_codewords) # Number of coded bits
    k = int(n*code_rate) # Number of information bits

    binary_source = BinarySource()
    mapper = Mapper("qam", modulation_order)
    demapper = Demapper("maxlog", "qam", modulation_order)
    rg_mapper = ResourceGridMapper(rg)

    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder, hard_out=True)
    soft_decoder = LDPC5GDecoder(encoder, hard_out=False)
    intlvr = RowColumnInterleaver(3072, axis=-1)
    dintlvr = Deinterleaver(interleaver=intlvr)

    # temporary EbNo settings
    nvar = ebnodb2no(10.0, modulation_order, code_rate, rg)

    # Source and LDPC encoder
    b = binary_source([batch_size, 1, rg.num_streams_per_tx, num_codewords, encoder.k])
    c = encoder(b)
    c = tf.reshape(c, [batch_size, 1, rg.num_streams_per_tx, num_codewords*encoder.n])

    # Interleaver and mapper
    d = intlvr(c)
    x = mapper(d)

    # Resource grid
    x_rg = rg_mapper(x)
    x_rg = tf.reshape(x_rg, (batch_size, rg.num_ofdm_symbols, -1))  # [batch_size, num_ofdm_sym, fft_size]
    data_symbol_indices = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]
    x_rg = tf.gather(x_rg, data_symbol_indices, axis=1)  # [batch_size, num_ofdm_sym, fft_size]
    x_rg = tf.transpose(x_rg, [0, 2, 1])  # [batch_size, fft_size, num_ofdm_sym]

    # STBC encoding
    tx = stbc_encode(x_rg)  # [batch_size, fft_size, num_ofdm_sym, num_tx_ant]

    # Extract channel for each tx-rx pairs from ns-3 simulator
    h_freq_ns3 = extract_sttd_channel(slot_idx=2, batch_size=batch_size, num_tx_ue=4, num_rx_ue=num_rx-2)  # [batch_size, num_rx, num_rxue_ant, num_txue_ant, num_ofdm_sym, fft_size]
    h_freq_ns3 = tf.transpose(h_freq_ns3, [0, 5, 4, 1, 2, 3])  # [batch_size, fft_size, num_ofdm_sym, num_rx, num_rxue_ant, num_txue_ant]
    h_freq_ns3 = flatten_dims(h_freq_ns3, num_dims=2, axis=3)  # [batch_size, fft_size, num_ofdm_sym, num_rx * num_rxue_ant, num_txue_ant]

    # Perfect channel estimation or LMMSE channel estimation for precoding
    h_csi = tf.reshape(h_freq_ns3, (*h_freq_ns3.shape[0:2], 6, 2, -1, 2))  # [batch_size, fft_size, num_ofdm_sym/2, 2, num_rx * num_rxue_ant, num_txue_ant]
    # average over two consecutive symbols
    h_csi = tf.reduce_mean(h_csi, axis=3)  # [batch_size, fft_size, num_ofdm_sym/2, num_rx * num_rxue_ant, num_txue_ant]
    h_csi = tf.reshape(h_csi, (*h_csi.shape[:-2], -1, 2, 2))  # [batch_size, fft_size, num_ofdm_sym/2, num_rx, num_rxue_ant, num_txue_ant]
    h_csi = tf.transpose(h_csi, [0,1,3,2,4,5])  # [batch_size, fft_size, num_rx, num_ofdm_sym/2, num_rxue_ant, num_txue_ant]
    h_csi = flatten_dims(h_csi, num_dims=2, axis=3)  # [batch_size, fft_size, num_rx, num_ofdm_sym, num_txue_ant]

    # apply dMIMO channels to the resource grid in the frequency domain.
    ry = tf.linalg.matmul(h_freq_ns3, tf.expand_dims(tx, -1))  # [batch_size, fft_size, num_ofdm_sym, num_rx * num_rxue_ant, 1]
    ry = tf.reshape(ry, (*ry.shape[:-2], -1, 2))  # [batch_size, fft_size, num_ofdm_sym, num_rx, num_rxue_ant]
    ry = tf.transpose(ry, [0, 1, 3, 2, 4]) # [batch_size, fft_size, num_rx, num_ofdm_sym, num_rxue_ant]

    # STBC decoding
    # assuming perfect CSI
    yd, csi = stbc_decode(ry, h_csi)  # [batch_size, fft_size, num_rx, num_ofm_sym]

    # reshape
    yd = tf.transpose(yd, [0, 2, 3, 1])  # [batch_size, num_rx, num_ofm_sym, fft_size]
    csi = tf.transpose(csi, [0, 2, 3, 1])

    if preldpc_combining:
        # Demapping
        yd = yd / tf.cast(csi, tf.complex64)  # CSI scaling
        llr = demapper([yd, nvar / csi])  # [batch_size, num_rx, num_codewords * encoder.n]
        llr_sign = tf.math.sign(llr)  # 1-bit hard decision
        llr_mag = tf.cast(tf.math.abs(llr) >= 2, tf.float32) + 1.0  # 2.0x or 1.0x
        llr_quant = llr_sign * llr_mag  # 2-bit LLR
        # Pre-LDPC Combining
        llr_comb = tf.reduce_mean(llr_quant, axis=1)
        # Deinterleaving
        llr_comb = tf.reshape(llr_comb, (batch_size, -1))
        decllr = dintlvr(llr_comb)
        # LDPC decoding
        decllr = tf.reshape(decllr, [batch_size, 1, rg.num_streams_per_tx, num_codewords, encoder.n])
        b_hat = decoder(decllr)
        ber = compute_ber(b, b_hat)

    else:
        # Demapping
        yd = yd / tf.cast(csi, tf.complex64)  # CSI scaling
        llr = demapper([yd, nvar / csi])  # [batch_size, num_rx, num_codewords * encoder.n]
        # Deinterleaver
        llr = tf.reshape(llr, (batch_size, num_rx, -1))
        llr = dintlvr(llr)
        # LPDC soft decoding (per Rx)
        llr = tf.reshape(llr, [batch_size, num_rx, rg.num_streams_per_tx, num_codewords, encoder.n])
        decllr = soft_decoder(llr[:,0])
        dec_sign = tf.math.sign(decllr)  # 1-bit quantization
        dec_mag = tf.cast(tf.math.abs(decllr) >= 16, tf.float32) + 1.0  # 2.0x or 1.0x
        decllr_all = dec_sign # * dec_mag  # 2-bit LLR
        for k in range(1,num_rx):
            decllr = soft_decoder(llr[:,k:k+1])
            dec_sign = tf.math.sign(decllr)  # 1-bit quantization
            dec_mag = tf.cast(tf.math.abs(decllr) >= 16, tf.float32) + 1.0  # 2.0x or 1.0x
            decllr_all += dec_sign #* dec_mag  # 2-bit LLR
        # Combining all Rx
        decllr_all = decllr_all / num_rx
        b_hat = tf.cast(decllr_all > 0, tf.float32)
        ber = compute_ber(b, b_hat)

    # goodput for 4ms/baseline or 3ms/STTD (2.0x)
    goodput = (1.0 - ber.numpy()) * rg.num_data_symbols.numpy() * modulation_order * 2.0

    return ber.numpy(), goodput

