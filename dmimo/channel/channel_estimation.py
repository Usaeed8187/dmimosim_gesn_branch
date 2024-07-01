"""
Channel estimation for dMIMO scenarios
"""

import numpy as np
import tensorflow as tf

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator
from sionna.mapping import Mapper
from sionna.utils import BinarySource, ebnodb2no

from .dmimo_channels import dMIMOChannels
from .interpolation import LMMSELinearInterp


def estimate_freq_cov(dmimo_chans: dMIMOChannels, start_slot, total_slots=5, fft_size=512):

    fft_size = tf.cast(fft_size, tf.int64)
    freq_cov_mat = tf.zeros([fft_size, fft_size], tf.complex64)

    for slot_idx in np.arange(start_slot, start_slot+total_slots, 1):
        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        h_freq, snrdb = dmimo_chans.load_channel(slot_idx=slot_idx)
        # [batch_size, num_rx_ant, num_tx_ant, num_ofdm_symbols, fft_size]
        h_freq = np.squeeze(h_freq, axis=(1, 3))

        # [batch_size, num_tx_ant, num_rx_ant, num_ofdm_symbols, fft_size]
        h_samples = tf.transpose(h_freq, (0, 2, 1, 3, 4))

        # [num_batch, num_tx_ant, num_rx_ant, fft_size, num_ofdm_symbols]
        h_samples_ = tf.transpose(h_samples, [0,1,2,4,3])
        # [num_tx_ant, num_rx_ant, fft_size, fft_size]
        freq_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        # [fft_size, fft_size]
        freq_cov_mat_ = tf.reduce_mean(freq_cov_mat_, axis=(0,1,2))
        # [fft_size, fft_size]
        freq_cov_mat += freq_cov_mat_

    freq_cov_mat /= tf.complex(tf.cast(10, tf.float32), tf.cast(0.0, tf.float32))

    return freq_cov_mat


def estimate_freq_time_cov(dmimo_chans: dMIMOChannels, start_slot, total_slots=5, num_ofdm_syms=14, fft_size=512):

    fft_size = tf.cast(fft_size, tf.int64)
    num_ofdm_syms = tf.cast(num_ofdm_syms, tf.int64)
    freq_cov_mat = tf.zeros([fft_size, fft_size], tf.complex64)
    time_cov_mat = tf.zeros([num_ofdm_syms, num_ofdm_syms], tf.complex64)

    for slot_idx in np.arange(start_slot, start_slot+total_slots, 1):
        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        h_freq, snrdb = dmimo_chans.load_channel(slot_idx=slot_idx)
        # [batch_size, num_rx_ant, num_tx_ant, num_ofdm_symbols, fft_size]
        h_freq = np.squeeze(h_freq, axis=(1, 3))

        # [batch_size, num_tx_ant, num_rx_ant, num_ofdm_symbols, fft_size]
        h_samples = tf.transpose(h_freq, (0, 2, 1, 3, 4))

        # [num_batch, num_tx_ant, num_rx_ant, fft_size, num_ofdm_symbols]
        h_samples_ = tf.transpose(h_samples, [0,1,2,4,3])
        # [num_tx_ant, num_rx_ant, fft_size, fft_size]
        freq_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        # [fft_size, fft_size]
        freq_cov_mat_ = tf.reduce_mean(freq_cov_mat_, axis=(0,1,2))
        # [fft_size, fft_size]
        freq_cov_mat += freq_cov_mat_

        # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
        time_cov_mat_ = tf.matmul(h_samples, h_samples, adjoint_b=True)
        # [num_ofdm_symbols, num_ofdm_symbols]
        time_cov_mat_ = tf.reduce_mean(time_cov_mat_, axis=(0,1,2))
        # [num_ofdm_symbols, num_ofdm_symbols]
        time_cov_mat += time_cov_mat_

    freq_cov_mat /= tf.complex(tf.cast(10, tf.float32), tf.cast(0.0, tf.float32))
    time_cov_mat /= tf.complex(tf.cast(fft_size*total_slots, tf.float32), 0.0)

    return freq_cov_mat, time_cov_mat


def lmmse_channel_estimation(dmimo_chans: dMIMOChannels, rg: ResourceGrid, slot_idx, cache_slots=5, ebno_db=10.0):

    # Only allow channel estimation from slot 1 onward
    assert slot_idx > 0, "Current slot index must be a positive integer"

    # Make sure slot_idx is always non-negative
    if slot_idx - cache_slots < 0:
        cache_slots = slot_idx
    start_slot = slot_idx - cache_slots

    num_bits_per_symbol = 2  # use QPSK modulation
    binary_source = BinarySource()
    mapper = Mapper("qam", num_bits_per_symbol)
    rg_mapper = ResourceGridMapper(rg)

    freq_cov_mat = estimate_freq_cov(dmimo_chans, start_slot=start_slot, total_slots=cache_slots)
    lmmse_int = LMMSELinearInterp(rg.pilot_pattern, freq_cov_mat)
    ls_estimator = LSChannelEstimator(rg, interpolator=lmmse_int)

    # Calculate noise variance for LS channel estimation
    nvar = ebnodb2no(ebno_db, num_bits_per_symbol, 0.5)

    # Generate OFDM grid signals
    bs = binary_source([1, 1, rg.num_streams_per_tx, rg.num_data_symbols * num_bits_per_symbol])
    dx = mapper(bs)
    dx_rg = rg_mapper(dx)

    # Pass through ns3 channels
    # output has shape: [1, num_rx, num_rx_ant, num_ofdm_sym, fft_size]
    ry = dmimo_chans([dx_rg, slot_idx])

    #
    # LMMSE channel estimation
    #
    num_rx_ant = ry.shape[2]
    h_all = []
    err_var_all = []
    # loop for individual receiver antennas in each batch to reduce memory requirement
    for idx in range(num_rx_ant):
        ry1 = ry[:1, :1, idx:idx+1, :, :]
        h_hat, err_var = ls_estimator([ry1, nvar])
        h_all.append(h_hat)
        err_var_all.append(err_var)

    h_all = tf.concat(h_all, axis=2)
    evar_all = tf.concat(err_var_all, axis=2)

    return h_all, evar_all

