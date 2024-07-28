"""
OFDM frequency and timing synchronization
"""

import tensorflow as tf
import numpy as np

from sionna.signal import fft, ifft

from dmimo.config import SimConfig


def cfo_val(cfg: SimConfig, cfo_hz):
    """
    Compute CFO value relative to subcarrier spacing

    :param cfg: Simulation configuration
    :param cfo_hz: CFO standard deviation (in Hz)
    :return: normalized CFO standard deviation
    """
    return cfo_hz / cfg.subcarrier_spacing


def sto_val(cfg: SimConfig, sto_ns):
    """
    Compute STO value relative to baseband sample duration

    :param cfg: Simulation configuration
    :param sto_ns: STO standard deviation (in nanosecond)
    :return: normalized STO standard deviation
    """
    ts = 1.0 / (cfg.subcarrier_spacing * cfg.fft_size)
    return (sto_ns * 1e-9) / ts


def add_frequency_offset(x, cfo_sigma):
    """
    Add frequency offset errors to OFDM signals
    1) BS antennas has zero CFO errors
    2) all antennas on the same UE have the same CFO

    :param x: OFDM signal grid
    :param cfo_sigma: normalized CFO standard deviation
    :return: OFDM signal grid with random frequency offsets added
    """

    # x has shape [batch_size, num_tx, num_tx_ant, num_ofdm_sym, num_subcarriers]
    num_bs_ant, num_ue_ant = 4, 2
    num_total_ant = x.shape[2]
    num_ue = int(np.ceil((num_total_ant - num_bs_ant) / num_ue_ant))  # TODO: param for BS/UE antennas/streams
    num_ofdm_sym, fft_size = x.shape[-2:]

    # random CFO for UEs
    cfo = np.random.normal(size=(num_ue, 1, 1))
    cfo = np.concatenate((np.zeros((4, 1, 1)), np.repeat(cfo, repeats=2, axis=0)), axis=0)
    cfo = cfo[:num_total_ant]
    cfo_phase = cfo_sigma * cfo * np.linspace(0, num_ofdm_sym, num_ofdm_sym * fft_size, endpoint=False).reshape((1, num_ofdm_sym, fft_size))
    cfo_phase = np.exp(2j * np.pi * cfo_phase)
    cfo_phase = np.reshape(cfo_phase, (1, 1, -1, num_ofdm_sym, fft_size))

    # convert signal to time-domain
    xt = ifft(x)
    # apply phase rotation by frequency offset
    xt = tf.cast(cfo_phase, tf.complex64) * xt
    # convert signal back to frequency-domain
    xf = fft(xt)

    return xf


def add_timing_offset(x, sto_sigma):
    """
    Modeling fractional STO in frequency domain
    1) BS antennas has zero STO errors
    2) all antennas on the same UE have the same STO

    :param x: OFDM signal grid
    :param sto_sigma: normalized STO standard deviation
    :return: OFDM signal grid with random timing offsets added
    """

    # x has shape [batch_size, num_tx, num_tx_ant, num_ofdm_sym, num_subcarriers]
    num_bs_ant, num_ue_ant = 4, 2
    num_total_ant = x.shape[2]
    num_ue = int(np.ceil((num_total_ant - num_bs_ant) / num_ue_ant))  # TODO: param for BS/UE antennas/streams
    num_ofdm_sym, fft_size = x.shape[-2:]

    # Generate random STO for UEs
    sto = np.random.normal(size=(num_ue, 1, 1))
    sto = np.concatenate((np.zeros((4, 1, 1)), np.repeat(sto, repeats=2, axis=0)), axis=0)
    sto = sto[:num_total_ant]
    # maximum relative STO magnitude is 0.5
    #sto[sto > 0.5] = 0.5
    #sto[sto < -0.5] = -0.5
    # compute phase shift in frequency domain
    sto_shift = sto_sigma * sto * np.linspace(-0.5, 0.5, fft_size, endpoint=False).reshape((1, 1, fft_size))
    phase_shift = np.exp(2j * np.pi * sto_shift)
    phase_shift = np.reshape(phase_shift, (1, 1, -1, 1, fft_size))

    # apply STO to BS/UE streams
    x = tf.cast(phase_shift, tf.complex64) * x

    return x
