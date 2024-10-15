import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
import sionna

class quantized_CSI_feedback(Layer):
    """CSI feedback report generation"""

    def __init__(self,
                method,
                num_tx_streams,
                architecture,
                snrdb,
                dtype=tf.complex64,
                **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)
        
        self.method = method
        if self.method == '5G':
            self.N_1 = 2 # Number of quantization points in the horizontal dimension
            self.N_2 = 1 # Number of quantization points in the vertical dimension
            self.O_1 = 4 # Horizontal oversampling factor
            self.O_2 = 1 # Vertical oversampling factor
        self.num_tx_streams = num_tx_streams
        self.architecture = architecture
        
        snr_linear = 10**(snrdb/10)
        self.snr_linear = np.mean(snr_linear)

        self.num_BS_Ant = 4
        self.num_UE_Ant = 2


    def call(self, h_est):
        
        if self.method == '5G':

            codebook = self.cal_codebook(h_est)
            PMI, rate_for_selected_precoder, precoding_matrix = self.cal_PMI(codebook, h_est)
            
            CSI_feedback_report = [PMI, rate_for_selected_precoder, precoding_matrix]
        
        elif self.method == 'RVQ':

            CSI_feedback_report = cal_RVQ_CSI(h_est)
        else:
            raise Exception(f"The {self.method} CSI feedback mechanism has not been implemented. The simulator supports 5G standard CSI feedback and RVQ CSI feedback only.")
        
        return CSI_feedback_report
    
    def cal_PMI(self, codebook, h_est):

        N_t = h_est.shape[4]
        N_r = h_est.shape[2]

        num_rx_nodes = int((N_r - self.num_BS_Ant)/self.num_UE_Ant) + 1

        A_info = 0.83
        B_info = 0.73

        if self.architecture == 'baseline':

            num_codebook_elements = np.product(codebook.shape[:-2])
            codebook = codebook.reshape(-1, codebook.shape[-2], codebook.shape[-1])

            per_precoder_rate = np.zeros((num_codebook_elements))

            PMI = np.zeros((h_est.shape[-1]),dtype=int)
            rate_for_selected_precoder = np.zeros((h_est.shape[-1]))
            
            for codebook_idx in range(num_codebook_elements):

                h_eff = self.calculate_effective_channel(h_est, codebook[codebook_idx,...])

                snr_linear = np.sum(self.snr_linear)
                n_var = self.cal_n_var(h_eff, snr_linear)

                mmse_inv = tf.matmul(h_eff, h_eff, adjoint_b=True)/self.num_tx_streams + n_var

                per_stream_sinr = self.compute_sinr(h_eff, mmse_inv, n_var)

                avg_sinr = self.eesm_average(per_stream_sinr, 0.25, 4)

                curr_codebook_rate = A_info * np.log2(1 + B_info * avg_sinr)
                per_precoder_rate[:, codebook_idx] = np.sum(curr_codebook_rate, axis=-1)
        
            PMI[n] = np.where(per_precoder_rate == np.max(per_precoder_rate))[0][0]
            rate_for_selected_precoder[n] = per_precoder_rate[PMI[n]]

        return [PMI, rate_for_selected_precoder, codebook[PMI,...]]

    def calculate_effective_channel(self, h_est, precoding_matrix):
    
        h_est_reshaped = tf.transpose(h_est, [0, 1, 3, 5, 6, 2, 4])
        h_est_reshaped = tf.cast(h_est_reshaped, dtype=precoding_matrix.dtype)

        h_eff = tf.matmul(h_est_reshaped, precoding_matrix)

        return h_eff
    
    def cal_n_var(self, h_eff, snr_linear):
        
        prod = tf.matmul(h_eff, h_eff, adjoint_b=True)
        sig_pow = np.abs(np.mean(np.trace(prod, axis1=-2, axis2=-1)))

        n_var = sig_pow / snr_linear

        return n_var
    
    def compute_sinr(self, h_eff, mmse_inv, n_var):
        N_s = h_eff.shape[-1]
        sinr_list = []

        for i in range(N_s):
            h_i = tf.gather(h_eff, i, axis=6)
            h_i = tf.expand_dims(h_i, -1)
            
            # Compute the numerator: |diag(h_i^H * MMSE_R_inv * h_i)|^2
            numerator = tf.abs(tf.linalg.diag_part(tf.matmul(tf.linalg.adjoint(h_i), tf.matmul(mmse_inv, h_i))))**2
            
            # Compute the denominator: n_var * diag(real(h_i^H * MMSE_R_inv * MMSE_R_inv^H * h_i)) + sum(|diag(h_i^H * MMSE_R_inv * h_j)|^2)
            mmse_inv_h_i = tf.matmul(tf.linalg.adjoint(mmse_inv), h_i)
            real_part = tf.linalg.diag_part(tf.math.real(tf.matmul(tf.linalg.adjoint(h_i), tf.matmul(mmse_inv, mmse_inv_h_i))))
            interference_sum = tf.zeros_like(real_part)

            for j in range(N_s):
                if j != i:
                    h_j = tf.gather(h_eff, j, axis=6)
                    h_j = tf.expand_dims(h_j, -1)
                    interference_sum += tf.abs(tf.linalg.diag_part(tf.matmul(tf.linalg.adjoint(h_i), tf.matmul(mmse_inv, h_j))))**2
            
            denominator = n_var * real_part + interference_sum
            
            # Calculate SINR for h_i
            sinr_i = numerator / denominator
            sinr_list.append(sinr_i)

        # Stack the SINR values to form the final SINR tensor
        sinr = tf.stack(sinr_list, axis=-1)
        return sinr
    
    def eesm_average(self, sinr, er, mod):

        if mod == 2:  # QPSK
            beta = 0.413 * er + 1.3661
        elif mod == 4:  # 16-QAM
            beta = 4.4492 * er**2 + 4.5655 * er + 1.2982
        elif mod == 6:  # 64-QAM
            beta = 4.1182 * np.exp(2.4129 * er)
        else:
            raise ValueError('Supported modulation sizes are 2, 4, and 6 only.')

        N = int(np.size(sinr) / (sinr.shape[-1] * sinr.shape[-3]))
        exp_sum = np.sum(np.exp(-sinr / beta), axis=(0,1,2,3,5))
        exp_sum = 1 / N * exp_sum

        if np.any(exp_sum == 0):
            eesm_avg_sinr = np.mean(sinr)
        else:
            eesm_avg_sinr = -beta * np.log(exp_sum)

        return eesm_avg_sinr

    def cal_codebook(self, h_est):
        """
        Computes PMI codebook for 4x2 and 4x4 MIMO configuration (format N_t x N_r)
        Consult 3GPP TS 38.214 Section 5 for details
        """

        N_t = h_est.shape[4]
        N_r = h_est.shape[2]
        P_CSI_RS = N_t

        if N_t == 4 and N_r == 2:
            
            if self.num_tx_streams == 1:
                
                i_11 = np.arange(0, self.N_1 * self.O_1)
                i_12 = np.arange(0, self.N_2 * self.O_2)
                i_2 = np.arange(0,4)


        elif N_t == 4 and N_r == 4:

            if self.num_tx_streams == 1:
                
                i_11 = np.arange(0, self.N_1 * self.O_1)
                i_12 = np.arange(0, self.N_2 * self.O_2)
                i_2 = np.arange(0,4)

                l_all = i_11
                m_all = i_12
                n_all = i_2

                W = np.zeros((len(l_all), len(m_all), len(n_all), N_t, self.num_tx_streams), dtype=complex)

                for l in l_all:
                    for m in m_all:

                        v_lm = self.compute_v_lm(l, m)
                        
                        for n in n_all:

                            phi_n = np.exp(1j * np.pi * n / 2)

                            W[l,m,n,...] = np.vstack((v_lm, phi_n * v_lm))

                W = 1/np.sqrt(P_CSI_RS) * W

            elif self.num_tx_streams == 2:

                i_11 = np.arange(0, self.N_1 * self.O_1)
                i_12 = np.arange(0, self.N_2 * self.O_2)
                i_13 = np.arange(0,2)
                k_1 = np.array((0, self.O_1))
                k_2 = np.array((0, 0))
                i_2 = np.arange(0,2)

                l_all = i_11
                m_all = i_12
                n_all = i_2

                W = np.zeros((len(l_all), len(m_all), len(i_13), len(n_all), N_t, self.num_tx_streams), dtype=complex)

                for l in l_all:
                    for m in m_all:
                        for i_13_idx in i_13:
                            

                            l_ = l + k_1[i_13_idx]
                            m_ = m + k_2[i_13_idx]

                            v_lm = self.compute_v_lm(l, m)
                            v_l_m_ = self.compute_v_lm(l_, m_)
                            
                            for n in n_all:

                                phi_n = np.exp(1j * np.pi * n / 2)
                                
                                col_1 = np.vstack((v_lm, phi_n * v_lm))
                                col_2 = np.vstack((v_l_m_, -phi_n * v_l_m_))
                                W[l,m,i_13_idx,n,...] = np.hstack((col_1, col_2))
                
                W = 1/np.sqrt(2 * P_CSI_RS) * W
            
            elif self.num_tx_streams == 3:

                i_11 = np.arange(0, self.N_1 * self.O_1)
                i_12 = np.arange(0, self.N_2 * self.O_2)
                k_1 = self.O_1
                k_2 = 0
                i_2 = np.arange(0,2)

                l_all = i_11
                m_all = i_12
                n_all = i_2

                W = np.zeros((len(l_all), len(m_all), len(n_all), N_t, self.num_tx_streams), dtype=complex)

                for l in l_all:
                    for m in m_all:                            

                        l_ = l + k_1
                        m_ = m + k_2

                        v_lm = self.compute_v_lm(l, m)
                        v_l_m_ = self.compute_v_lm(l_, m_)
                        
                        for n in n_all:

                            phi_n = np.exp(1j * np.pi * n / 2)
                            
                            col_1 = np.vstack((v_lm, phi_n * v_lm))
                            col_2 = np.vstack((v_l_m_, phi_n * v_l_m_))
                            col_3 = np.vstack((v_lm, -phi_n * v_lm))
                            W[l,m,n,...] = np.hstack((col_1, col_2, col_3))
                
                W = 1/np.sqrt(3 * P_CSI_RS) * W
            
            elif self.num_tx_streams == 4:

                i_11 = np.arange(0, self.N_1 * self.O_1)
                i_12 = np.arange(0, self.N_2 * self.O_2)
                k_1 = self.O_1
                k_2 = 0
                i_2 = np.arange(0,2)

                l_all = i_11
                m_all = i_12
                n_all = i_2

                W = np.zeros((len(l_all), len(m_all), len(n_all), N_t, self.num_tx_streams), dtype=complex)

                for l in l_all:
                    for m in m_all:                            

                        l_ = l + k_1
                        m_ = m + k_2

                        v_lm = self.compute_v_lm(l, m)
                        v_l_m_ = self.compute_v_lm(l_, m_)
                        
                        for n in n_all:

                            phi_n = np.exp(1j * np.pi * n / 2)
                            
                            col_1 = np.vstack((v_lm, phi_n * v_lm))
                            col_2 = np.vstack((v_l_m_, phi_n * v_l_m_))
                            col_3 = np.vstack((v_lm, -phi_n * v_lm))
                            col_4 = np.vstack((v_l_m_, -phi_n * v_l_m_))
                            W[l,m,n,...] = np.hstack((col_1, col_2, col_3, col_4))
                
                W = 1/np.sqrt(4 * P_CSI_RS) * W
            else:
                raise Exception(f"5G standard PMI feedback for {self.num_tx_streams} spatial streams has not been implemented. The simulator supports 1-4 spatial streams only.")

        else:
            raise Exception(f"5G standard PMI feedback for {N_t} x {N_r} MIMO order has not been implemented. The simulator supports MIMO orders 4x2 and 4x4 only.")
        
        return W


    def compute_u_m(self, m):

        if self.N_2 == 1:
            u_m = 1
        elif self.N_2 > 1:
            u_m = np.exp((2j * np.pi * m * np.arange(0, self.N_2)) / (self.O_2 * self.N_2) )
        else:
            raise Exception(f"Incorrect choice of N_2")

        return u_m

    def compute_v_lm(self, l, m):

        u_m = self.compute_u_m(m)

        v_l = np.exp( (2j * np.pi * l * np.arange(0, self.N_1)) / (self.O_1 * self.N_1))

        v_l_m = np.outer(u_m, v_l).flatten()
        v_l_m = v_l_m.reshape(-1, 1)

        return v_l_m

    def reconstruct_channel(self, precoding_matrix, snr_assumed_dBm, n_var, bs_txpwr_dbm):

        rx_sig_pow = n_var * 10**(snr_assumed_dBm/10)
        tx_sig_pow = 10**(bs_txpwr_dbm/10)
        s = np.sqrt(rx_sig_pow / tx_sig_pow)

        h_freq_csi_reconstructed = precoding_matrix * s

        return h_freq_csi_reconstructed