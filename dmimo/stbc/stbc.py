"""
Space-time block codes (STBC)

Reference: S. M. Alamouti, "A simple transmit diversity technique for wireless communications,"
in IEEE Journal on Selected Areas in Communications, vol. 16, no. 8, pp. 1451-1458, Oct. 1998.

Alamouti model:
    For a single receive antenna:
        at time t1: r1 = h1 . s1 + h2 . s2
        at time t2: r2 = h1 . (-s2*) + h2 . (s1*)

        z1 = h1* . (r1) + h2 . (r2*) = (|h1|**2 + |h2|**2) . (s1) + h1* . (n1) + h2 . (n2*)
        z2 = h2* . (r1) - h1 . (r2*) = (|h1|**2 + |h2|**2) . (s2) + h2* . (n1) + h1 . (n2*)

    Then, for M_r antennas we sum over z1's for detecting s1 and sum over z2's for detecting s2.

3/4 - rate STBC code with 4 transmit antennas

    We consider the model that uses 4 transmit antennas to send 3 symbols in the course of 4 time slots.

    S =
    |   s1   s2   s3   0   |
    |  -s2*  s1*  0    s3  |
    |  -s3*  0    s1* -s2  |
    |   0   -s3*  s2*  s1  |

    For a single receive antenna:
        at time t1: r1 = h1 . (s1)   + h2 . (s2)  + h3 . (s3)  + h4 . (0)
        at time t2: r2 = h1 . (-s2*) + h2 . (s1*) + h3 . (0)   + h4 . (s3)
        at time t3: r3 = h1 . (-s3*) + h2 . (0)   + h3 . (s1*) + h4 . (-s2)
        at time t3: r4 = h1 . (0)    + h2 . (-s3*)+ h3 . (s2*) + h4 . (s1)

        z1 = h1*.(r1) + h2.(r2*) + h3.(r3*) + h4*.(r4)
          = (|h1|**2 + |h2|**2 + |h3|**2 + |h4|**2).(s1) + noise
        
        z2 = h2*.(r1) - h1.(r2*) - h4*.(r3) + h3.(r4_)
          = (|h1|**2 + |h2|**2 + |h3|**2 + |h4|**2) . (s2) + noise
        
        z3 = h3*.(r1) + h4*.(r2) - h1.(r3*) - h2.(r4*)
          = (|h1|**2 + |h2|**2 + |h3|**2 + |h4|**2) . (s3) + noise
        
    Then, for M_r antennas we sum over z1's for detecting s1, sum over z2's for detecting s2 and
    sum over z3's for detecting s3.

1/2 - rate STBC code with 8 transmit antennas
    We consider the model that uses 8 transmit antennas to send 4 symbols in the course of 8 time slots.

    S =
    | s1    s2    s3    0     s4    0     0     0  |
    |-s2*   s1*   0     s3    0     s4    0     0  |
    | s3*   0    -s1*   s2    0     0     s4    0  |
    | 0     s3*  -s2*  -s1    0     0     0     s4 |
    | s4*   0     0     0    -s1*   s2   -s3    0  |
    | 0     s4*   0     0    -s2*  -s1    0    -s3 |
    | 0     0     s4*   0    -s3*   0     s1    s2 |
    | 0     0     0     s4*   0    -s3*  -s2*   s1*|
    

    For a single receive antenna:
        r̲ = Sh̲ , where h̲ = [h1  h2  h3  ...  h8] is the channel vector from the transmitters
        to that specific receive antenna.

        z1 = h1*.r1 + h2.r2* - h3.r3* - h4*.r4 - h5.r5* - h6*.r6 + h7*.r7 + h8.r8*
          = (|h1|**2 + |h2|**2 + |h3|**2 + |h4|**2 + |h5|**2 + |h6|**2 + |h7|**2 + |h8|**2).(s1) + noise
        
        z2 = h2*.r1 - h1.r2* + h4*.r3 - h3.r4* + h6*.r5  - h5.r6* + h8*.r7 - h7.r8*
          = (|h1|**2 + |h2|**2 + |h3|**2 + |h4|**2 + |h5|**2 + |h6|**2 + |h7|**2 + |h8|**2).(s1) + noise
        
        z3 = h3*.r1 + h4*.r2 + h1.r3* + h2.r4* - h7*.r5 - h8*.r6 - h5.r7* - h6.r8* 
          = (|h1|**2 + |h2|**2 + |h3|**2 + |h4|**2 + |h5|**2 + |h6|**2 + |h7|**2 + |h8|**2).(s1) + noise

        z4 = h5*.r1 + h6*.r2 + h7*.r3 + h8*.r4 + h1.r5* + h2.r6* + h3.r7* + h4.r8*
          = (|h1|**2 + |h2|**2 + |h3|**2 + |h4|**2 + |h5|**2 + |h6|**2 + |h7|**2 + |h8|**2).(s1) + noise
        
    Then, for M_r antennas we sum over z1's for detecting s1, sum over z2's for detecting s2,
    sum over z3's for detecting s3 and sum over z4's for detecting s4.

"""

import tensorflow as tf
import numpy as np

"""
class OSTBC():
    import sympy as sp # Importing inside the class so that users who don't want to install SymPy would be good to go.
    def __init__(self, scheme:str = None, S_matrix:sp.matrices.dense.MutableDenseMatrix = None) -> None:
        '''
        A class that handles orthogonal space-time block codes (OSTBC) -- Incomplete. DO NOT USE.

        Arguments
        ---------

            ``scheme``:
            Currently supported schemes are 'alamouti', '434' and '848'. If None, the user has to provide the STBC characteristic matrix ``S_matrix``.

            ``S_matrix`` : The STBC characteristic matrix. This is a T by Nt matrix where each row represents
              a time slot and each column represents a transmit antenna

        Attributes
        ----------

            ``S_transpose_S``: The result of multiplying the transpose of a row of S by another row of S.
            The result is a numpy array of shape (T,T,Nt,Nt)

            ``S_hermitian_S``: The result of multiplying the hermitian of a row of S by another row of S.
            The result is a numpy array of shape (T,T,Nt,Nt)
        '''
        assert ((scheme is not None) and (S_matrix is None)) , 'User input S_matrix is not supported yet.'
        if scheme == 'alamouti':
            s1,s2 = OSTBC.sp.symbols('s_1,s_2')
            s1_ = OSTBC.sp.conjugate(s1)
            s2_ = OSTBC.sp.conjugate(s2)
            S = OSTBC.sp.Matrix([[s1,s2],
                                 [-s2_, s1_]])
            self.info_syms = [s1,s2]
        elif scheme == '434':
            s1,s2,s3 = OSTBC.sp.symbols('s_1,s_2,s_3')
            s1_ = OSTBC.sp.conjugate(s1)
            s2_ = OSTBC.sp.conjugate(s2)
            s3_ = OSTBC.sp.conjugate(s3)
            S = OSTBC.sp.Matrix([[s1,s2,s3,0],
                           [-s2_, s1_, 0, s3],
                           [-s3_,0,s1_,-s2],
                           [0,-s3_,s2_,s1]])
            self.info_syms = [s1,s2,s3]
        elif scheme == '848':
            s1,s2,s3,s4 = OSTBC.sp.symbols('s_1,s_2,s_3,s_4')
            s1_ = OSTBC.sp.conjugate(s1)
            s2_ = OSTBC.sp.conjugate(s2)
            s3_ = OSTBC.sp.conjugate(s3)
            s4_ = OSTBC.sp.conjugate(s4)
            S = OSTBC.sp.Matrix([[s1,s2,s3,0,s4,0,0,0],
                                 [-s2_, s1_, 0, s3,0,s4,0,0],
                                 [s3_,0,-s1_,s2,0,0,s4,0],
                                 [0,s3_,-s2_,-s1,0,0,0,s4],
                                 [s4_,0,0,0,-s1_,s2,-s3,0],
                                 [0,s4_,0,0,-s2_,-s1,0,-s3],
                                 [0,0,s4_,0,-s3_,0,s1,s2],
                                 [0,0,0,s4_,0,-s3_,-s2_,s1_]])
            self.info_syms = [s1,s2,s3,s4]
        else:
            raise ValueError('scheme should be either alamouti, 434 or 848')
        A = OSTBC.sp.conjugate(S.T)*S
        assert A == OSTBC.sp.diag(OSTBC.sp.diag(A)), 'This is not an orthogonal STBC...'
        self.S = S
        self.T = S.shape[0] # Number of STBC time slots
        self.Nt = S.shape[1] # Number of STBC transmit antennas
        # Distinct symbols in the matrix (S.atoms() returns all the distinct elements of S)
        info_syms = ([element for element in S.atoms() if type(element)==OSTBC.sp.core.symbol.Symbol])
        self.num_STBC_info_syms = len(info_syms)

        self.S_transpose_S = np.zeros([S.shape[0],S.shape[0],S.shape[1],S.shape[1]])
        for i_row in range(S.shape[0]):
            for j_row in range(S.shape[0]):
                A = S[j_row,:].T * S[i_row,:]
                self.S_transpose_S[i_row,j_row] = np.array(self._take_expectation(A))
                pass
            pass
        self.S_hermitian_S = np.zeros([S.shape[0],S.shape[0],S.shape[1],S.shape[1]])
        for i_row in range(S.shape[0]):
            for j_row in range(S.shape[0]):
                A = OSTBC.sp.conjugate(S[j_row,:].T) * S[i_row,:]
                self.S_hermitian_S[i_row,j_row] = np.array(self._take_expectation(A)).astype(int)
                pass
            pass

        # Here, we will figure out the receive processor
        S_array = np.array(S)
        self.which_index_h = np.zeros([len(self.info_syms),S_array.shape[0]],dtype=int) # As large as the number of time slots
        self.conjugate_h = np.zeros([len(self.info_syms),S_array.shape[0]],dtype=bool) # Same shape as h_indices_processed
        self.negate_h = np.zeros([len(self.info_syms),S_array.shape[0]],dtype=bool) # Same shape as h_indices_processed

        for i_symbol,symbol in enumerate(self.info_syms):
            # h_tilde, a vector that should get multiplied by the vector [r,r.conjugate] to give z.
            h_indices_processed = np.zeros(S_array.shape[0],dtype=int) # As large as the number of time slots
            conjugation = np.zeros(S_array.shape[0],dtype=bool) # Same shape as h_indices_processed
            negation = np.zeros(S_array.shape[0],dtype=bool) # Same shape as h_indices_processed
            for conjugate_flag in [False,True]:
                if conjugate_flag:
                    s = OSTBC.sp.conjugate(symbol)
                else:
                    s = symbol
                for i_time in range(S_array.shape[0]):
                    for i_antenna in range(S_array.shape[1]):
                        if S_array[i_time,i_antenna] == s:
                            h_indices_processed[i_time] = i_antenna
                            conjugation[i_time] = (not conjugate_flag) # Whatever conjugation on r, the reverse on h.
                        elif S_array[i_time,i_antenna] == -s:
                            h_indices_processed[i_time] = i_antenna
                            conjugation[i_time] = (not conjugate_flag)
                            negation[i_time] = True
            self.which_index_h[i_symbol] = h_indices_processed
            self.conjugate_h[i_symbol] = conjugation
            self.negate_h[i_symbol] = negation
        self.conjugate_r = ~self.conjugate_h # Whenever h is conjugated, r is not and vice versa

    def _take_expectation(self, A:sp.matrices.dense.MutableDenseMatrix):
        E_A = OSTBC.sp.zeros(*A.shape)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                a = A[i,j]
                if a != 0:
                    assert a.func == OSTBC.sp.core.mul.Mul or a.func == OSTBC.sp.core.power.Pow
                    if a.func == OSTBC.sp.core.mul.Mul: # If it is power, the expectation should be zero
                        first_multiplier = a.args[0]
                        second_multiplier = a.args[1]
                        conj_flag_1 = conj_flag_2 = False
                        if type(first_multiplier) == OSTBC.sp.conjugate:
                            conj_flag_1 = True
                            first_multiplier = OSTBC.sp.conjugate(first_multiplier)
                        if type(second_multiplier) == OSTBC.sp.conjugate:
                            conj_flag_2 = True
                            second_multiplier = OSTBC.sp.conjugate(second_multiplier)
                        if conj_flag_1 == conj_flag_2:
                            E_A[i,j] = 0
                        else:
                            if first_multiplier != second_multiplier:
                                E_A[i,j] = 0
                            else:
                                E_A[i,j] = 1
        return E_A
    def find_interference_power(self,h_intended:tf.Tensor,h_not_intended:tf.Tensor):
        '''
        Computes the interference power of one cluster on the signal coming from
        the other cluster.

        WARNING: This function assumes that the intended signal and the interference
        are both using the same STBC scheme. Different schemes should be supported
        in the future.

        Arguments
        ---------

            ``h_intended``: A tensor of shape (...,Mr, Nt) where Mr and Nt is
            the number of receive and transmit antennas. This is the channel
            from the intended cluster to the receiver

            ``h_not_intended``: A tensor of shape (...,Mr, Nt) where Mr and Nt is
            the number of receive and transmit antennas. This is the channel
            from the interfering cluster to the receiver.

        Output
        ------

            ``per_symbol_intf_power``: a tensor of size (..., self.num_STBC_info_syms)
            which represents the interference power for each info symbol

        '''
        assert h_intended.shape[-1] == h_not_intended.shape[-1] == self.Nt
        assert h_intended.shape[-2] == h_not_intended.shape[-2]
        # Create h_tilde
        h_tilde_all = []
        for i_sym in range(self.num_STBC_info_syms):
            h_tilde = []
            for t in range(self.T):
                if self.conjugate_r[i_sym,t]:
                    h_tilde.append(tf.zeros_like(h_intended[...,0]))
                elif self.conjugate_h[i_sym,t]:
                    conjugate_or_not = (lambda x: tf.math.conj(x)) if self.conjugate_h[i_sym,t] else (lambda x:x)
                    h_tilde.append(conjugate_or_not(h_intended[...,self.which_index_h[i_sym,t]]))
            for t in range(self.T):
                if self.conjugate_r[i_sym,t]:
                    conjugate_or_not = (lambda x: tf.math.conj(x)) if self.conjugate_h[i_sym,t] else (lambda x:x)
                    h_tilde.append(conjugate_or_not(h_intended[...,self.which_index_h[i_sym,t]]))
                elif self.conjugate_h[i_sym,t]:
                    h_tilde.append(tf.zeros_like(h_intended[...,0]))

            h_tilde_all.append(tf.stack(h_tilde,axis=-1)) # tf.stack(h_tilde,axis=-1): shape = (...,Mr,2T)
            pass
        del h_tilde
        h_tilde_all = tf.stack(h_tilde_all,axis=-2) # shape = (..., Mr, num_info_syms, 2T)
        assert h_tilde_all.shape[-1] == 2*self.T

        # Figure out 𝔼{r rᴴ} for the interference r
        shape = [self.T,self.T,*[1 for _ in range(h_intended.ndim-1)],self.Nt,self.Nt]
        S_hermitian_S = tf.reshape(tf.convert_to_tensor(self.S_hermitian_S,dtype=h_intended.dtype),shape)
        S_transpose_S = tf.reshape(tf.convert_to_tensor(self.S_transpose_S,dtype=h_intended.dtype),shape)

        W = (tf.math.conj(h_not_intended[...,tf.newaxis,:]) @
             S_hermitian_S @
             (h_not_intended[...,:,tf.newaxis])) # (T,T,...,1,1)
        # W = tf.transpose()
        W_hat = ((h_not_intended[...,tf.newaxis,:]) @
                 S_transpose_S @
                 (h_not_intended[...,:,tf.newaxis])) # (T,T,...,1,1)
        Q = tf.concat([tf.concat((W,W_hat),axis=1),
                       tf.concat((tf.math.conj(W_hat),tf.math.conj(W)),axis=1)],
                       axis=0)

        Q = tf.transpose(Q,(*range(2,Q.ndim),0,1)) # (...,2T,2T)

        # Interference Power
        per_symbol_intf_power = h_tilde_all[...,tf.newaxis,tf.newaxis,:] @ Q @ tf.math.conj(h_tilde_all[...,tf.newaxis,:,tf.newaxis])
        per_symbol_intf_power = per_symbol_intf_power[...,0,0,0] # (...,Mr,num_info_syms)
        per_symbol_intf_power = tf.reduce_sum(per_symbol_intf_power,axis=-2) # (...,num_info_syms)

        return per_symbol_intf_power
"""


def alamouti_encode(x):
    """
    Alamouti encoder

    :param x: input symbols, shape [..., num_syms]
    :return: encoded symbols, shape [...,num_syms,2], where 2 is the number of Tx Antennas
    """

    # check length of input symbols
    assert x.shape[-1] % 2 == 0, "total number of symbols must be even"

    x_t1 = tf.reshape(x, (*x.shape[:-1], -1, 1, 2))  # [...,num_syms/2,1,2]
    x_first = x_t1[...,0:1] # First symbol Shape is [...,num_syms/2,1,1]
    x_second = x_t1[...,1:2] # Second symbol Shape is [...,num_syms/2,1,1]
    x_t2 = tf.concat([-tf.math.conj(x_second),tf.math.conj(x_first)], axis = -1) # [...,num_syms/2,1,2]

    x_alamouti = tf.concat((x_t1,x_t2),axis=-2) # [...,num_syms/2,2,2]
    x_alamouti = tf.reshape(x_alamouti,[*x_alamouti.shape[:-3],-1,2]) # [...,num_syms,2]
    return x_alamouti


def alamouti_decode(r, h):
    """
    Alamouti decoder.
    We have M_r receive antennas, for num_syms consecutive symbols

    :param r: received symbols. shape [..., num_syms/2, 2, M_r], where 2 is representing first or second received symbol
    :param h: channel estimation for channels, shape  [..., num_syms/2, M_r, 2], where 2 is representing the first and second transmit antenna
    :return: estimation symbols, shape [..., num_syms]
    """

    # check input data dimension
    r = tf.transpose(r, (*range(r.ndim-2),r.ndim-1,r.ndim-2)) # [..., num_syms/2, M_r, 2]
    assert r.shape == h.shape, "channel estimation must have matched shape as received symbols"
    assert h.shape[-1] == 2, "total number of tx antennas must be two"

    # split received symbols into two sets (y0, y1), last dimension is receive antenna index
    # see table III in reference
    # r0 = y[..., 0, 0], r2 = y[..., 0, 1]
    # r1 = y[..., 1, 0], r3 = y[..., 1, 1]

    z1 = tf.math.conj(h[...,0:1])*r[...,0:1] + h[...,1:2]*tf.math.conj(r[...,1:2]) # z1 = h1^* r1 + h2 r2^*
    z2 = tf.math.conj(h[...,1:2])*r[...,0:1] - h[...,0:1]*tf.math.conj(r[...,1:2]) # z2 = h2^* r1 - h1 r2^*

    z = tf.concat((z1,z2),axis=-1) # Shape = [..., num_syms/2, M_r, 2]
    z = tf.reduce_sum(z,axis=-2) # Sum over M_r receive antennas (..., num_syms/2, 2)
    z = tf.reshape(z, (*r.shape[:-3],r.shape[-3]*2)) # (..., num_syms)
 
    h_eq = tf.math.abs(h[...,0:1])**2 + tf.math.abs(h[...,1:2])**2
    h_eq = tf.concat((h_eq,h_eq),axis=-1) # Duplicating. Shape = (..., num_syms/2, Mr, 2)
    h_eq = tf.reduce_sum(h_eq,axis=-2) # (..., num_syms/2, 2)
    h_eq = tf.reshape(h_eq, (*r.shape[:-3],r.shape[-3]*2)) # (..., num_syms)

    return z, h_eq


def stbc_encode_434(x):
    """
    Space-time block code (STBC) encoder with 4 transmit antennas, sending 3 symbols in 4 time slots (434)
    
    :param x: input symbols, shape [..., num_syms], num_syms must be a multiple of 3.
    
    :return: encoded symbols, shape [...,4 * num_syms/3, 4 ], where the last dimension is the number of Tx Antennas
    """

    # check length of input symbols
    assert x.shape[-1] % 3 == 0, "total number of symbols must be a multiple of 3"
    num_syms = x.shape[-1]
    x_t1 = tf.reshape(x, (*x.shape[:-1], num_syms//3, 1, 3))  # [...,num_syms/3,1,3]
    x_first = x_t1[..., 0:1]  # First symbol. Shape is [...,num_syms/3,1,1]
    x_second = x_t1[..., 1:2]  # Second symbol. Shape is [...,num_syms/3,1,1]
    x_third = x_t1[..., 2:3]  # Third symbol. Shape is [...,num_syms/3,1,1]
    x_zeros = tf.zeros_like(x_first) # Null symbol. Shape is [...,num_syms/3,1,1]
    x_t1 = tf.concat((x_t1,x_zeros),axis=-1) # [...,num_syms/3,1,4]
    x_t2 = tf.concat([-tf.math.conj(x_second),tf.math.conj(x_first),x_zeros,x_third]
                     , axis = -1) # [...,num_syms/3,1,4]
    x_t3 = tf.concat([-tf.math.conj(x_third),x_zeros,tf.math.conj(x_first),-x_second]
                     , axis = -1) # [...,num_syms/3,1,4]
    x_t4 = tf.concat([x_zeros,-tf.math.conj(x_third),tf.math.conj(x_second),x_first]
                     , axis = -1) # [...,num_syms/3,1,4]

    x_stbc = tf.concat((x_t1,x_t2,x_t3,x_t4),axis=-2) # [...,num_syms/3,4,4]
    x_stbc = tf.reshape(x_stbc,[*x_stbc.shape[:-3],num_syms//3 * 4, 4]) # [...,num_syms/3 * 4, 4]
    return x_stbc

def stbc_decode_434(r, h):
    """
    Space-time block code (STBC) decoder with 4 transmit antennas, sending 3 symbols in 4 time slots (434)
    We have M_r receive antennas, for num_syms consecutive symbols
    
    :param r: received symbols. shape [..., num_syms/4, 4, M_r], where 4 is representing 4 consecutive time slots
    
    :param h: channel estimation for channels, shape [..., num_syms/4, M_r, 4], where 4 is representing the transmit antennas
    
    :return: estimation symbols, shape [..., 3*num_syms/4]
    """

    # check input data dimension
    r = tf.transpose(r, (*range(r.ndim-2),r.ndim-1,r.ndim-2)) # [..., num_syms/4, M_r, 4]
    assert r.shape == h.shape, "Dimensions of r and h do not make sense."
    assert h.shape[-1] == 4, "total number of tx antennas must be four"

    # z1 = h1*.(r1) + h2.(r2*) + h3.(r3*) + h4.(r4)
    z1 = (tf.math.conj(h[...,0:1])*(r[...,0:1]) +
          (h[...,1:2])*tf.math.conj(r[...,1:2]) + 
          (h[...,2:3])*tf.math.conj(r[...,2:3]) +
          tf.math.conj(h[...,3:4])*(r[...,3:4]))
    
    # z2 = h2*.(r1) - h1.(r2*) - h4*.(r3) + h3*.(r4_)
    z2 = (tf.math.conj(h[...,1:2])*(r[...,0:1]) -
          (h[...,0:1])*tf.math.conj(r[...,1:2]) -
          tf.math.conj(h[...,3:4])*(r[...,2:3]) +
          (h[...,2:3])*tf.math.conj(r[...,3:4]))
    
    # z3 = h3*.(r1) + h4*.(r2) - h1.(r3*) - h2.(r4*)
    z3 = (tf.math.conj(h[...,2:3])*(r[...,0:1]) +
          tf.math.conj(h[...,3:4])*(r[...,1:2]) -
          (h[...,0:1])*tf.math.conj(r[...,2:3]) -
          (h[...,1:2])*tf.math.conj(r[...,3:4]))
    
    z = tf.concat((z1,z2,z3),axis=-1) # Shape = [..., num_syms/4, M_r, 3]
    z = tf.reduce_sum(z,axis=-2) # Sum over M_r receive antennas (..., num_syms/4, 3)
    z = tf.reshape(z, (*r.shape[:-3],r.shape[-3]*3)) # (..., 3*num_syms/4)
 
    h_eq = tf.reduce_sum(tf.math.abs(h)**2,axis=-1,keepdims=True) # (..., num_syms/4, M_r, 1)
    h_eq = tf.concat((h_eq,h_eq,h_eq),axis=-1) # Duplicating. Shape = (..., num_syms/4, M_r, 3)
    h_eq = tf.reduce_sum(h_eq,axis=-2) # (..., num_syms/4, 3)
    h_eq = tf.reshape(h_eq, (*r.shape[:-3],r.shape[-3]*3)) # (..., 3*num_syms/4)

    return z, h_eq


def stbc_encode_848(x):
    """
    Space-time block code (STBC) encoder with 8 transmit antennas, sending 4 symbols in 8 time slots (848)

    :param x: input symbols, shape [..., num_syms], num_syms must be a multiple of 4.
    
    :return: encoded symbols, shape [...,8 * num_syms/4, 8 ], where the last dimension is the number of Tx Antennas
    """

    # check length of input symbols
    assert x.shape[-1] % 4 == 0, "total number of symbols must be a multiple of 4"
    num_syms = x.shape[-1]
    x_reshaped = tf.reshape(x, (*x.shape[:-1], num_syms//4, 1, 4))  # [...,num_syms/4,1,4]
    x_first = x_reshaped[...,0:1] # First symbol. Shape is [...,num_syms/4,1,1]
    x_first_star = tf.math.conj(x_first)
    x_second = x_reshaped[...,1:2] # Second symbol. Shape is [...,num_syms/4,1,1]
    x_second_star = tf.math.conj(x_second)
    x_third = x_reshaped[...,2:3] # Third symbol. Shape is [...,num_syms/4,1,1]
    x_third_star = tf.math.conj(x_third)
    x_fourth = x_reshaped[...,3:4] # Fourth symbol. Shape is [...,num_syms/4,1,1]
    x_fourth_star = tf.math.conj(x_fourth)
    x_zeros = tf.zeros_like(x_first) # Null symbol. Shape is [...,num_syms/4,1,1]
    x_t1 = tf.concat((x_first, x_second, x_third,x_zeros,x_fourth,x_zeros,x_zeros,x_zeros),
                     axis=-1) # [...,num_syms/4,1,8]
    x_t2 = tf.concat([-x_second_star,x_first_star,x_zeros,x_third,x_zeros,x_fourth,x_zeros,x_zeros]
                     , axis = -1) # [...,num_syms/4,1,8]
    x_t3 = tf.concat([x_third_star,x_zeros,-x_first_star,x_second,x_zeros,x_zeros,x_fourth,x_zeros]
                     , axis = -1) # [...,num_syms/4,1,8]
    x_t4 = tf.concat([x_zeros,x_third_star,-x_second_star,-x_first,x_zeros,x_zeros,x_zeros,x_fourth]
                     , axis = -1) # [...,num_syms/4,1,8]
    x_t5 = tf.concat((x_fourth_star,x_zeros,x_zeros,x_zeros, -x_first_star, x_second, -x_third,x_zeros),
                     axis=-1) # [...,num_syms/4,1,8]
    x_t6 = tf.concat([x_zeros,x_fourth_star,x_zeros,x_zeros,-x_second_star,-x_first,x_zeros,-x_third]
                     , axis = -1) # [...,num_syms/4,1,8]
    x_t7 = tf.concat([x_zeros,x_zeros,x_fourth_star,x_zeros,-x_third_star,x_zeros,x_first,x_second]
                     , axis = -1) # [...,num_syms/4,1,8]
    x_t8 = tf.concat([x_zeros,x_zeros,x_zeros,x_fourth_star,x_zeros,-x_third_star,-x_second_star,x_first_star]
                     , axis = -1) # [...,num_syms/4,1,8]
    

    x_stbc = tf.concat((x_t1,x_t2,x_t3,x_t4,x_t5,x_t6,x_t7,x_t8)
                       ,axis=-2) # [...,num_syms/4,8,8]
    x_stbc = tf.reshape(x_stbc,[*x_stbc.shape[:-3],num_syms//4 * 8, 8]) # [...,num_syms/4 * 8, 8]
    return x_stbc

def stbc_decode_848(r, h):
    """
    Space-time block code (STBC) decoder with 8 transmit antennas, sending 4 symbols in 8 time slots (434)
    We have M_r receive antennas, for num_syms consecutive symbols
    
    :param y: received symbols. shape [..., num_syms/8, 8, M_r], where 8 is representing 8 consecutive time slots
    
    :param h: channel estimation for channels, shape  [..., num_syms/8, M_r, 8], where 8 is representing the transmit antennas
    
    :return: estimation symbols, shape [..., 4*num_syms/8]
    """

    # check input data dimension
    r = tf.transpose(r, (*range(r.ndim-2),r.ndim-1,r.ndim-2)) # [..., num_syms/4, M_r, 8]
    assert r.shape == h.shape, "Dimensions of r and h do not make sense."
    assert h.shape[-1] == 8, "total number of tx antennas must be eight"

    # z1 = h1*.r1 + h2.r2* - h3.r3* - h4*.r4 - h5.r5* - h6*.r6 + h7*.r7 + h8.r8*
    z1 = (  tf.math.conj(h[...,0:1])*(r[...,0:1])
          + (h[...,1:2])*tf.math.conj(r[...,1:2]) 
          - (h[...,2:3])*tf.math.conj(r[...,2:3])
          - tf.math.conj(h[...,3:4])*(r[...,3:4])
          - (h[...,4:5])*tf.math.conj(r[...,4:5])
          - tf.math.conj(h[...,5:6])*(r[...,5:6])
          + tf.math.conj(h[...,6:7])*(r[...,6:7])
          + (h[...,7:8])*tf.math.conj(r[...,7:8]))
    
    # z2 = h2*.r1 - h1.r2* + h4*.r3 - h3.r4* + h6*.r5  - h5.r6* + h8*.r7 - h7.r8*
    z2 = (  tf.math.conj(h[...,1:2])*(r[...,0:1])
          - (h[...,0:1])*tf.math.conj(r[...,1:2]) 
          + tf.math.conj(h[...,3:4])*(r[...,2:3])
          - (h[...,2:3])*tf.math.conj(r[...,3:4])
          + tf.math.conj(h[...,5:6])*(r[...,4:5])
          - (h[...,4:5])*tf.math.conj(r[...,5:6])
          + tf.math.conj(h[...,7:8])*(r[...,6:7])
          - (h[...,6:7])*tf.math.conj(r[...,7:8]))
    
    # z3 = h3*.r1 + h4*.r2 + h1.r3* + h2.r4* - h7*.r5 - h8*.r6 - h5.r7* - h6.r8* 
    z3 = (  tf.math.conj(h[...,2:3])*(r[...,0:1])
          + tf.math.conj(h[...,3:4])*(r[...,1:2])
          + (h[...,0:1])*tf.math.conj(r[...,2:3])
          + (h[...,1:2])*tf.math.conj(r[...,3:4])
          - tf.math.conj(h[...,6:7])*(r[...,4:5])
          - tf.math.conj(h[...,7:8])*(r[...,5:6])
          - (h[...,4:5])*tf.math.conj(r[...,6:7])
          - (h[...,5:6])*tf.math.conj(r[...,7:8]))
    
    # z4 = h5*.r1 + h6*.r2 + h7*.r3 + h8*.r4 + h1.r5* + h2.r6* + h3.r7* + h4.r8* 
    z4 = (  tf.math.conj(h[...,4:5])*(r[...,0:1])
          + tf.math.conj(h[...,5:6])*(r[...,1:2])
          + tf.math.conj(h[...,6:7])*(r[...,2:3])
          + tf.math.conj(h[...,7:8])*(r[...,3:4])
          + (h[...,0:1])*tf.math.conj(r[...,4:5])
          + (h[...,1:2])*tf.math.conj(r[...,5:6])
          + (h[...,2:3])*tf.math.conj(r[...,6:7])
          + (h[...,3:4])*tf.math.conj(r[...,7:8]))
    
    z = tf.concat((z1,z2,z3,z4),axis=-1) # Shape = [..., num_syms/8, M_r, 4]
    z = tf.reduce_sum(z,axis=-2) # Sum over M_r receive antennas (..., num_syms/8, 4)
    z = tf.reshape(z, (*r.shape[:-3],r.shape[-3]*4)) # (..., 4*num_syms/8)
 
    h_eq = tf.reduce_sum(tf.math.abs(h)**2,axis=-1,keepdims=True) # (..., num_syms/8, M_r, 1)
    h_eq = tf.concat((h_eq,h_eq,h_eq,h_eq),axis=-1) # Duplicating. Shape = (..., num_syms/8, M_r, 4)
    h_eq = tf.reduce_sum(h_eq,axis=-2) # (..., num_syms/8, 4)
    h_eq = tf.reshape(h_eq, (*r.shape[:-3],r.shape[-3]*4)) # (..., 4*num_syms/8)

    return z, h_eq


# Module test
if __name__ == "__main__":
    # Import sionna modules
    from sionna.utils import BinarySource
    from sionna.mapping import Mapper, Demapper
    from sionna.channel import AWGN
    from sionna.utils import ebnodb2no
    from sionna.utils.metrics import compute_ber
    ### GPU Handling
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        print(gpu)
    # Set memory growth
    if gpus: # if gpus is not empty
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    scheme = 'Alamaouti' # Alamouti, STBC434
    scheme = 'STBC434' # Alamouti, STBC434
    scheme = 'STBC848' # Alamouti, STBC434
    if scheme == 'Alamouti':
        # Simulation params
        my_complex_dtype = tf.complex128
        my_real_dtype = (tf.float64 if my_complex_dtype == tf.complex128 else tf.float32)
        batch_size = 128
        num_frames = 512
        num_symbols = 14  # must be even
        num_bits_per_symbol = 6  # QPSK
        ebno_db = 100.0
        if ebno_db>=100:
            noisy = False
        else:
            noisy = True
        no = tf.cast(ebnodb2no(ebno_db, num_bits_per_symbol, 1.0),my_real_dtype)
        N_t = 2
        assert N_t == 2
        M_r = 4

        # Create layer/modules
        binary_source = BinarySource(dtype=my_real_dtype)
        mapper = Mapper("qam", num_bits_per_symbol, dtype=my_complex_dtype)
        demapper = Demapper("maxlog", "qam", num_bits_per_symbol, hard_out=True, dtype=my_complex_dtype)
        add_noise = AWGN(dtype=my_complex_dtype)

        # Transmitter processing
        s = binary_source([batch_size, num_frames, num_symbols * num_bits_per_symbol])
        x = mapper(s)  # [..., num_syms]
        tx = alamouti_encode(x)  # [..., num_syms, N_t]
        tx = tf.reshape(tx, (*tx.shape[:-2], tx.shape[-2]//2, 2, N_t,1)) # [..., num_syms/2 , 2, N_t, 1]

        # Generate Rayleigh fading channel coefficients
        
        h_shape = [*tx.shape[:-3], 1, M_r, N_t] # [..., num_syms/2 , 1, M_r, N_t]
        h = tf.cast(tf.complex(tf.math.sqrt(0.25), 0.0),my_complex_dtype) * tf.complex(tf.random.normal(h_shape,dtype=my_real_dtype), tf.random.normal(h_shape,dtype=my_real_dtype))

        # Channel processing
        ry = tf.linalg.matmul(h, tx)  # [..., num_syms/2, 2, M_r,1]
        ry = ry[...,0] # [..., num_syms/2, 2, M_r]
        if noisy: ry = add_noise([ry, no])

        # Receiver processing
        yd, csi = alamouti_decode(ry, h[...,0,:,:])  # assuming perfect CSI. y.shape == csi.shape == [...,num_syms]

        # Demapping
        yd = yd / tf.cast(csi, yd.dtype)  # CSI scaling
        d = demapper([yd, no / csi])

        # Estimate BER
        avg_ber = compute_ber(d, s)
        print("Simulation of STBC in Rayleigh fading channel")
        print("EbNo: {:.1f}dB  BER: {:.2e}".format(ebno_db, avg_ber))
        print("tf.reduce_mean(tf.abs(x-yd))=",tf.sqrt(tf.reduce_mean(tf.abs(x-yd)**2)))

    elif scheme == 'STBC434':
        # Simulation params
        my_complex_dtype = tf.complex128
        my_real_dtype = (tf.float64 if my_complex_dtype == tf.complex128 else tf.float32)
        batch_size = 128
        num_frames = 512
        num_symbols = 15  # must be a multiple of 3
        num_bits_per_symbol = 4  # QPSK
        ebno_db = 100.0
        if ebno_db>=100:
            noisy = False
        else:
            noisy = True
        no = tf.cast(ebnodb2no(ebno_db, num_bits_per_symbol, 1.0),my_real_dtype)
        N_t = 4
        assert N_t == 4
        M_r = 2

        # Create layer/modules
        binary_source = BinarySource(dtype=my_real_dtype)
        mapper = Mapper("qam", num_bits_per_symbol, dtype=my_complex_dtype)
        demapper = Demapper("maxlog", "qam", num_bits_per_symbol, hard_out=True, dtype=my_complex_dtype)
        add_noise = AWGN(dtype=my_complex_dtype)

        # Transmitter processing
        s = binary_source([batch_size, num_frames, num_symbols * num_bits_per_symbol])
        x = mapper(s)  # [..., num_syms]
        tx = stbc_encode_434(x)  # [..., num_syms/3, N_t]
        tx = tf.reshape(tx, (*tx.shape[:-2], tx.shape[-2]//4, 4, N_t,1)) # [..., num_syms/3 , 4, N_t, 1]

        # Generate Rayleigh fading channel coefficients
        
        h_shape = [*tx.shape[:-3], 1, M_r, N_t] # [..., num_syms/4 , 1, M_r, N_t]
        h = tf.cast(tf.complex(tf.math.sqrt(0.25), 0.0),my_complex_dtype) *\
              tf.complex(tf.random.normal(h_shape,dtype=my_real_dtype), tf.random.normal(h_shape,dtype=my_real_dtype))

        # Channel processing
        ry = tf.linalg.matmul(h, tx)  # [..., num_syms/4, 4, M_r,1]
        ry = ry[...,0] # [..., num_syms/4, 4, M_r]
        if noisy: ry = add_noise([ry, no])

        # Receiver processing
        yd, csi = stbc_decode_434(ry, h[...,0,:,:])  # assuming perfect CSI. y.shape == csi.shape == [...,num_syms]

        # Demapping
        yd = yd / tf.cast(csi, yd.dtype)  # CSI scaling
        d = demapper([yd, no / csi])

        # Estimate BER
        avg_ber = compute_ber(d, s)
        print("Simulation of STBC in Rayleigh fading channel")
        print("EbNo: {:.1f}dB  BER: {:.2e}".format(ebno_db, avg_ber))
        print("tf.reduce_mean(tf.abs(x-yd))=",tf.sqrt(tf.reduce_mean(tf.abs(x-yd)**2)))
        
    elif scheme == 'STBC848':
        # Simulation params
        my_complex_dtype = tf.complex128
        my_real_dtype = (tf.float64 if my_complex_dtype == tf.complex128 else tf.float32)
        batch_size = 128
        num_frames = 512
        num_symbols = 16  # must be a multiple of 4
        num_bits_per_symbol = 4  # 16QAM
        ebno_db = 100.0
        if ebno_db>=100:
            noisy = False
        else:
            noisy = True
        no = tf.cast(ebnodb2no(ebno_db, num_bits_per_symbol, 1.0),my_real_dtype)
        N_t = 8
        assert N_t == 8
        M_r = 1

        # Create layer/modules
        binary_source = BinarySource(dtype=my_real_dtype)
        mapper = Mapper("qam", num_bits_per_symbol, dtype=my_complex_dtype)
        demapper = Demapper("maxlog", "qam", num_bits_per_symbol, hard_out=True, dtype=my_complex_dtype)
        add_noise = AWGN(dtype=my_complex_dtype)

        # Transmitter processing
        s = binary_source([batch_size, num_frames, num_symbols * num_bits_per_symbol])
        x = mapper(s)  # [..., num_syms]
        tx = stbc_encode_848(x)  # [..., 8*num_syms/4, N_t]
        tx = tf.reshape(tx, (*tx.shape[:-2], tx.shape[-2]//8, 8, N_t,1)) # [..., num_syms/4 , 8, N_t, 1]

        # Generate Rayleigh fading channel coefficients
        
        h_shape = [*tx.shape[:-3], 1, M_r, N_t] # [..., num_syms/4 , 1, M_r, N_t]
        h = tf.cast(tf.complex(tf.math.sqrt(1/N_t), 0.0),my_complex_dtype) *\
              tf.complex(tf.random.normal(h_shape,dtype=my_real_dtype), tf.random.normal(h_shape,dtype=my_real_dtype))

        # Channel processing
        ry = tf.linalg.matmul(h, tx)  # [..., num_syms/4, 8, M_r,1]
        ry = ry[...,0] # [..., num_syms/4, 8, M_r]
        if noisy: ry = add_noise([ry, no])

        # Receiver processing
        yd, csi = stbc_decode_848(ry, h[...,0,:,:])  # assuming perfect CSI. y.shape == csi.shape == [...,num_syms]

        # Demapping
        yd = yd / tf.cast(csi, yd.dtype)  # CSI scaling
        d = demapper([yd, no / csi])

        # Estimate BER
        avg_ber = compute_ber(d, s)
        print("Simulation of STBC in Rayleigh fading channel")
        print("EbNo: {:.1f}dB  BER: {:.2e}".format(ebno_db, avg_ber))
        print("tf.reduce_mean(tf.abs(x-yd))=",tf.sqrt(tf.reduce_mean(tf.abs(x-yd)**2)))
        

#%%
# beta = OSTBC(scheme='848')
# h_intended = tf.complex(tf.random.normal([10,3,8]),tf.random.normal([10,3,8]))
# h_not_intended = tf.complex(tf.random.normal([10,3,8]),tf.random.normal([10,3,8]))
# khar = beta.find_interference_power(h_intended,h_not_intended)
# pass
