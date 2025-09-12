import numpy as np

class multimode_esn_pred:
    """Reservoir computer operating on 2-mode channel tensors.

    The reservoir state ``S`` has dimensions ``[d_t, d_r]`` corresponding to
    reduced representations of resource blocks (RBs), transmit antennas, and
    receive antennas.  Inputs ``Y`` are tensors with dimensions
    ``[N_t, N_r]``.
    """

    def __init__(self,
                 N_f,  # number of RBs / subcarriers
                 N_t,  # number of transmit antennas
                 N_r,  # number of receive antennas
                 d_t,  # Tx state dimension
                 d_r,  # Rx state dimension
                 window_len=1,  # optional window length along RB mode
                 dtype=np.complex64,
                 debug=False):

        # Store tensor dimensions
        self.N_f = N_f
        self.N_t = N_t
        self.N_r = N_r
        # Store state dimensions
        self.d_t = d_t
        self.d_r = d_r
        # Input window length and activation
        self.window_len = window_len
        self.dtype = dtype

        self.debug = debug

        self.reset_state()

    
    def init_weights(self):
        
        self.W = self.sparse_mat(self.N_n)
        self.W_in = 2 * (self.RS.rand(self.N_n, self.N_in) - 0.5)
        self.W_tran = np.concatenate([self.W, self.W_in], axis=1)

    def sparse_mat(self, m):
        if self.type == 'real':
            W = self.RS.rand(m, m) - 0.5
            W[self.RS.rand(*W.shape) < self.sparsity] = 0
        else:
            W = 2*(self.RS.rand(m, m) - 0.5) + 2j*(self.RS.rand(m, m) - 0.5)
            W[self.RS.rand(*W.shape) < self.sparsity] = 0+1j*0
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        W = W * (self.spectral_radius / radius)
        return W
