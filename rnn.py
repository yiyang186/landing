import numpy as np

from configure import *
from layers import *


class RNN(object):
    """
    A RNN produces captions from image features using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the RNN.
    """

    def __init__(self, input_dim=12, time_dim=60, hidden_dim=32, 
                 cell_type='rnn', dtype=np.float32):
        """
        Construct a new RNN instance.

        Inputs:
        - input_dim: Dimension D of flight data vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - time_dim: Dimension T for the time length in attension layers.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.params = {}

        # Initialize parameters for the RNN
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = np.random.randn(input_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(input_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)

        # Initialize vrtg weights
        self.params['W_vrtg'] = np.random.randn(hidden_dim)
        self.params['W_vrtg'] /= np.sqrt(hidden_dim)
        self.params['b_vrtg'] = 0.0

        # Initialize hard landing weights
        self.params['W_hard'] = np.random.randn(hidden_dim)
        self.params['W_hard'] /= np.sqrt(hidden_dim)
        self.params['b_hard'] = 0.0

        # Initialize attension weights
        self.params['W_atts'] = np.random.randn(hidden_dim, time_dim)
        self.params['W_atts'] /= time_dim
        self.params['b_atts'] = np.zeros(time_dim)

        # Initialize hidden layes 0
        self.h0 = np.zeros(hidden_dim)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)


    def loss(self, data, target):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss and gradients on all parameters.

        Inputs:
        - data: iuput data
        - target: Max VRTG after touch down for every flights

        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        """
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vrtg, b_vrtg = self.params['W_vrtg'], self.params['b_vrtg']
        W_hard, b_hard = self.params['W_hard'], self.params['b_hard']
        W_atts, b_atts = self.params['W_atts'], self.params['b_atts']
        loss, grads = 0.0, {}

        out, cache_rnn = eval(self.cell_type + '_forward')(
            data, self.h0, Wx, Wh, b)                                #(N, T, H)
        vrtg, cache_vrtg = temporal_dot_forward(out, W_vrtg, b_vrtg) #(N, T)
        hard, cache_hard = temporal_dot_forward(out, W_hard, b_hard) #(N, T)
        atts, cache_atts = attension_forward(out[:, -1, :], W_atts, b_atts) #(N, T)
        vrtg[:, -1] += atts * hard                   
        y = np.c_[data[:, 1:, INDEX_VRTG], target]
        loss, dout = temporal_leastsquare_loss(vrtg, y)
        
        

        return loss, grads


    def sample(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the <START>
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Inputs:
        - features: Array of input image features of shape (N, D).
        - max_length: Maximum length T of generated captions.

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        vocab_size = W_embed.shape[0]
        word = np.zeros((N, vocab_size))
        word[:, self._start] = 1
        
        prev_h = features.dot(W_proj) + b_proj # (N, hidden_dim)
        if self.cell_type == 'lstm':
            prev_c = np.zeros_like(prev_h)
        
        for i in range(max_length):
            x = word.dot(W_embed) # (N, wordvec_dim)
            
            if self.cell_type == 'rnn':
                next_h, _ = rnn_step_forward(x, prev_h, Wx, Wh, b) # (N, hidden_dim) 
                
            if self.cell_type == 'lstm':
                next_h, next_c, _ = lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)
                prev_c = next_c
                
            word = next_h.dot(W_vocab) + b_vocab # (N, vocab_size)
            captions[:, i] = word.argmax(axis=1) # (N, max_length)

            prev_h = next_h
            word -= word
            word[np.arange(N), captions[:, i]] = 1.0

        return captions
