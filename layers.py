import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine layer.

    Inputs:
    - x: A numpy array containing input data, of shape (N, H)
    - w: A numpy array of weights, of shape (H, T)
    - b: A numpy array of biases, of shape (T,)

    Returns a tuple of:
    - out: output, of shape (N, T)
    - cache: (x, w, b)
    """
    out = x.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, T)
    - cache: Tuple of:
      - x: Input data, of shape (N, H)
      - w: Weights, of shape (H, T)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, H)
    - dw: Gradient with respect to w, of shape (H, T)
    - db: Gradient with respect to b, of shape (T,)
    """
    x, w, b = cache
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.T.dot(dout)
    db = dout.sum(axis=0)
    return dx, dw, db

def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h = np.tanh(x.dot(Wx) + prev_h.dot(Wh) + b)
    cache = (x, prev_h, Wx, Wh, next_h)
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    (x, prev_h, Wx, Wh, next_h) = cache
    dout = (1 - next_h ** 2) * dnext_h
    dx = dout.dot(Wx.T)
    dprev_h = dout.dot(Wh.T)
    dWx = x.T.dot(dout)
    dWh = prev_h.T.dot(dout)
    db = dout.sum(axis=0)
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    N, T, D = x.shape
    N, H = h0.shape
    h, prev_h, cache = np.zeros((N, T, H)), h0, []
    
    for t in range(T):
        h[:, t, :], cache_step = rnn_step_forward(x[:, t, :], prev_h, Wx, Wh, b)
        cache.append(cache_step)
        prev_h = h[:, t, :]

    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    (x, prev_h, Wx, Wh, next_h) = cache[0]
    N, D = x.shape
    N, T, H = dh.shape
    dnext_h = np.zeros((N, H))
    dx = np.zeros((N, T, D))
    dWx, dWh, db = np.zeros((D, H)), np.zeros((H, H)), np.zeros(H)
    
    for t in reversed(range(T)):
        dx_t, dprev_h, dWx_t, dWh_t, db_t = rnn_step_backward(
            dh[:, t, :] + dnext_h, cache.pop())
        dx[:, t, :] = dx_t
        dWx += dWx_t
        dWh += dWh_t
        db += db_t
        dnext_h = dprev_h
    dh0 = dnext_h

    return dx, dh0, dWx, dWh, db


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    (N, D), (N, H) = x.shape, prev_h.shape
    a = x.dot(Wx) + prev_h.dot(Wh) + b
    a_i, a_f, a_o, a_g = np.split(a, 4, axis=1)
    i, f, o, g = sigmoid(a_i), sigmoid(a_f), sigmoid(a_o), np.tanh(a_g)
    next_c = f * prev_c + i * g
    tanh_c = np.tanh(next_c)
    next_h = o * tanh_c
    
    cache = (x, prev_h, prev_c, Wx, Wh, i, f, o, g, tanh_c)
    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    (N, H) = dnext_h.shape
    (x, prev_h, prev_c, Wx, Wh, i, f, o, g, tanh_c) = cache
    
    dnext_c += dnext_h * o * (1 - tanh_c ** 2)
    dprev_c = dnext_c * f
    
    da_i = dnext_c * g * i * (1 - i)
    da_f = dnext_c * prev_c * f * (1 - f)
    da_o = dnext_h * tanh_c * o * (1 - o)
    da_g = dnext_c * i * (1 - g ** 2)
    da = np.concatenate((da_i, da_f, da_o, da_g), axis=1)
    
    dx = da.dot(Wx.T)
    dprev_h = da.dot(Wh.T)
    dWx = x.T.dot(da)
    dWh = prev_h.T.dot(da)
    db = da.sum(axis=0)

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    (N, T, D), (N, H) = x.shape, h0.shape
    h, prev_h, prev_c, cache = np.zeros((N, T, H)), h0, 0, []
    
    for t in range(T):
        h[:, t, :], next_c, cache_step = lstm_step_forward(
            x[:, t, :], prev_h, prev_c, Wx, Wh, b)
        cache.append(cache_step)
        prev_h, prev_c = h[:, t, :], next_c

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    x, *_ = cache[0]
    _, D = x.shape
    (N, T, H) = dh.shape
    
    dx = np.zeros((N, T, D))
    dnext_c, dnext_h = np.zeros((N, H)), np.zeros((N, H))
    dWx, dWh, db = np.zeros((D, 4*H)), np.zeros((H, 4*H)), np.zeros(4*H)
    
    for i in reversed(range(T)):
        dx_t, dprev_h, dprev_c, dWx_t, dWh_t, db_t = lstm_step_backward(
            dnext_h + dh[:, i, :], dnext_c, cache.pop())
        dx[:, i, :] = dx_t
        dnext_h = dprev_h
        dnext_c = dprev_c
        dWx += dWx_t
        dWh += dWh_t
        db += db_t
    dh0 = dprev_h
    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_dot_forward(x, w, b):
    """
    Forward pass for a temporal dot product layer. 

    Inputs:
    - x: Input data of shape (N, T, H)
    - w: Weights of shape (H, )
    - b: Biases; scalar

    Returns a tuple of:
    - out: Output data of shape (N, T)
    - cache: Values needed for the backward pass
    """
    N, T, H = x.shape
    out = x.reshape(N * T, H).dot(w).reshape(N, T) + b
    out = np.tanh(out)
    cache = (x, w, out)
    return out, cache


def temporal_dot_backward(dout, cache):
    """
    Backward pass for temporal dot product layer.

    Input:
    - dout: Upstream gradients of shape (N, T)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, H)
    - dw: Gradient of weights, of shape (H, )
    - db: Gradient of biases, scalar
    """
    x, w, out = cache
    N, T, H = x.shape

    dout = dout * (1 - out ** 2)
    dx = dout.reshape(N * T, 1).dot(w.reshape(1, H)).reshape(N, T, H)
    dw = dout.reshape(1, N * T).dot(x.reshape(N * T, H)).reshape(H)
    db = dout.sum()

    return dx, dw, db


def softmax_forward(x):
    """
    Forward pass for a softmax layer after rnn. Note that it is a layer, not a
    loss or an activation function.  

    Inputs:
    - x: Input data of shape (N, T)

    Returns a tuple of:
    - out: Output data of shape (N, T)
    - cache: Values needed for the backward pass
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    cache = (probs,)
    return probs, cache


def softmax_backward(dout, cache):
    """
    Backward pass for softmax backward layer.

    Inputs:
    - dout: Upstream gradients of shape (N, T)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T)
    """
    N, T = dout.shape
    probs,  = cache
    dx = np.zeros(dout.shape)
    indics = np.arange(T)
    for i in range(N):
        p = probs[i].reshape(-1, 1)
        A = np.diag(probs[i]) - p.dot(p.T)
        dx[i] = dout[i].dot(A)
    return dx


def attension_forward(x, w, b):
    """
    Forward pass for a attension layer after rnn. A softmax makes it
    look like probability.

    Inputs:
    - x: Last hidden layer output of rnn, of shape (N, H).
    - w: Weights of shape (H, T)
    - b: Biases of shape (T, )

    Returns a tuple of:
    - out: Attension vector, of shape (N, T)
    - cache: Values needed for the backward pass
    """
    out, cache_aff = affine_forward(x, w, b)
    out, cache_smax = softmax_forward(out)
    cache = (x, w, cache_aff, cache_smax)
    return out, cache


def attension_backward(dout, cache):
    """
    Backward pass for a attension layer.

    Inputs:
    - dout: Upstream gradients; scalar
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T)
    - dw: Gradient of weights, of shape (H, T)
    - db: Gradient of biases, of shape (T, )
    """
    (x, w, cache_aff, cache_smax) = cache
    dout = softmax_backward(dout, cache_smax)
    dx, dw, db = affine_backward(dout, cache_aff)
    return dx, dw, db


def temporal_leastsquare_loss(x, y):
    """
    Computes the loss and gradient for temporal least square regression.

    Inputs:
    - x: Input data, of shape (N, T) where x[i, j](j < T-1) is the predict in 
      the jth second for the ith flight, where x[i, T-1] is the predict of max 
      vrtg after touch down for the ith flight.
    - y: Vector of real VRTG, of shape (N, T).

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N, T = x.shape
    loss = np.sum((x - y) ** 2 / N / 2)
    dout = (x - y) / N
    return loss, dout

def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
