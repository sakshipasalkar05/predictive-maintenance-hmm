import numpy as np
from scipy.special import logsumexp


def get_params(model):
    """Extract log start probs, log transition matrix, means and variances from a fitted GaussianHMM."""
    covars = model.covars_
    var = np.array([np.diag(c) for c in covars]) if covars.ndim == 3 else covars
    with np.errstate(divide="ignore"):
        logpi = np.log(model.startprob_)
        logA = np.log(model.transmat_)
    return logpi, logA, model.means_, var


def log_emission(X, means, var):
    """log P(x_t | state i) for diagonal Gaussians. Returns array (T, n_states)."""
    T, n = X.shape[0], means.shape[0]
    out = np.zeros((T, n))
    for i in range(n):
        out[:, i] = -0.5 * np.sum(
            np.log(2 * np.pi * var[i]) + (X - means[i]) ** 2 / var[i], axis=1
        )
    return out


def forward(logB, logpi, logA):
    T, n = logB.shape
    la = np.zeros((T, n))
    la[0] = logpi + logB[0]
    for t in range(1, T):
        la[t] = logB[t] + logsumexp(la[t - 1][:, None] + logA, axis=0)
    return la


def backward(logB, logA):
    T, n = logB.shape
    lb = np.zeros((T, n))
    for t in range(T - 2, -1, -1):
        lb[t] = logsumexp(logA + logB[t + 1] + lb[t + 1], axis=1)
    return lb


def normalize_rows(lx):
    return np.exp(lx - logsumexp(lx, axis=1, keepdims=True))


def viterbi(logB, logpi, logA):
    T, n = logB.shape
    delta = np.zeros((T, n))
    psi = np.zeros((T, n), dtype=int)
    delta[0] = logpi + logB[0]
    for t in range(1, T):
        scores = delta[t - 1][:, None] + logA
        psi[t] = scores.argmax(axis=0)
        delta[t] = logB[t] + scores.max(axis=0)
    path = np.zeros(T, dtype=int)
    path[-1] = delta[-1].argmax()
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path