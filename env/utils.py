import numpy as np

def spectral_radius(A):
    return np.max(np.abs(np.linalg.eigvals(A)))
