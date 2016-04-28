import numpy as np

def gaussian_log_pdf(mu, s):
    def lpdf(x):
        normalizing_constant = -(np.log(2 * np.pi) / 2) - np.log(s)
        return normalizing_constant - ((x - mu)**2 / (2 * s**2))
    return lpdf
