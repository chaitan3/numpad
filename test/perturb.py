import numpy as np

def perturbation((xc, yc), pf):
    p = 1e-3*np.exp(-(xc+5)**2/2 - (yc-0)**2/2)
    pf[0] = p
