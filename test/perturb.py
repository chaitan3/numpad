import numpy as np

def perturbation((xc, yc), pf):
    #euler
    p = 1e-4*np.exp(-(xc+5)**2/2 - (yc-0)**2/2)
    #navierstokes
#p = 1e-3*np.exp(-(xc+10)**2/2 - (yc-15)**2/2)
    pf[0] = p
