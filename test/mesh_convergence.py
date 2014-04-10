from matplotlib import pyplot as plt
from numpy import *

a = array([[0.0403873274904, 0.0614442631391, 0.122696726814],
           [9.05400518295, 18.0371987378, 36.3219284436],
           [0.814753061745, 0.889954747992, 1.47061463376],
           [886.182897538, 1697.05833566, 3356.85815468]
    ])

n = [50, 100, 200]
plt.loglog(n, a[0]/n, label='w1a')
plt.loglog(n, a[1]/n, label='w2a')
plt.loglog(n, a[2]/n, label='w3a')
plt.loglog(n, a[3]/n, label='w4a')
plt.legend()
plt.xlabel('n')
plt.ylabel('frobenius norm')
plt.show()
