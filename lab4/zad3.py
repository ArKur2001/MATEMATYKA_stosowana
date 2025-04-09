import numpy as np
import matplotlib.pyplot as plt


M = 3



def S_narow(x):
    return 1

def S_wide(x):
    if x > 1.4 and x < 1.6:
        return 10
    else:
        return 1









def h(n):
    return 



fig = plt.figure()
S1 = fig.add_subplot(221)
S1.set_title("Weighting function S_1, narrow transition band")
S1.set_ylim(0, 2)
S1.plot(np.arange(0, np.pi, 0.001), [S_narow(x) for x in np.arange(0, np.pi, 0.001)])

S2 = fig.add_subplot(223)
S2.set_title("Weighting function S_2, wide transition band")
S2.set_ylim(0, 2)
S2.plot(np.arange(0, np.pi, 0.001), [S_wide(x) for x in np.arange(0, np.pi, 0.001)])







plt.show()