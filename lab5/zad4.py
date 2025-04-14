import numpy as np
import matplotlib.pyplot as plt
import math



M = 20000
w = math.pi / 2


n = np.arange(-M, M) # discrete time vector
x = np.random.normal(0, 1, 2*M) # gaussian noise

def LTI(t, w):
    for n in t:
        if n == 0:
            return w / math.pi
        else:
            return math.sin(w * n) / (math.pi * n)
    

d = np.convolve(x, LTI(n, w), mode='same') # convolution of noise with LTI system

plt.subplot(2, 2, 1)
plt.plot(n, x, 'r')
plt.subplot(2, 2, 2)
plt.plot(n, d, 'r')


plt.show()