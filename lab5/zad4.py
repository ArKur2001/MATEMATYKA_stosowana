import numpy as np
import matplotlib.pyplot as plt
import math



M = 20
N = 1000

w = math.pi / 2


n_lpf = np.arange(-M, M+1) # discrete time vector
x = np.random.normal(0, 1, (2*M)+1) # gaussian noise

print(n_lpf)


def LPF(n, w):
    if n == 0:
        return w / math.pi
    else:
        return math.sin(w * n) / (math.pi * n)
    
def lms_filter(x, d, mu=0.01):
    N = len(x)
    h = np.zeros((M+2)+1)
    y = np.zeros(N)
    e = np.zeros(N)
    h_history = np.zeros((N, M))
    
    for n in range(M, N):
        x_window = x[n:n-M:-1]
        y[n] = np.dot(h, x_window)
        e[n] = d[n] - y[n]
        h += mu * e[n] * x_window
        h_history[n] = h
        
    return y, e, h_history

def rls_filter(x, d, filter_order, delta=0.01, lambda_=0.99):
    N = len(x)
    h = np.zeros(filter_order)
    P = np.eye(filter_order) / delta
    y = np.zeros(N)
    e = np.zeros(N)
    h_history = np.zeros((N, filter_order))
    
    for n in range(filter_order, N):
        x_window = x[n:n-filter_order:-1]
        y[n] = np.dot(h, x_window)
        e[n] = d[n] - y[n]
        
        # RLS update equations
        k = np.dot(P, x_window) / (lambda_ + np.dot(x_window, np.dot(P, x_window)))
        h += k * e[n]
        P = (P - np.outer(k, np.dot(x_window, P))) / lambda_
        
        h_history[n] = h
        
    return y, e, h_history



h_lpf = [LPF(n, w) for n in n_lpf] # impulse response of LTI system

d = np.convolve(x, h_lpf)

y, e, h_lms = lms_filter(x, d, mu=0.01)



plt.subplot(2, 2, 1)
plt.plot(n_lpf, h_lpf, 'r')
plt.plot(n_lpf, h_lms, 'g')
plt.subplot(2, 2, 2)


plt.show()