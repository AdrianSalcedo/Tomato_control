import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def rhs(y, t_zero):
    s_p = y[0]
    l_p = y[1]
    i_p = y[2]
    s_v = y[3]
    i_v = y[4]

    s_p_prime = beta * (l_p + i_p) - a * s_p * i_v
    l_p_prime = a * s_p * i_v - b * l_p - beta * l_p
    i_p_prime = b * l_p - beta * i_p
    s_v_prime = - Lambda * s_v * i_p - g * s_v + (1 - theta) * mu
    i_v_prime = Lambda * s_v * i_p - g * i_v + theta * mu
    rhs_np_array = np.array([s_p_prime, l_p_prime, i_p_prime, s_v_prime, i_v_prime])
    return (rhs_np_array)


beta = 0.01
a = 0.1
g = 0.06
mu = 0.3
theta = 0.4
b = 0.075
Lambda = 0.003

y_zero = np.array([0.998999, 0.001, 0.000001, 0.92, 0.08])
t = np.linspace(0, 70, 1000)
sol = odeint(rhs, y_zero, t)
plt.plot(t, sol[:, 2], 'b')#, label='$I_p$')
plt.plot([0.6+7, 1.1+13, 1.7+21, 2.4+28,2.8+35,3.4+42], [0.005, 0.007, 0.008, 0.02,.17+0.4,0.031+0.8], 'ro')
#plt.plot(t, sol[:, 4], 'r', label='$I_v$')
#plt.legend(loc='best')
plt.xlabel('$t$')
plt.ylabel('proporción de plantas infectadas $I_p$')
plt.ylim(-0.05,1)
plt.xlim(0,70)
plt.grid()
plt.show()
