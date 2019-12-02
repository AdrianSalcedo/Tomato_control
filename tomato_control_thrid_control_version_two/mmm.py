from forward_backward_sweep import ForwardBackwardSweep
from matplotlib import rcParams
import pandas as pd

# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Tahoma']
params = {
    'figure.titlesize': 10,
    'axes.titlesize':   10,
    'axes.labelsize':   10,
    'font.size':        10,
    'legend.fontsize':  8,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'text.usetex':      True
}
rcParams.update(params)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np

#
#
#
beta = 0.01
a = 0.1
b = 0.075
psi = 0.003
gamma = 0.06
theta = 0.2
mu = 0.3
#
#
# Initial conditions
s_p_zero = 0.9992
l_p_zero = 0.0
i_p_zero = 0.0008
s_v_zero = 0.84
i_v_zero = 0.16
# Functional Cost
A_1 = 0.5
A_2 = 0.2
A_3 = 1.2
c_1 = 1.0
c_2 = 1.1
c_3 = 1.2

name_file_1 = 'figure_1_tomato_controls.pdf'
#

fbsm = ForwardBackwardSweep()
fbsm.set_parameters(beta, a, b, psi, gamma, theta, mu,
                       A_1, A_2, A_3, c_1, c_2, c_3,
                       s_p_zero, l_p_zero, i_p_zero, s_v_zero, i_v_zero)

t = fbsm.t
x_wc = fbsm.runge_kutta_forward(fbsm.u)
#
[x, lambda_, u] = fbsm.forward_backward_sweep()
# '''TODO: save x,lambda_,u,x_wc'''

mpl.style.use('ggplot')
ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=3)
ax1.plot(t, x_wc[:, 2],
         label="Without control",
         color='darkgreen'
         )
ax1.plot(t, x[:, 2],
         label="Optimal controlled",
         color='orange')
ax1.set_ylabel(r'Infected plants ratio $I_p$')
ax1.set_xlabel(r'Time (days)')
ax1.legend(loc=0)

ax2 = plt.subplot2grid((3, 2), (0, 1))
ax2.plot(t, u[:, 0],
         label="$u_1(t)$ : Remove latent plant",
         color='orange')
ax2.set_ylabel(r'$u_1(t)$')

ax3 = plt.subplot2grid((3, 2), (1, 1))
ax3.plot(t, u[:, 1],
         label="$u_2(t)$ : Remove infected plant",
         color='darkgreen')
ax3.set_ylabel(r'$u_2(t)$')

ax4 = plt.subplot2grid((3, 2), (2, 1))
ax4.plot(t, u[:, 2],
         label="$u_3(t)$ : insecticide",
         color='darkred')
ax4.set_ylabel(r'$u_3(t)$')
ax4.set_xlabel(r'Time(days)')
#ax2.set_ylabel(r'Controls')
#ax2.legend(loc=0)
#
plt.tight_layout()
#
fig = mpl.pyplot.gcf()
fig.set_size_inches(5.5, 5.5 / 1.618)
fig.savefig(name_file_1,
            # additional_artists=art,
            bbox_inches="tight")
#######################################################################################################################
plt.figure()
Cost_value = (A_1 * x[:, 2] + A_2 * x[:, 1]+ A_3 * x[:, 4]+ c_1 * u[:, 0] ** 2 + c_2 * u[:, 1] ** 2 + c_3 * u[:, 2] ** 2) * (365 / 10000)

Int_Cost_value = np.cumsum(Cost_value)
#print(Int_Cost_value[len(t)])
#np.save('time.npy',t)
np.save('three_control_cost.npy',Int_Cost_value)
plt.plot(t,Int_Cost_value)
plt.show()