from forward_backward_sweep import ForwardBackwardSweep
from matplotlib import rcParams

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
A_1 = .5
A_2 = 0.3
A_3 = 0.0

c_3 = 0.1

<<<<<<< HEAD
name_file_1 = 'figure_1_tomato_one_control.eps'
=======
name_file_1 = 'figure_1_tomato_one_control.pdf'
>>>>>>> 8b296fbe19fe88c04da51ab9643ccb674d6ea233

#

fbsm = ForwardBackwardSweep()
fbsm.set_parameters(beta, a, b, psi, gamma, theta, mu,
                       A_1, A_2, A_3, c_3,
                       s_p_zero, l_p_zero, i_p_zero, s_v_zero, i_v_zero)

t = fbsm.t
x_wc = fbsm.runge_kutta_forward(fbsm.u)
#
[x, lambda_, u] = fbsm.forward_backward_sweep()

mpl.style.use('ggplot')
# plt.ion()
# n_whole = fbsm.n_whole
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

<<<<<<< HEAD
ax2 = plt.subplot2grid((3, 2), (0, 1), rowspan=3)
ax2.plot(t, u[:, 0],
         label="$u_3(t)$ ",
         color='orange')
ax2.set_ylabel(r'$u_3(t): Fumigation$')
=======
ax2 = plt.subplot2grid((3, 2), (1, 1))
ax2.plot(t, u[:, 0],
         label="$u_3(t)$ : Fumigation",
         color='orange')
ax2.set_ylabel(r'$u_3(t)$')
>>>>>>> 8b296fbe19fe88c04da51ab9643ccb674d6ea233
ax2.set_xlabel(r'Time(days)')

plt.tight_layout()
#
fig = mpl.pyplot.gcf()
fig.set_size_inches(5.5, 5.5 / 1.618)
fig.savefig(name_file_1,
            # additional_artists=art,
            bbox_inches="tight")
