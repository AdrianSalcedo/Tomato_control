# coding=utf-8
import numpy as np

"""
    Here we reproduce the simulation of
    [1].
    
    The optimal control problem reads:
    \begin{equation}
    \int_{0}^T
        \left[
            B_1 I(t)
            + B_2 \left[\frac{R(t)}{K}\right]^m [u_1(t)]^2 + B_3 [u_2(t)]^2
        \right] dt,
        \qquad  m\geq 1,
    \end{equation}
    subject to
    \begin{equation}
        \begin{aligned}
            \frac{dS}{dt} &=
                \mu N
                - \beta \frac{S I}{N}
                - \mu \frac{N}{K} S - u_1(t) S,
            \\
            \frac{dI}{dt} &=
                \beta \frac{S I}{N}
                - (\gamma  + \mu) I
                - \mu \frac{N}{K} I
                - u_2(t) I,
        \\
        \frac{dR}{dt} &=
            \gamma I
            - \mu \frac{N}{K} R
            + u_1(t) S
            + u_2(t) I,
        \\
        S(0) &= S_0, \quad
        I(0) = I_0, \quad
        R(0) = R_0. \quad
        \end{aligned}
    \end{equation}

    [1] Elsa Schaefer and Holly Gaff. Optimal control applied to vaccination
    and treatment strategies for various epidemiological models. Mathematical
    Biosciences and Engineering, 6(3):469–492, jun 2009. ISSN 1551-0018.
    doi: 10.3934/mbe.2009.6.469.
    URL http://www.aimsciences.org/journals/displayArticles.jsp?paperID=4251.
"""


class OptimalControlProblem(object):
    def __init__(self, t_0=0.0, t_f=70.0, dynamics_dim=5, control_dim=1,
                 s_p_zero=0.9992, l_p_zero=0.0, i_p_zero=0.0008, s_v_zero=0.84, i_v_zero=0.16
                 ):
        # Parameters for the test example
        self.t_0 = t_0
        self.t_f = t_f
        self.dynamics_dim = dynamics_dim
        self.control_dim = control_dim
        #
        self.beta = 0.01
        self.a = 0.1
        self.b = 0.075
        self.Lambda = 0.003
        self.gamma = 0.06
        self.theta = 0.2
        self.mu = 0.3
        #
        
        #
        # initial conditions
        self.s_p_zero = s_p_zero
        self.l_p_zero = l_p_zero
        self.i_p_zero = i_p_zero
        self.s_v_zero = s_v_zero
        self.i_v_zero = i_v_zero

        self.n_p_whole = s_p_zero + l_p_zero + i_p_zero
        self.lambda_final = np.zeros([1, dynamics_dim])
        #
        # Functional Cost
        #
        self.A_1 = 1.0
        self.A_2 = 1.0
        self.A_3 = 1.0
        self.c_1 = 0.1
        self.u_1_lower = 0.03
        self.u_1_upper = 0.5
        #self.u_2_lower = 0.00
        #self.u_2_upper = 0.6
    
    def set_parameters(self, beta, a, b, Lambda, gamma, theta, mu,
                       A_1, A_2, A_3, c_1,
                       s_p_zero, l_p_zero, i_p_zero, s_v_zero, i_v_zero):
        #
        self.beta = beta
        self.a = a
        self.b = b
        self.Lambda = Lambda
        self.gamma = gamma
        self.theta = theta
        self.mu = mu

        self.A_1 = A_1
        self.A_2 = A_2
        self.A_3 = A_3
        self.c_1 = c_1
        self.s_p_zero = s_p_zero
        self.l_p_zero = l_p_zero
        self.i_p_zero = i_p_zero
        self.s_v_zero = s_v_zero
        self.i_v_zero = i_v_zero
    
    def g(self, x_k, u_k):
        beta = self.beta
        a = self.a
        b = self.b
        Lambda = self.Lambda
        gamma = self.gamma
        theta = self.theta
        mu = self.mu
        #A_1 = self.A_1
        #A_2 = self.A_2
        #A_3 = self.A_3
        #c_1 = self.c_1

        s_p = x_k[0, 0]
        l_p = x_k[0, 1]
        i_p = x_k[0, 2]
        s_v = x_k[0, 3]
        i_v = x_k[0, 4]

        n_p_whole = s_p + l_p + i_p
        u_1 = u_k[0]
        #u_2 = u_k[0, 1]


        rhs_s_p = beta * (l_p + i_p) - a * s_p * i_v + u_1 * (l_p + i_p)
        rhs_l_p = a * s_p * i_v - b * l_p - beta * l_p - u_1 * l_p
        rhs_i_p = b * l_p - beta * i_p - u_1 * i_p
        rhs_s_v = - Lambda * i_p * s_v - gamma * s_v + (1 - theta) * mu
        rhs_i_v = Lambda * i_p * s_v - gamma * i_v + theta * mu

        rhs_pop = np.array([rhs_s_p, rhs_l_p, rhs_i_p, rhs_s_v, rhs_i_v])
        self.n_p_whole = n_p_whole
        rhs_pop = rhs_pop.reshape([1, self.dynamics_dim])
        return rhs_pop
    
    def lambda_function(self, x_k, u_k, lambda_k):
        beta = self.beta
        a = self.a
        b = self.b
        Lambda = self.Lambda
        gamma = self.gamma
        theta = self.theta
        mu = self.mu
        A_1 = self.A_1
        A_2 = self.A_2
        A_3 = self.A_3
        c_1 = self.c_1

        s_p = x_k[0, 0]
        l_p = x_k[0, 1]
        i_p = x_k[0, 2]
        s_v = x_k[0, 3]
        i_v = x_k[0, 4]

        n_p_whole = s_p + l_p + i_p
        u_1 = u_k
        #u_2 = u_k[0, 1]

        lambda_1 = lambda_k[0, 0]
        lambda_2 = lambda_k[0, 1]
        lambda_3 = lambda_k[0, 2]
        lambda_4 = lambda_k[0, 3]
        lambda_5 = lambda_k[0, 4]

        rhs_l_1 = a * i_v * (lambda_1 - lambda_2)

        rhs_l_2 = - A_2 + beta * (lambda_2 - lambda_1) + b * (lambda_2 - lambda_3) + u_1 * (lambda_2 - lambda_1)

        rhs_l_3 = - A_1 + (beta + u_1) * (lambda_3 - lambda_1) + Lambda * s_v * (lambda_4 - lambda_5)

        rhs_l_4 = Lambda * i_p * (lambda_4 - lambda_5) + gamma * lambda_4

        rhs_l_5 = -A_3 + a * s_p * (lambda_1 - lambda_2) + Lambda * s_v * (lambda_4 - lambda_5) + gamma * lambda_5

        #
        #
        #
        rhs_l = np.array([rhs_l_1, rhs_l_2, rhs_l_3, rhs_l_4, rhs_l_5])
        rhs_l = rhs_l.reshape([1, 5])
        return rhs_l
    
    def optimality_condition(self, x_k, u_k, lambda_k, n_max):
        u_1_lower = self.u_1_lower
        #u_2_lower = self.u_2_lower
        u_1_upper = self.u_1_upper
        #u_2_upper = self.u_2_upper
        c_1 = self.c_1

        #
        s_p = x_k[:, 0]
        l_p = x_k[:, 1]
        i_p = x_k[:, 2]
        s_v = x_k[:, 3]
        i_v = x_k[:, 4]

        lambda_1 = lambda_k[:, 0]
        lambda_2 = lambda_k[:, 1]
        lambda_3 = lambda_k[:, 2]
        #lambda_4 = lambda_k[:, 3]
        #lambda_5 = lambda_k[:, 4]
        
        aux_1 = (l_p * (lambda_2 - lambda_1) + i_p * (lambda_3 - lambda_1)) / (2 * c_1)
        #aux_2 = i * (lambda_2 - lambda_3) / (2 * b_3)
        
        positive_part_1 = np.max([u_1_lower * np.ones(n_max), aux_1], axis=0)
        #positive_part_2 = np.max([u_2_lower * np.ones(n_max), aux_2], axis=0)
        
        u_aster_1 = np.min([positive_part_1, u_1_upper * np.ones(n_max)],
                           axis=0)
        #u_aster_2 = np.min([positive_part_2, u_2_upper * np.ones(n_max)],
                           #axis=0)
        
        u_aster = np.zeros([n_max, 1])
        u_aster[:, 0] = u_aster_1
        #u_aster[:, 1] = u_aster_2
        return u_aster
