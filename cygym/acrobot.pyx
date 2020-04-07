#!python

# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: nonecheck=False

"""
Credit:
    
This code has been adapted from the original OpenAI release:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/acrobot.py

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann <cdann@cdann.de>"

# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py

"""

## Loading packages
from gym.spaces import Box, Discrete
from gym.utils import seeding
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sin, cos, M_PI


# Defining our numpy types
ctypedef np.float_t np_float_t


cdef class CyAcrobotEnv():


    cdef:
        float dt
        float LINK_LENGTH_1
        float LINK_LENGTH_2
        float LINK_MASS_1
        float LINK_MASS_2
        float LINK_COM_POS_1
        float LINK_COM_POS_2
        float LINK_MOI
        float MAX_VEL_1
        float MAX_VEL_2
        tuple AVAIL_TORQUE
        float torque_noise_max
        
        readonly object observation_space
        readonly object action_space
        np_float_t[:] state
        
    cdef object np_random
    metadata = {}
    reward_range = (-float('inf'), float('inf'))
    spec = None
    def unwrapped(self): return self

    
    cdef inline void seed(self, seed=1):
        if seed == 0:
            self.np_random, _ = seeding.np_random()
        else:
            self.np_random, _ = seeding.np_random(seed)
        

    def __cinit__(self):
        self.dt = 0.2
        self.LINK_LENGTH_1 = 1.0
        self.LINK_LENGTH_2 = 1.0
        self.LINK_MASS_1 = 1.0
        self.LINK_MASS_2 = 1.0
        self.LINK_COM_POS_1 = 0.5
        self.LINK_COM_POS_2 = 0.5
        self.LINK_MOI = 1.0
        self.MAX_VEL_1 = 4*M_PI
        self.MAX_VEL_2 =9*M_PI
        self.AVAIL_TORQUE = (1., 0., 1.)
        self.torque_noise_max = 0.

        self.observation_space = Box(low=np.array((-1.0, -1.0, -1.0, -1.0, -self.MAX_VEL_1, -self.MAX_VEL_2)),
                                     high=np.array((1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2)))
        
        self.action_space = Discrete(3)
        self.seed()

    cpdef np.ndarray[dtype=np_float_t, ndim=1] reset(self):
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4,))
        return np.array(self._get_ob())

    cpdef tuple step(self, int a):
        cdef:
            np_float_t[:] s = self.state
            float torque = self.AVAIL_TORQUE[a]
            np_float_t[:] s_augmented
            np_float_t[:,:] nsa
            np_float_t[:] ns
            bint terminal
            float reward
            int nsa_ind

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.array((s[0], s[1], s[2], s[3], torque))

        nsa = rk4(self._dsdt, s_augmented, (0, self.dt))
        # only care about final timestep of integration returned by integrator
        
        nsa_ind = nsa.shape[0]-1
        ns = np.array((wrap(nsa[nsa_ind][0], -M_PI, M_PI),
                       wrap(nsa[nsa_ind][1], -M_PI, M_PI),
                       bound(nsa[nsa_ind][2], -self.MAX_VEL_1, self.MAX_VEL_1),
                       bound(nsa[nsa_ind][3], -self.MAX_VEL_2, self.MAX_VEL_2)))

        self.state = ns
        terminal = self._terminal()
        reward = -1. if not terminal else 0.
        return (np.array(self._get_ob()), reward, terminal, {})

    cpdef tuple _get_ob(self):
        cdef np_float_t[:] s = self.state
        return (cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3])

    cpdef bint _terminal(self):
        cdef np_float_t[:] s = self.state
        return bool(-cos(s[0]) - cos(s[1] + s[0]) > 1.)

    cpdef np_float_t[:] _dsdt(self, np_float_t[:] s_augmented):
        cdef:
            float m1 = self.LINK_MASS_1
            float m2 = self.LINK_MASS_2
            float l1 = self.LINK_LENGTH_1
            float lc1 = self.LINK_COM_POS_1
            float lc2 = self.LINK_COM_POS_2
            float I1 = self.LINK_MOI
            float I2 = self.LINK_MOI
            float g = 9.8
            float a = s_augmented[s_augmented.shape[0]-1]
            np_float_t[:] s = s_augmented[:s_augmented.shape[0]-1]
            float theta1 = s[0]
            float theta2 = s[1]
            float dtheta1 = s[2]
            float dtheta2 = s[3]
            float d1 = m1 * lc1**2 + m2 * \
                      (l1**2 + lc2**2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
            float d2 = m2 * (lc2**2 + l1 * lc2 * cos(theta2)) + I2
            float phi2 = m2 * lc2 * g * cos(theta1 + theta2 - M_PI / 2.)
            float phi1 = - m2 * l1 * lc2 * dtheta2**2 * sin(theta2) \
                         - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)  \
                         + (m1 * lc1 + m2 * l1) * g * cos(theta1 - M_PI / 2) + phi2
                
            float ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * sin(theta2) - phi2) \
                             / (m2 * lc2**2 + I2 - d2**2 / d1)
            float ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return np.array((dtheta1, dtheta2, ddtheta1, ddtheta2, 0.))


cdef float wrap(float x, float m,  float M):
    """
    :param x: a scalar
    :param m: minimum possible value in range
    :param M: maximum possible value in range
    Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
    """
    cdef float diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x

cdef float bound(float x, float m, float M):
    """
    :param x: scalar
    Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].
    """
    return min(max(x, m), M)


cdef np_float_t[:,:] rk4(object derivs, np_float_t[:] y0, tuple t):
    
    cdef:
        int i
        np.ndarray[dtype=np_float_t, ndim=2] yout = np.empty((len(t), y0.shape[0]), dtype=np.float64)
        float thist
        float dt
        float dt2
        np_float_t[:] k1
        np_float_t[:] k2
        np_float_t[:] k3
        np_float_t[:] k4

    yout[0] = y0

    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = derivs(y0)
        k2 = derivs(y0 + np.multiply(dt2, k1))
        k3 = derivs(y0 + np.multiply(dt2, k2))
        k4 = derivs(y0 + np.multiply(dt, k3))
        yout[i + 1] = np.add(y0, np.array(dt / 6.0 * (k1 + np.multiply(2.0, k2) + np.multiply(2.0, k3) + k4)))
    return yout