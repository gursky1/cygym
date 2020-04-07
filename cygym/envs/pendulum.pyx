#!python

# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: nonecheck=False

## Loading packages
from gym.spaces import Box
from gym.utils import seeding
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sin, cos, M_PI

# Defining our numpy types
ctypedef np.float_t np_float_t
ctypedef np.int_t np_int_t


cdef class FastPendulumEnv():
    
    cdef:
        int max_speed
        float max_torque
        float dt
        float g
        float m
        float l
        np_float_t[:] state
        readonly object action_space
        readonly object observation_space
        
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


    def __cinit__(self, float g=10.0):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.g = g
        self.m = 1.
        self.l = 1.

        self.action_space = Box(low=-self.max_torque, high=self.max_torque, shape=(1,))
        self.observation_space = Box(low=np.array((-1., -1., -self.max_speed)),
                                     high=np.array((1., 1., self.max_speed)))

        self.seed()

    cpdef tuple step(self, float u):
        cdef:
            float th = self.state[0]
            float thdot = self.state[1]
            float mt = self.max_torque
            float ms = self.max_speed
            float g = self.g
            float m = self.m
            float l = self.l
            float dt = self.dt
            float costs
            float newthdot
            float newth

        if u < -mt:
            u = -mt
        elif u > mt:
            u = mt
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * sin(th + M_PI) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        if newthdot < -ms:
            newthdot = -ms
        elif newthdot > ms:
            newthdot = ms

        self.state = np.array((newth, newthdot))
        return (np.array(self._get_obs()), -costs, False, {})

    
    cpdef np.ndarray[dtype=np_float_t, ndim=1] reset(self):
        cdef np.ndarray[dtype=np_float_t, ndim=1] high = np.array((M_PI, 1.))
        self.state = self.np_random.uniform(low=-1*high, high=high)
        return np.array(self._get_obs())

    cpdef tuple _get_obs(self):
        cdef float theta = self.state[0]
        cdef float thetadot = self.state[1]
        return (cos(theta), sin(theta), thetadot)


cdef float angle_normalize(float x):
    return (((x+M_PI) % (2*M_PI)) - M_PI)