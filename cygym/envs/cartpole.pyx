#!python

# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: nonecheck=False

## Importing packages
from gym.spaces import Box, Discrete
from gym.utils import seeding
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sin, cos, M_PI

# Defining our numpy types
ctypedef np.float_t np_float_t
ctypedef np.int_t np_int_t


cdef class FastCartPoleEnv():
    
    cdef:
        float gravity
        float masscart
        float masspole
        float total_mass
        float length
        float polemass_length
        float force_mag
        float tau
        int kinematics_integrator
        float theta_threshold_radians
        float x_threshold
        np_float_t[:] state
        int steps_beyond_done
        readonly object action_space
        readonly object observation_space
        
    cdef object np_random
    metadata = {}
    reward_range = (-float('inf'), float('inf'))
    spec = None
    
    @property
    def unwrapped(self): return self
    
    cdef inline void seed(self, seed=1):
        if seed == 0:
            self.np_random, _ = seeding.np_random()
        else:
            self.np_random, _ = seeding.np_random(seed)
        
        
    def __cinit__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = 1.1
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = 0.05
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * M_PI / 360
        self.x_threshold = 2.4
        
        cdef float m_max = np.finfo(np.float64).max
        cdef np.ndarray[dtype=np_float_t, ndim=1] high = np.array((self.x_threshold * 2,
                                                                   m_max,
                                                                   self.theta_threshold_radians * 2,
                                                                   m_max))
        
        self.action_space = Discrete(2)
        self.observation_space = Box(-1*high, high)
        self.seed()

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds

    cpdef tuple step(self, int action):
        
        cdef:
            np_float_t[:] state = self.state
            float x = state[0]
            float x_dot = state[1]
            float theta = state[2]
            float theta_dot = state[3]
            float force = self.force_mag if action==1 else -self.force_mag
            float costheta = cos(theta)
            float sintheta = sin(theta)
            float temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
            float thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
            float xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
            bint done
            float reward

        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        state = np.array((x,x_dot,theta,theta_dot))
        self.state = state
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done == -1:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            self.steps_beyond_done += 1
            reward = 0.0

        return (np.asarray(state), reward, done, {})
    

    cpdef np.ndarray[dtype=np_float_t, ndim=1] reset(self):
        cdef np_float_t[:] state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state = state
        self.steps_beyond_done = -1
        return np.asarray(state)