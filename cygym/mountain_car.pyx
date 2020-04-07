#!python

# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: nonecheck=False

"""
Credit:
    
This code has been adapted from the original OpenAI release:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py

"""

## Loading packages
from gym.spaces import Box, Discrete
from gym.utils import seeding
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport cos


# Defining our numpy types
ctypedef np.float_t np_float_t


cdef class CyMountainCarEnv():
    
    cdef:
        float min_position
        float max_position
        float max_speed
        float goal_position
        float goal_velocity
        float force
        float gravity
        readonly object action_space
        readonly object observation_space
        tuple state
        
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
        

    def __cinit__(self, goal_velocity = 0):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = goal_velocity
        
        self.force=0.001
        self.gravity=0.0025

        self.action_space = Discrete(3)
        self.observation_space = Box(np.array((self.min_position, -self.max_speed)),
                                     np.array((self.max_position, self.max_speed)))

        self.seed()

    cpdef tuple step(self, int action):
        cdef:
            float position = self.state[0]
            float velocity = self.state[1]
            float force = self.force
            float gravity = self.gravity
            float max_speed = self.max_speed
            float min_position = self.min_position
            float max_position = self.max_position
            bint done
            float reward = -1.0
            tuple state

        velocity += (action-1)*force + cos(3*position)*(-gravity)
        
        if velocity < -max_speed:
            velocity = -max_speed
        elif velocity > max_speed:
            velocity = max_speed

        position += velocity
        
        if position < min_position:
            position = min_position
        elif position > max_position:
            position = max_position
            
        if (position == min_position and velocity<0):
            velocity = 0

        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)

        state = (position, velocity)
        self.state = state
        return (np.array(state), reward, done, {})

    cpdef np.ndarray[dtype=np_float_t, ndim=1] reset(self):
        cdef tuple state = (self.np_random.uniform(low=-0.6, high=-0.4), 0)
        self.state = state
        return np.array(state)
