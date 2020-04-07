#!python

# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: nonecheck=False

"""
Credit:
    
This code has been adapted from the original OpenAI release:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py

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


cdef class CyContinuous_MountainCarEnv():
    
    cdef:
        float min_action
        float max_action
        float min_position
        float max_position
        float max_speed
        float goal_position
        float goal_velocity
        float power
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
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45 # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.goal_velocity = goal_velocity
        self.power = 0.0015
        
        self.action_space = Box(low=self.min_action, high=self.max_action, shape=(1,))
        self.observation_space = Box(np.array((self.min_position, -self.max_speed)),
                                     np.array((self.max_position, self.max_speed)))

        self.seed()


    cpdef tuple step(self, float action):
        cdef:
            float position = self.state[0]
            float velocity = self.state[1]
            float force = min(max(action, -1.0), 1.0)
            float power = self.power
            float max_speed = self.max_speed
            float max_position = self.max_position
            float min_position = self.min_position
            float goal_position = self.goal_position
            float goal_velocity = self.goal_velocity
            bint done
            float reward
            tuple state

        velocity += force*power -0.0025 * cos(3*position)
        if (velocity > max_speed):
            velocity = max_speed
        elif (velocity < max_speed):
            velocity = -max_speed
        position += velocity
        
        if (position > max_position):
            position = max_position
        elif (position < min_position):
            position = min_position
        
        if (position == min_position and velocity<0):
            velocity = 0

        done = bool(position >= goal_position and velocity >= goal_velocity)

        reward = 0
        if done:
            reward = 100.0
        reward-= (action**2)*0.1

        #state = np.array((position, velocity))
        state = (position, velocity)
        self.state = state
        return (np.array(state), reward, done, {})

    cpdef np.ndarray[dtype=np_float_t, ndim=1] reset(self):
        cdef tuple state = (self.np_random.uniform(low=-0.6, high=-0.4), 0)
        self.state = state
        return np.array(state)