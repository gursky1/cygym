#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 20:09:33 2020

@author: jacob
"""


from .envs.cartpole import FastCartPoleEnv
from .envs.pendulum import FastPendulumEnv
from .envs.acrobot import FastAcrobotEnv
from .envs.mountain_car import FastMountainCarEnv
from .envs.continuous_mountain_car import FastContinuous_MountainCarEnv
"""
from gym.envs.registration import registry, register, make, spec

register(
    id='FastCartPole-v0',
    entry_point='fast_gym.cartpole:FastCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='FastMountainCar-v0',
    entry_point='fast_gym.mountain_car:FastMountainCarEnv',
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id='FastMountainCarContinuous-v0',
    entry_point='fast_gym.continuous_mountain_car:FastContinuous_MountainCarEnv',
    max_episode_steps=999,
    reward_threshold=90.0,
)

register(
    id='FastPendulum-v0',
    entry_point='fast_gym.pendulum:FastPendulumEnv',
    max_episode_steps=200,
)

register(
    id='FastAcrobot-v0',
    entry_point='fast_gym.acrobot:AcrobotEnv',
    reward_threshold=-100.0,
    max_episode_steps=500,
)
"""