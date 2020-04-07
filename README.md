# cygym
This repository contains cythonized versions on the OpenAI Gym classic control environments.  Note that is this package is actively under development.

## Installation

Cygym can be installed using the following git+pip code:

```
git clone https://github.com/gursky1/cygym
cd cygym
pip install -e .
```

## Quick Start

Cygym environments are not yet compatible with `gym.make()`, so they need to be imported as classes:

```
# Importing packages
from gym.wrappers import TimeLimit
from cygym import CyCartPoleEnv

# Adding a timelimit to our environment
env = TimeLimit(CyCartPoleEnv(), 500)

# Running through some dummy actions
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
```

## Planned Features

+ Compatibility with OpenAI Gym registery (`gym.make()`)
+ Optional fast rendering
+ Cythonizing Box2d enviroments

## Contributing

Cygym is looking for contributors, so if are interested please reach out to the primary maintainer gurskyjacob@gmail.com!
