import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

import pandas as pd
import random
import numpy as np
from kaggle_environments.envs.rps.utils import get_score
import matplotlib.pyplot as plt
import seaborn as sns
from kaggle_environments import make, evaluate
import keras
import collections
import sys
import os

#Util method for getting the state in the q table
def get_state(action, op_action):
    return action * 3 + op_action
# Current action
cur_action = 0
# Epsilon: Exploration rate
eps = 0.1
# History
history = []
# Q_table (Policies) Shape: (9, 3)
policies = [[0] *3] * 9
#Learning rate
lr = 0.7
# Discount rate for q_table
discount_rate = 0.3
# Epsilon decay rate
decay_rate = 0.9

def update_q_table(op_action):
    global policies
    global discount_rate
    global lr
    global history
    reward = get_score(cur_action, op_action)
    if len(history) > 1:
        previous_state_id = get_state(history[len(history) - 2][0], history[len(history) - 2][1])
        state_id = get_state(cur_action, op_action)
        policies[previous_state_id][cur_action] = policies[previous_state_id][cur_action] * (1 - lr) \
        + lr * (reward + discount_rate * np.max(policies[state_id][:]))

def mdp(observation, configuration):
    global cur_action
    global history
    global policies
    if observation.step > 0:
        history.append([cur_action, observation.lastOpponentAction])
        update_q_table(observation.lastOpponentAction)
    
    explore_rate = np.random.random()
    if explore_rate < eps:
        cur_action = random.randint(0, 2)
        explore_rate *= decay_rate
    else:
        if observation.step > 0:
            state_id = get_state(cur_action, observation.lastOpponentAction)
            cur_action = int(np.argmax(policies[state_id][:]))
        else:
            cur_action = random.randint(0, 2)
    return cur_action        

env = make("rps", configuration={"episodeSteps": 1000}, debug="True")
env.reset()
env.run(["mdp.py", "statistical"])
env.render(mode="ipython", width=400, height=400)

