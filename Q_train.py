import numpy as np
import random
import env_loader
import os

env_settings = env_loader.load_env('env.csv')
n_states = env_settings['n_states']
goal_state = env_settings['goal_state']
start_state = env_settings['start_state']
obstacles = env_settings['obstacles']
water = env_settings['water']
actions = env_settings['actions']

alpha = 0.5
epsilon = 0.3
episodes = 30
max_steps = 1000

Q_table = np.zeros((n_states, len(actions)))

def get_reward(state):
    if state == goal_state:
        return 100
    elif state in obstacles:
        return -100
    elif state in water:
        return -10
    else:
        return -1

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(len(actions)))
    else:
        return np.argmax(Q_table[state])

if not os.path.exists('policy'):
        os.makedirs('policy')

for episode in range(episodes):
    state = start_state
    steps = 0
    while state != goal_state and steps < max_steps:
        action_index = choose_action(state)
        action = actions[action_index]
        next_state = max(0, min(n_states - 1, state + action))
        reward = get_reward(next_state)
        Q_table[state, action_index] += alpha * (reward + np.max(Q_table[next_state]) - Q_table[state, action_index])
        state = next_state
        steps += 1

    if episode % 1 == 0 or episode == episodes - 1:
        policy = [actions[np.argmax(Q_table[state])] for state in range(n_states)]
        np.save(f'policy/episode_{episode}.npy', policy)

print("学習終了")
