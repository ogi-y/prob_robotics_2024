import numpy as np
import matplotlib.pyplot as plt
import env_loader
import os

env_settings = env_loader.load_env('env.csv')
n_states = env_settings['n_states']
goal_state = env_settings['goal_state']
start_state = env_settings['start_state']
obstacles = env_settings['obstacles']
water = env_settings['water']
actions = env_settings['actions']

eps = 0.3
max_steps = 1000
episodes = 30
num_trials = 50

def choose_action(state, policy):
    if np.random.uniform(0, 1) < eps:
        return np.random.choice([-1, 1])
    else:
        return policy[state]

def get_goal_time(policy):
    state = start_state
    time_steps = 0
    while state != goal_state and time_steps < max_steps:
        action = choose_action(state, policy)
        state = max(0, min(n_states - 1, state + action))
        time_steps += 1
    if state == goal_state:
        return time_steps
    else:
        return max_steps

def calculate_average_goal_time():
    goal_times = []
    
    for episode_num in range(episodes):
        episode_goal_times = []
        policy_path = f'policy/episode_{episode_num}.npy'
        if not os.path.exists(policy_path):
            print(f"Policy file {policy_path} not found. Skipping.")
            goal_times.append(np.nan)
            continue

        policy = np.load(policy_path)
        for _ in range(num_trials):
            goal_time = get_goal_time(policy)
            episode_goal_times.append(goal_time)
        
        goal_times.append(np.mean(episode_goal_times))
    
    return goal_times

goal_times = calculate_average_goal_time()

plt.plot(range(1,episodes+1), goal_times, marker='o', linestyle='-', color='b')
plt.title('Average Goal Reaching Step per Episode')
plt.xlabel('Episode')
plt.ylabel('Average Steps to Goal')
plt.xticks(range(1,episodes+1))
plt.grid(True)

for i, goal_time in enumerate(goal_times):
    plt.text(i, goal_time+5, f'{goal_time:.1f}', ha='center', fontsize=10)

#plt.show()
plt.savefig('images/average_goal_time.png')