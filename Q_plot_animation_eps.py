import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import env_loader
import os

env_settings = env_loader.load_env('env.csv')
n_states = env_settings['n_states']
goal_state = env_settings['goal_state']
start_state = env_settings['start_state']
obstacles = env_settings['obstacles']
water = env_settings['water']
actions = env_settings['actions']

episode = 14
policy = np.load(f'policy/episode_{episode}.npy')

eps = 0.3

state = start_state
cumulative_reward = 0
goal_reached = False

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
    if random.uniform(0, 1) < eps:
        return random.choice([-1, 1])
    else:
        return policy[state]

def animate_agent(i, line, ax, reward_text, cumulative_reward_text, frame_text):
    global state, cumulative_reward, goal_reached

    if state != goal_state:
        action = choose_action(state)
        state = max(0, min(n_states - 1, state + action))

    reward = get_reward(state)
    if goal_reached == False:
        cumulative_reward += reward

    line.set_data([state], [0.2])

    reward_text.set_text(f"Reward: {reward}")
    cumulative_reward_text.set_text(f"Total Reward: {cumulative_reward}")
    frame_text.set_text(f"Frame: {i}")

    if state == goal_state :    
        goal_reached = True
        ax.set_title("Goal Reached!")

    return line, reward_text, cumulative_reward_text, frame_text

def plot_animation():
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.set_xlim(-0.5, n_states - 0.5)
    ax.set_ylim(-1, 1)

    ax.plot(np.arange(n_states), np.zeros(n_states), color='black', lw=2)
    for obs in obstacles:
        ax.add_patch(patches.Rectangle((obs - 0.5, 0), 1, 0.5, facecolor='red'))
    for wat in water:
        ax.add_patch(patches.Rectangle((wat - 0.5, -0.5), 1, 0.5, facecolor='blue'))
    flag_pole_g = patches.Rectangle((goal_state, 0), 0.1, 0.6, facecolor='black')
    ax.add_patch(flag_pole_g)
    flag_g = patches.Polygon([(goal_state, 0.3), (goal_state+0.4, 0.5), (goal_state, 0.7)], 
                            closed=True, facecolor='red')
    ax.add_patch(flag_g)
    flag_pole_s = patches.Rectangle((start_state, 0), 0.1, 0.6, facecolor='black')
    ax.add_patch(flag_pole_s)
    flag_s = patches.Polygon([(start_state, 0.3), (start_state + 0.4, 0.5), (start_state, 0.7)], 
                            closed=True, facecolor='green')
    ax.add_patch(flag_s)
    agent, = ax.plot([start_state], [0.2], 'go', markersize=10)
    reward_text = ax.text(0, 1.5, "Reward: 0", fontsize=12, color='black')
    cumulative_reward_text = ax.text(0, 1, "Cumulative Reward: 0", fontsize=12, color='black')
    frame_text = ax.text(0, -1, "Frame: 0", fontsize=12, color='black')
    for i in range(n_states):
        if policy[i] == 1:
            ax.text(i, 0.5, "→", ha='center', fontsize=12, color='black')
        elif policy[i] == -1:
            ax.text(i, 0.5, "←", ha='center', fontsize=12, color='black')

    ax.axis('off')
    ax.axis('equal')

    ani = animation.FuncAnimation(fig, animate_agent, frames=100, fargs=(agent, ax, reward_text, cumulative_reward_text, frame_text),
                                  interval=500, repeat=False)
    #plt.show()
    if not os.path.exists('images'):
        os.makedirs('images')
    ani.save(f'images/episode_{episode}.gif', fps=2)

plot_animation()
