import numpy as np
import random
import env_loader

# 環境の設定
env_settings = env_loader.load_env('env.csv')
n_states = env_settings['n_states']
goal_state = env_settings['goal_state']
start_state = env_settings['start_state']
obstacles = env_settings['obstacles']
water = env_settings['water']
actions = env_settings['actions']

alpha = 0.5
epsilon = 0.3
episodes = 20

# Qテーブルの初期化
Q_table = np.zeros((n_states, len(actions)))

# 報酬関数
def get_reward(state):
    if state == goal_state:
        return 100  # ゴールに到達した場合の報酬
    elif state in obstacles:
        return -100  # 障害物にぶつかった場合のペナルティ
    elif state in water:
        return -10  # パンダにぶつかった場合のペナルティ
    else:
        return -1  # 移動に対するペナルティ

# 行動選択（ε-greedy法）
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(len(actions)))  # 探索
    else:
        return np.argmax(Q_table[state])  # 活用

# 学習ループ
for episode in range(episodes):
    state = start_state  # エピソードの開始地点
    while state != goal_state:
        # 行動選択
        action_index = choose_action(state)
        action = actions[action_index]
        
        # 状態遷移
        next_state = max(0, min(n_states - 1, state + action))
        
        # 報酬の取得
        reward = get_reward(next_state)
        
        # Q値の更新
        Q_table[state, action_index] += alpha * (
            reward + np.max(Q_table[next_state]) - Q_table[state, action_index]
        )
        
        # 状態を更新
        state = next_state

    # 100回ごとに方策を保存
    if (episode + 1) % 1 == 0:
        policy = [actions[np.argmax(Q_table[state])] for state in range(n_states)]
        np.save(f'policy/episode_{episode + 1}.npy', policy)

print("学習終了")
