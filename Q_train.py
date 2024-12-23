import numpy as np
import random

# 環境の設定
n_states = 20  # 数直線の長さ
goal_state = n_states - 1  # ゴール地点
start_state = 5  # 開始地点
obstacles = [2]  # 障害物の位置
actions = [-1, 1]  # 行動: 左(-1)か右(+1)

# Q学習のパラメータ
alpha = 0.5  # 学習率
gamma = 0.9  # 割引率
epsilon = 0.3  # ε-greedy法の探索率
episodes = 20  # エピソード数

# Qテーブルの初期化
Q_table = np.zeros((n_states, len(actions)))

# 報酬関数
def get_reward(state):
    if state == goal_state:
        return 100  # ゴールに到達した場合の報酬
    elif state in obstacles:
        return -100  # 障害物にぶつかった場合のペナルティ
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
            reward + gamma * np.max(Q_table[next_state]) - Q_table[state, action_index]
        )
        
        # 状態を更新
        state = next_state

    # 100回ごとに方策を保存
    if (episode + 1) % 1 == 0:
        policy = [actions[np.argmax(Q_table[state])] for state in range(n_states)]
        np.save(f'policy/episode_{episode + 1}.npy', policy)

print("学習終了")
