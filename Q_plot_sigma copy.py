import numpy as np
import matplotlib.pyplot as plt

# 環境の設定
n_states = 20  # 数直線の長さ
goal_state = n_states - 1  # ゴール地点
start_state = 5  # 開始地点
obstacles = [2]  # 障害物の位置

# σ（探索の確率）
sigma = 0.3  # 0.1 の確率でランダム行動を選択

# 行動選択（ランダムまたは方策に従う）
def choose_action(state, policy):
    if np.random.uniform(0, 1) < sigma:
        return np.random.choice([-1, 1])  # ランダムに行動
    else:
        return policy[state]  # 方策に従う

# 1回のエピソードにおけるゴール到達時間を計算
def get_goal_time(policy):
    state = start_state
    time_steps = 0
    while state != goal_state and time_steps < 500:  # 100ステップを上限
        action = choose_action(state, policy)  # σ に従って行動を選択
        state = max(0, min(n_states - 1, state + action))  # 状態遷移
        time_steps += 1
    if state == goal_state:  # ゴールに到達した場合
        return time_steps
    else:
        return 100  # ゴールに到達しなかった場合、100を返す（打ち切り）

# すべてのエピソードについてゴール到達時間を計算
def calculate_average_goal_time():
    goal_times = []
    
    for episode_num in range(1, 21):  # episode1.npy から episode20.npyまで
        episode_goal_times = []
        
        for _ in range(10):  # 各エピソードを10回試行
            policy = np.load(f'policy/episode_{episode_num}.npy')  # 方策の読み込み
            goal_time = get_goal_time(policy)  # ゴール到達時間の計算
            episode_goal_times.append(goal_time)
        
        # 各エピソードの平均ゴール到達時間を記録
        goal_times.append(np.mean(episode_goal_times))
    
    return goal_times

# ゴール到達時間の平均を計算
goal_times = calculate_average_goal_time()

# 結果をグラフにプロット
plt.plot(range(1, 21), goal_times, marker='o', linestyle='-', color='b')
plt.title('Average Goal Reaching Time per Episode')
plt.xlabel('Episode')
plt.ylabel('Average Time Steps to Goal')
plt.xticks(range(1, 21))
plt.grid(True)

# 各点の上に座標を表示
for i, goal_time in enumerate(goal_times):
    plt.text(i+1, goal_time+5, str(goal_time), ha='center', fontsize=10)

plt.show()
