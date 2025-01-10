import csv

def load_env(file_path):
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        env = {}
        for row in reader:
            key = row[0]
            values = row[1:]
            if key in ["n_states", "goal_state", "start_state"]:  
                env[key] = int(values[0])
            else:
                env[key] = [int(v) for v in values]
        return env

"""# 環境設定の読み込み
env_settings = load_env('env.csv')

# 環境変数の設定
n_states = env_settings['n_states']
goal_state = env_settings['goal_state']
start_state = env_settings['start_state']
obstacles = env_settings['obstacles']
puddle = env_settings['puddle']
actions = env_settings['actions']

print("環境設定を読み込みました:")
print(f"n_states: {n_states}, goal_state: {goal_state}, start_state: {start_state}")
print(f"obstacles: {obstacles}, puddle: {puddle}, actions: {actions}")"""
