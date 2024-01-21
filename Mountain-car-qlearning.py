#%%
from tqdm import tqdm
import gymnasium as gym, seaborn as sns, matplotlib.pyplot as plt
import random; random.seed(0)
import numpy as np; np.random.seed(0)

sns.set_style('whitegrid')
env = gym.make('MountainCar-v0')

def normalize_state(state):
    
    n_state = state.copy()
    step_c1 = (0.6 + 1.2 ) / 12
    step_c2 = (0.07 + 0.07) / 12

    n_state[0] = (n_state[0] + 1.2) // step_c1
    n_state[1] = (n_state[1] + 0.07) // step_c2

    return n_state.astype(int) #- 6


def normalize_state_tile_coding(state):

    tiles = 12
    n = 6
    
    range_c1 = (0.6 + 1.2 ) / tiles
    range_c2 = (0.07 + 0.07) / tiles
    

    increment_c1 = range_c1 / n
    increment_c2 = range_c2 / n

    n_state = []
    for i in range(n):
        n_state += [((state[0] + 1.2  + i*increment_c1) // range_c1)]
        n_state += [((state[1] + 0.07 + i*increment_c2) // range_c2) ]

    return np.array(n_state).astype(int)

def q_function(state):

    return q[state[0],state[1]]
    # return np.matmul(w.T, state)

alpha = 0.1
epsilon = 0.99
gamma = 1.0
w = np.random.randn(12, 3)
# w = np.random.randn(2, 3)
q = np.zeros((30, 30, 3))

iters = int(1e3)

episodes = int(1e4)
exp_decay = epsilon/200

losses = []
rec = []

z = tqdm(range(episodes))
for episode in z:

    recc = 0

    prev_state, info = env.reset()
    # prev_state = normalize_state_tile_coding(prev_state)
    prev_state = normalize_state(prev_state)

    for _ in range(iters):

        q_values = q_function(prev_state)
        action = np.random.choice(np.arange(3)) if np.random.uniform() < epsilon else np.argmax(q_values)

        observation, reward, terminated, _, _ = env.step(action)
        # state = normalize_state_tile_coding(observation)
        state = normalize_state(observation)
        # print(state, normalize_state_tile_coding(observation))

        if terminated:
            reward = 0

        q[prev_state[0], prev_state[1], action] = (1. - alpha)*q[prev_state[0], prev_state[1], action] + alpha*(reward  + gamma * q_function(state).max()*(1-terminated))
        # delta_w = (reward  + gamma * q_function(state).max()*(1-terminated) - q_values[action]) * prev_state
        recc += reward
        # w[:, action] += alpha*delta_w
        # print(w)

        prev_state = state

        if terminated:
            break
        
    rec += [recc]
    epsilon = max( epsilon - exp_decay, 1e-2)
    z.set_postfix_str(f'epsilon: {epsilon:.3f} reward: {np.mean(rec):.3f}')
env.close()

plt.plot(rec)
plt.show()
#%%

env = gym.make('MountainCar-v0', render_mode='human')
prev_state, info = env.reset()
prev_state = normalize_state(prev_state)

for _ in range(int(1e4)):

    q_values = q_function(prev_state)

    action = np.argmax(q_values)
    observation, reward, terminated, _, _ = env.step(action)
    print(action)

    prev_state = normalize_state(observation)
    prev_action = action

    if terminated:
        break

env.close()
# %%
