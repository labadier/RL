#%%
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

v_s = np.zeros((5, 5))
special_states = [(0,1), (0,3)]
special_transitions = [(4, 1), (2, 3)]
special_states_rewards = [10, 5]
epsilon = 1e-2
gamma = 0.9

convergence = False

mf = [1, -1, 0, 0]
mc = [0, 0, -1, 1]

global_expected_reward = []

while not convergence:

    v_new = np.zeros_like(v_s)

    for f in range(5):
        for c in range(5):

            if (f, c) in special_states:
                v_new[f, c] = special_states_rewards[special_states.index((f, c))] + gamma*v_s[special_transitions[special_states.index((f, c))]]
                continue

            max_value = -1 << 30
            for i in range(len(mf)):
                nf = f + mf[i]
                nc = c + mc[i]

                rw = 0
                if nf >= 5 or nf < 0 or nc >= 5 or nc < 0:
                    nf = f; nc = c
                    rw = -1

                max_value = max(max_value,
                                rw + gamma*v_s[nf, nc])
            v_new[f, c] = max_value

    global_expected_reward += [np.mean(v_new)]
    convergence |= np.max(np.abs(v_s - v_new)) < epsilon
    v_s = v_new

sns.set()
plt.ylabel('Average Reward')
plt.xlabel('Steps')
plt.plot(global_expected_reward)
plt.title('Value Iteration $gamma$ = 0.9')
print(v_s)
  
# %%
