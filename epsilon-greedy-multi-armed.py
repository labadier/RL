#%%
import numpy as np
from matplotlib import pyplot as plt

def compute_arms_increment():
  return np.random.normal(size=(10,), loc=0, scale=1.0)

def interact(action):
  return np.random.normal(loc=q[action], scale=1.0)

def update_expected_reward( reward, action_taken, q_hat, n_a=None, alpha = None):
  

  if alpha is not None:
    q_hat[action_taken] = q_hat[action_taken] + alpha*(reward - q_hat[action_taken])
  else:
    n_a[action_taken] += 1
    q_hat[action_taken] = q_hat[action_taken] + (1./n_a[action_taken])*(reward - q_hat[action_taken])
  
  return q_hat

def take_action(epsilon, actions, q_hat):

  if np.random.uniform() > 1 - epsilon:
    return np.random.randint(0, actions)
  else: return np.argmax(q_hat)

setup = int(2e3)
epsilons = [0.1, 0.1]
steps = int(1e4)
history = np.zeros((len(epsilons), steps))

for ep, epsilon in enumerate(epsilons):
  
  for s in range(setup):

    q = np.zeros(10,) + np.random.normal(loc=0, scale=1.0)
    q_hat = np.zeros(10,)
    n_a = np.zeros(10,)

    for i in range(steps):
      action = take_action(epsilon=epsilon, actions=10, q_hat=q_hat)
      reward = interact(action)
      q_hat = update_expected_reward(reward=reward, action_taken=action, q_hat=q_hat, n_a=n_a, alpha = None if not ep else 0.1 )
      history[ep][i] += reward
      q += compute_arms_increment()
      # q /= np.sum(q)
  history[ep] /= (1.*setup)


colors = ['r', 'b', 'g']
plt.ylabel('Average Reward')
plt.xlabel('Steps')
for i, epsilon in enumerate(epsilons):
  plt.plot(list(history[i]), color=colors[i])
plt.legend(['sample averages', 'alpha = 0.1'], loc='upper right')
plt.title('Epsilon gredy. Action-value methods - avg sample and weighted avg')
  
# %%
