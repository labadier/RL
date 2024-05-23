#%%
import torch
from tqdm import tqdm
import numpy as np
import gymnasium as gym

from matplotlib import pyplot as plt

from collections import deque

env = gym.make('MountainCar-v0')

class ReplayBuffer:

    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = torch.zeros((self.mem_size, *input_shape), dtype=torch.float32)
        self.new_state_memory = torch.zeros((self.mem_size, *input_shape), dtype=torch.float32)
        self.action_memory = torch.zeros((self.mem_size), dtype=torch.long)
        self.reward_memory = torch.zeros((self.mem_size, 1), dtype=torch.float32)
        self.terminal_memory = torch.zeros((self.mem_size, 1), dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

class PModel(torch.nn.Module):
    def __init__(self):
        super(PModel, self).__init__()

        self.bootstrap = 0
        self.fc1 = torch.nn.Linear(2, 32)
        self.fc2 = torch.nn.Linear(32, env.action_space.n)

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.criterion = torch.nn.MSELoss()
        self.to(device)
        self.device = device
        

    def forward(self, x):
        
        x = torch.nn.functional.relu(self.fc1(x.to(self.device)))
        x = self.fc2(x)
        return x
    
    def make_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
    
class DVN(PModel):

    def __init__(self):
        super(DVN, self).__init__()

        self.bootstrap = 0
        self.fc1 = torch.nn.Linear(2, 32)
        self.fc2 = torch.nn.Linear(32, 1)

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.criterion = torch.nn.MSELoss()
        self.to(device)
        self.device = device
    
    def forward(self, x):
        
        x = torch.nn.functional.relu(self.fc1(x.to(self.device)))
        x = self.fc2(x)
        return x
    
    def make_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)

def pick_action(preferences):

    preferences = torch.nn.functional.softmax(preferences, dim=-1)
    action = torch.multinomial(preferences, 1)
  
    return action.item()

done = True
V = DVN()
P = PModel()

episodes = 100
iterations = 1000
bootstrap = 50
lr = 0.001
buffer_size = 100
batch_size = 32

gamma = 0.99

rewards_history = []
losses_history_q = []
losses_history_p = []

optimizer_vfunction = V.make_optimizer(lr)
optimizer_pfunction = P.make_optimizer(lr)

buffer = None
episodes = tqdm(range(episodes))

def preprocess_state(state):
    state = torch.tensor(state).float()

    return state

for episode in episodes:

    prev_state, _= env.reset()
    prev_state = preprocess_state(prev_state)

    for iteration in tqdm(range(iterations)):

        with torch.no_grad():
            v_values = V(prev_state.unsqueeze(0))
            preferences = P(prev_state.unsqueeze(0))
            act = pick_action(preferences.detach())

        state, reward, done,_, info = env.step(act)
        rewards_history.append(reward)
        state = preprocess_state(state)
      
        if buffer is None:
            buffer = ReplayBuffer(buffer_size, prev_state.shape)
        buffer.store_transition(prev_state, act, reward, state, not done)

        if buffer is not None and buffer.mem_cntr >= batch_size :
            
            prev_states, actions, rewards, states, dones = buffer.sample_buffer(batch_size)
            v_values = V(prev_states)
            v_values_1 = V(states).detach()
            
            preferences = P(prev_states)
            mask = torch.functional.F.one_hot(actions, env.action_space.n).to(V.device)

            log_p_values = torch.log(torch.functional.F.softmax(preferences, dim=-1))
            log_p_values = (mask*log_p_values).sum(axis=-1, keepdim=True)

            loss_q = V.criterion(rewards.to(V.device) + gamma*dones.to(V.device)*v_values_1, v_values)
            
            td0 = rewards.to(V.device) + gamma*dones.to(V.device)*v_values_1 - v_values
            loss_p = (td0.detach()*log_p_values).mean()
            V.bootstrap += 1

            losses_history_q.append(loss_q.item())
            losses_history_p.append(loss_p.item())
            
            loss_q.backward()
            loss_p.backward()

            optimizer_pfunction.step()
            optimizer_pfunction.zero_grad()

            optimizer_vfunction.step()
            optimizer_vfunction.zero_grad()

            if V.bootstrap >= bootstrap:
                torch.save(V.state_dict(), 'V_target.pt')
                torch.save(P.state_dict(), 'P.pt')
            
        prev_state = state

        if done:
            break
    episodes.set_postfix_str(f'reward: {np.mean(rewards_history[-100:]):.3f}')
env.close()

plt.plot(losses_history_q, label='V function')
plt.plot(losses_history_p, label='P function')

plt.show()
 # %%

# Q.load('Q_target.pt')
# P.load('P.pt')

env = gym.make('MountainCar-v0', render_mode='human')
prev_state, info = env.reset()
prev_state = preprocess_state(prev_state)

for _ in range(int(1e4)):

    action = pick_action(P(prev_state.unsqueeze(0)))

    observation, reward, terminated, _, _ = env.step(action)
    print(action)

    prev_state = preprocess_state(observation)
    prev_action = action

    if terminated:
        break

env.close()


# %%

# %%
