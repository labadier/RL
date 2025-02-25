#%%
import torch
from tqdm import tqdm
import numpy as np
import gymnasium as gym

from matplotlib import pyplot as plt

from collections import deque

env = gym.make('MountainCar-v0', render_mode='human')

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

class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.bootstrap = 0
        self.fc1 = torch.nn.Linear(2, 32)
        self.fc2 = torch.nn.Linear(32, env.action_space.n)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.MSELoss()
        self.to(device)
        self.device = device
        

    def forward(self, x):
        
        x = torch.nn.functional.relu(self.fc1(x.to(self.device)))
        x = self.fc2(x)
        return x
    
    def make_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)
    
def pick_action(q_values, epsilon):

    if torch.rand(1) < epsilon:
        return env.action_space.sample()
    else:
        return torch.argmax(q_values).item()

done = True
Q = DQN()
Q_prime = DQN()
episodes = 1000
iterations = 1000
bootstrap = 50
epsilon = 0.05
lr = 0.001
buffer_size = 100
batch_size = 32

gamma = 0.99

min_dist_req = 40
max_time_for_dist = 100

rewards_history = []
losses_history = []

optimizer = Q.make_optimizer(lr)

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
            q_values = Q(prev_state.unsqueeze(0))
            act = pick_action(q_values.detach(), epsilon)

        state, reward, done,_, info = env.step(act)
        rewards_history.append(reward)

        state = preprocess_state(state)
      
        if buffer is None:
            buffer = ReplayBuffer(buffer_size, prev_state.shape)
        buffer.store_transition(prev_state, act, reward, state, not done)

        if buffer is not None and buffer.mem_cntr >= batch_size :
            
            prev_states, actions, rewards, states, dones = buffer.sample_buffer(batch_size)
            q_values = Q(prev_states)
            q_start_values = Q_prime(states).detach().max(dim=-1, keepdim=True).values

            mask = torch.functional.F.one_hot(actions, env.action_space.n).to(Q.device)

            loss = Q.criterion(rewards.to(Q.device) + gamma*dones.to(Q_prime.device)*q_start_values, torch.sum(q_values*mask, axis=-1, keepdim=True))
            Q.bootstrap += 1

            losses_history.append(loss.item())
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if Q.bootstrap >= bootstrap:
                Q_prime.load_state_dict(Q.state_dict())
                torch.save(Q_prime.state_dict(), 'Q_target.pt')
            
        prev_state = state

        if done:
            break
    episodes.set_postfix_str(f'epsilon: {epsilon:.3f} reward: {np.mean(rewards_history[-100:]):.3f}')
env.close()

plt.plot(losses_history)
plt.show()
     # %%

