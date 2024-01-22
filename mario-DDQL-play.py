#%%
import torch
from tqdm import tqdm
import numpy as np

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from matplotlib import pyplot as plt

import torchvision

from collections import deque

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

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

        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=8, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=4, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )

        self.best_model = None
        self.bootstrap = 0
        self.fc1 = torch.nn.Linear(7*8*64, 256)
        self.fc2 = torch.nn.Linear(256, env.action_space.n)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.MSELoss()
        self.to(device)
        self.device = device

    def forward(self, x):
        x = self.feature_extractor(x.to(self.device))
        x = x.reshape(-1, 7*8*64)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def make_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)
    
def pick_action(q_values, epsilon):

    if torch.rand(1) < epsilon:
        return env.action_space.sample()
    else:
        return torch.argmax(q_values).item()

def preprocess_state(state):
    state = torch.tensor(state.copy()).float().permute(2,0,1)

    state = torchvision.transforms.Resize(size=30)(state)

    state /= 255.0
    # state = 0.299*state[:,:,0] + 0.587*state[:,:,1] + 0.114*state[:,:,2]
    state = state.max(dim=0, keepdim=True).values
    return state

done = True
Q = DQN()
Q.load_state_dict(torch.load('Q_target_best.pt'))
episodes = 1000
iterations = 10000
bootstrap = 50
epsilon = 0.05
lr = 0.001
buffer_size = 1000
batch_size = 256

gamma = 0.99

min_dist_req = 40
max_time_for_dist = 100

rewards_history = []
losses_history = []

optimizer = Q.make_optimizer(lr)

buffer = None
episodes = tqdm(range(episodes))

prev_state = preprocess_state(env.reset())

scenes = deque(maxlen=4)
scenes.append(prev_state)
for i in range(3):
    prev_state, _, _, _ = env.step(0)
    prev_state = preprocess_state(prev_state)
    if i < 2:
        scenes.append(prev_state)

last_x_pos = 0

for iteration in tqdm(range(iterations)):

    scenes.append(prev_state)

    prev_state = torch.cat(list(scenes), dim=0)

    with torch.no_grad():
        q_values = Q(prev_state.unsqueeze(0))
        act = q_values.detach().max(dim=-1).indices.item()

    state, reward, done, info = env.step(act)
    print(reward, info['x_pos'])
    state = preprocess_state(state)

    scenes.popleft()
        
    prev_state = state
    env.render()
    if done:
        break
env.close()
     # %%

