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
    state = state.max(dim=0, keepdim=True).values
    return state

done = True
Q = DQN()
Q_prime = DQN()
episodes = 100
iterations = 10000
bootstrap = 100
epsilon = 0.05
lr = 0.0001
buffer_size = 1000
batch_size = 256

gamma = 0.99

min_dist_req = 23
max_time_for_dist = 40


losses_history = []

optimizer = Q.make_optimizer(lr)

buffer = None


for episode in range(episodes):

    print(f'Episode: {episode}')

    scenes = deque(maxlen=4)
    prev_state = preprocess_state(env.reset())
    scenes.append(prev_state)

    for i in range(3):
        prev_state, _, _, _ = env.step(0)
        prev_state = preprocess_state(prev_state)
        if i < 2:
            scenes.append(prev_state)
    
    last_x_pos = 0
    iter = tqdm(range(iterations))
    rewards_history = []
    for iteration in iter:

        scenes.append(prev_state)

        prev_state = torch.cat(list(scenes), dim=0)

        with torch.no_grad():
            q_values = Q(prev_state.unsqueeze(0))
            act = pick_action(q_values.detach(), epsilon)
        
        rw = []
        for k in range(5):
            state, reward, done, info = env.step(act)
            rw.append(reward)
            if done:
                break   
        
        reward = sum(rw)
        rewards_history.append(reward)
        state = preprocess_state(state)

        scenes.popleft()
        new_state = torch.cat(list(scenes) + [state], dim=0)
      
        if buffer is None:
            buffer = ReplayBuffer(buffer_size, prev_state.shape)
        buffer.store_transition(prev_state, act, reward, new_state, not done)

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
                Q.bootstrap = 0 
                Q_prime.load_state_dict(Q.state_dict())

                if Q.best_model is None or Q.best_model < np.sum(rewards_history):
                    Q.best_model = np.sum(rewards_history)
                    torch.save(Q.state_dict(), 'Q_target_best.pt')
                    print(f'Best reward: {Q.best_model:.3f}')
                torch.save(Q_prime.state_dict(), 'Q_target.pt')

            if Q.best_model is not None and Q.best_model < info['x_pos']:
                torch.save(Q.state_dict(), 'Q_target_best.pt')
                print(f'Best reward: {Q.best_model:.3f}')
                Q.best_model = info['x_pos']
            
        prev_state = state

        if iteration and iteration%max_time_for_dist == 0:
            if info['x_pos'] - last_x_pos < min_dist_req:
                done = True
                print('too slow')
            else:
                last_x_pos = info['x_pos']

        if done:
            break
        iter.set_postfix_str(f'epsilon: {epsilon:.3f} reward: {np.sum(rewards_history):.3f} x_pos: {info["x_pos"]:.3f}')

    mean_rwh = np.sum(rewards_history)
    # if Q.best_model is None or Q.best_model < mean_rwh:
    #     Q.best_model = mean_rwh
    #     torch.save(Q.state_dict(), 'Q_target_best.pt')
    #     print(f'Best reward: {Q.best_model:.3f}')
env.close()

plt.plot(losses_history)
plt.show()
     # %%

