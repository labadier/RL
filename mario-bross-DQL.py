#%%
import torch
from tqdm import tqdm
import numpy as np

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from matplotlib import pyplot as plt

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=8, stride=5),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=4, stride=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )

        self.fc1 = torch.nn.Linear(11*12*64, 512)
        self.fc2 = torch.nn.Linear(512, env.action_space.n)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.MSELoss()
        self.to(device)
        

    def forward(self, x):

        x = self.feature_extractor(x/255.0)
        x = x.reshape(1, -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def make_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)
    
def pick_action(q_values, epsilon):

    with torch.no_grad():
        if torch.rand(1) < epsilon:
            return env.action_space.sample()
        else:
            return torch.argmax(q_values).item()

done = True
Q = DQN()
Q_prime = DQN()
episodes = 1000
iterations = 1000
bootstrap = 100
epsilon = 0.1
lr = 0.0001

rewards = []
losses = []

optimizer = Q.make_optimizer(lr)

episodes = tqdm(range(episodes))

for episode in episodes:

    prev_state = torch.tensor(env.reset().copy()).unsqueeze(0).permute(0,3,1,2)
    
    for iteration in range(iterations):

        q_values = Q(prev_state)
        act = pick_action(q_values.detach(), epsilon)

        state, reward, done, info = env.step(act)
        state = torch.tensor(state.copy()).unsqueeze(0).permute(0,3,1,2)

        loss = Q.criterion(reward + torch.max(Q_prime(state).detach()), q_values[0, act])

        losses.append(loss.item())
        rewards.append(reward)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if iteration%bootstrap == 0:
            Q_prime.load_state_dict(Q.state_dict())
            
        prev_state = state
        if done:
            break
    episodes.set_postfix_str(f'epsilon: {epsilon:.3f} reward: {np.mean(rewards[-100:]):.3f}')
env.close()

plt.plot(losses)
plt.show()
 # %%
