#%%
import random; random.seed(0)
import numpy as np; np.random.seed(0)
import torch; torch.manual_seed(0)
from tqdm import tqdm
import gymnasium as gym


from matplotlib import pyplot as plt

from collections import deque

# env = gym.make('MountainCar-v0')
env = gym.make('CartPole-v1')


class PModel(torch.nn.Module):
    def __init__(self, lr, temperature=1.0):
        super(PModel, self).__init__()

        self.bootstrap = 0
        self.fc1 = torch.nn.Linear(env.observation_space.shape[0], 32)
        self.fc2 = torch.nn.Linear(32, env.action_space.n)

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        self.to(device)
        self.device = device

        self.mean_rewards = 0
        self.temperature = temperature

        self.optimizer = self.make_optimizer(lr)
        

    def forward(self, x):
        
        x = torch.nn.functional.relu(self.fc1(x.to(self.device)))
        x = torch.functional.F.softmax(self.fc2(x) / self.temperature, dim=-1)
        return x
    
    def make_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
    
class DVN(torch.nn.Module):

    def __init__(self, lr):
        super(DVN, self).__init__()

        self.bootstrap = 0
        self.fc1 = torch.nn.Linear(env.observation_space.shape[0], 32)
        self.fc2 = torch.nn.Linear(32, 1)

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.criterion = torch.nn.MSELoss()
        self.to(device)
        self.device = device

        self.optimizer = self.make_optimizer(lr)
    
    def forward(self, x):
        
        x = torch.nn.functional.relu(self.fc1(x.to(self.device)))
        x = self.fc2(x)
        return x
    
    def make_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)

def preprocess_state(state):
    state = torch.tensor(state).float()
    return state


def update_actor_critic(buffer, gamma):

        
    rewards = torch.tensor([x['reward'] for x in buffer]).to(device=Actor.device).reshape(-1, 1)
    state = torch.cat([x['state'] for x in buffer]).to(device=Actor.device)
    next_state = torch.cat([x['next_state'] for x in buffer]).to(device=Actor.device)
    action = torch.stack([x['action'] for x in buffer]).to(device=Actor.device).reshape(-1, 1)
    dones = torch.tensor([float(x['end_state']) for x in buffer]).to(device=Actor.device).reshape(-1, 1)
    

    values = Critic(state)
    next_values = Critic(next_state).detach()

    loss_critic = Critic.criterion(rewards + gamma*next_values*(1 - dones), values)
    
    log_probs = torch.log(Actor(state).gather(1, action))
    loss_actor = -torch.mean(log_probs* (rewards + gamma*next_values*(1 - dones) - values).detach())


    loss_critic.backward()
    loss_actor.backward()

    Actor.optimizer.step()
    Actor.optimizer.zero_grad()

    Critic.optimizer.step()
    Critic.optimizer.zero_grad()

    return loss_critic.item(), loss_actor.item()


#compute entropy
def compute_entropy(distribution):

    return np.sum(-distribution*np.log(distribution))

episodes = 0
iterations = 1000
lr_critic = 0.01
lr_actor = 0.001
batch_size = 32

gamma = 0.99

rewards_history = []
losses_history_q = []
losses_history_p = []

Actor = PModel(lr_actor, temperature=5.0)
Critic = DVN(lr_critic)


buffer = []
entropies = []

for episode in range(episodes):

    prev_state, _= env.reset()
    prev_state = preprocess_state(prev_state)

    itera = tqdm(range(iterations))

    for iteration in itera:

        v_values = Critic(prev_state.unsqueeze(0))
        preferences = Actor(prev_state.unsqueeze(0))
        action = torch.tensor(np.random.choice(np.arange(preferences.shape[1]), p=preferences.cpu().detach().numpy().ravel()), requires_grad=False)
     
        entropies = entropies[-100:] + [compute_entropy(preferences.cpu().detach().numpy().ravel())]
        
        if iteration % 50 == 0:
            Actor.temperature = max(0.1, Actor.temperature * 0.995) 
            
        state, reward, done, _, info = env.step(action.item())

        state = preprocess_state(state)
        v_values_1 = Critic(state.unsqueeze(0)).detach()

        buffer.append({'reward': reward, 
                       'state': prev_state.unsqueeze(0), 
                       'next_state':state.unsqueeze(0), 
                       'action':action,
                       'end_state': done})

        rewards_history.append(reward)
        if len(buffer) > batch_size:
            loss_q, loss_p = update_actor_critic(buffer, gamma)
            buffer = buffer[1:]

            losses_history_q.append(loss_q)
            losses_history_p.append(loss_p)
            
    
        prev_state = state

        if done:
            print('Finished')
            break
        itera.set_postfix_str(f'reward: {np.mean(rewards_history[-100:]):.3f} loss_critic:{np.mean(losses_history_q[-200:]):.3f} loss_actor:{np.mean(losses_history_p[-200:]):.3f} temperature: {Actor.temperature:.3f} entropy: {np.mean(entropies)}')
    if episode % 10 == 0:
        torch.save(Critic.state_dict(), 'V_target.pt')
        torch.save(Actor.state_dict(), 'P.pt')
            
        
env.close()

plt.plot(losses_history_q, label='V function')
plt.plot(losses_history_p, label='P function')

plt.show()

# Q.load('Q_target.pt')
Actor.load('P.pt')
env = gym.make('CartPole-v1', render_mode='human')
prev_state, info = env.reset()
prev_state = preprocess_state(prev_state)

for _ in range(int(1e4)):

    preferences = Actor(prev_state.unsqueeze(0))
    preferences = torch.distributions.Categorical(preferences)
    
    action = preferences.sample()

    observation, reward, terminated, _, _ = env.step(action.item())
    print(action)

    prev_state = preprocess_state(observation)
    prev_action = action

    if terminated:
        break

env.close()



# %%
