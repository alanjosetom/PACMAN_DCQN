
import os
import io
import glob
import torch
import ale_py
import random
import base64
import imageio
import numpy as np
import torch.nn as nn
from PIL import Image
import gymnasium as gym
import torch.optim as optim
from collections import  deque
import torch.nn.functional as F
from torchvision import transforms
from IPython.display import HTML, display
from torch.utils.data import DataLoader, TensorDataset

### Creating the architecture of the Neural Network
class Network(nn.Module):
    def __init__(self,action_size,seed = 42):
        super(Network,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(3,32,kernel_size=8,stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,64,kernel_size=3,stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,128,kernel_size=3,stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128*10*10,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,action_size)

    def forward(self,state):
        x= F.relu(self.bn1(self.conv1(state)))
        x= F.relu(self.bn2(self.conv2(x)))
        x= F.relu(self.bn3(self.conv3(x)))
        x= F.relu(self.bn4(self.conv4(x)))
        x= x.view(x.size(0),-1)
        x= F.relu(self.fc1(x))
        x= F.relu(self.fc2(x))
        x= self.fc3(x)
        return x
    
### Implementing the DCQN class    

class Agent():
    def __init__(self,action_size):
        self.device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.local_network = Network(self.action_size).to(self.device)
        self.target_network = Network(self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_network.parameters(),lr = learning_rate)
        self.memory = deque(maxlen=10000)

    def step(self,state,action,reward,next_state,done):
        state = preprocessing_frame(state)
        next_state = preprocessing_frame(next_state)
        self.memory.append((state,action,reward,next_state,done))
        if len(self.memory) >minibatch_size:
            experience = random.sample(self.memory,k=minibatch_size)
            self.learn(experience,discount_factor)

    def act(self, state, epsilon = 0.):
        state = preprocessing_frame(state).to(self.device)
        self.local_network.eval()
        with torch.no_grad():
            action_values = self.local_network(state)
        self.local_network.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))


    def learn(self,experience,discount_factor):
        states,actions,rewards,next_states,dones = zip(*experience)
        states = torch.from_numpy(np.vstack([s.cpu().numpy() for s in states])).float().to(self.device) # Fix: Convert states to numpy array before stacking
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([s.cpu().numpy() for s in next_states])).float().to(self.device) # Fix: Convert next_states to numpy array before stacking
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)
        next_q_target = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        q_target = rewards + (discount_factor * next_q_target * (1-dones))
        q_expected = self.local_network(states).gather(1,actions)
        loss = F.mse_loss(q_expected,q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

### Preprocessing the frames

def preprocessing_frame(frame):
    frame = Image.fromarray(frame)
    processes = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])
    return processes(frame).unsqueeze(0)

def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action)
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)

def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

### Setting up the environment

env = gym.make('ALE/MsPacman-v5', render_mode='rgb_array')
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

### Initializing the hyperparameters

learning_rate = 5e-4
minibatch_size = 64
discount_factor = 0.99



number_episodes = 2000
max_num_steps_per_episode = 10000
epsilon_starting_value = 1.0
epsilon_ending_value = 0.01
epsilon_decay_value = 0.995
epsilon = epsilon_starting_value
score_100_episode = deque(maxlen= 100)


### Initializing the DCQN agent

agent = Agent(action_size)

### Training the DCQN agent

for episode in range(1,number_episodes+1):
    state, _ = env.reset()
    score = 0
    for t in range(max_num_steps_per_episode):
        action = agent.act(state,epsilon)
        next_state,reward,done,_,_ = env.step(action)
        agent.step(state,action,reward,next_state,done)
        state = next_state
        score += reward
        if done:
            break
    score_100_episode.append(score)
    epsilon = max(epsilon_ending_value,epsilon_decay_value*epsilon)
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode,np.mean(score_100_episode)),end="")
    if episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode,np.mean(score_100_episode)))
    if np.mean(score_100_episode)>=500:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100,np.mean(score_100_episode)))
        torch.save(agent.local_network.state_dict(),'checkpoint.pth')
        break



###visualizing the result

show_video_of_model(agent, 'MsPacmanDeterministic-v0')
show_video()