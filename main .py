import gym
from gym import spaces
import torch
import numpy as np
from torch import optim
from model import Model

def to_binary(x):
    if x > 0.5:
        return 1
    else:
        return 0

env = gym.make('CartPole-v1')
env.reset()
observation = env.reset()
model = Model(input_dim = 4, output_dim = 1)
opt = optim.Adam(model.parameters())
criterion = torch.nn.MSELoss()
iterations = 3000

print(env.action_space.sample())


for i in range(iterations):
    action = model(torch.from_numpy(observation).float())
    bi_action = to_binary(action.detach().numpy()[0])
    observation, reward, done, info = env.step(bi_action)
    loss = criterion(action, action * reward + 1)
    loss.backward()
    opt.step()
    env.render()
    if abs(observation[0]) > 2:
        env.reset()
    
    if i % 100 == 0:
        print("iteration : ", i)

env.close()