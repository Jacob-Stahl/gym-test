import gym
from gym import spaces
import numpy as np
import multiprocessing as mp
import pickle as pkl
from model import Model

env = gym.make('CartPole-v1')

pool_size = 32
epochs = 20
iterations = 10000
generation_size = 128
generation = [Model()] * generation_size
base_spread = spread = 0.1
model_name = "model.pickle"

def next(generation, spread, top_model):

    record = top_model.score
    generation_size = len(generation)
    for model in generation:
        if model.score > record:
            record = model.score
            top_model = model

    for i in range(generation_size):
        generation[i] = top_model.salt(spread)
        generation[i].score = 0

    return generation, record, top_model

def eval(model):

    observation = env.reset()

    for i in range(iterations):
        action = model.forward(observation)
        observation, reward, _, _ = env.step(action)
        model.score += reward
        if abs(observation[0]) > 2:
            model.score -= 10
            env.reset()

    return model

def view(model):

    observation = env.reset()

    for i in range(10000):
        action = model.forward(observation)
        observation, _, _, _ = env.step(action)
        
        if abs(observation[0]) > 2:
            env.reset()
        env.render()

pool = mp.Pool(pool_size)

top_record = 0
top_model = Model()
for i in range(epochs):

    if __name__ == '__main__':
        print("epoch : ", i, " top record : ", top_record, " spread : ", round(spread, 6), end = "")
        scored_generation = pool.map(eval, generation)
        generation, record, top_model = next(scored_generation, spread, top_model)
        print(" record : ", record)
        spread = base_spread * ((iterations - record) / iterations)
        if record > top_record:
            top_record = record
        else:
            pass

with open(model_name, 'wb') as handle:
    pkl.dump(top_model, handle)
        
view(top_model)
env.close()