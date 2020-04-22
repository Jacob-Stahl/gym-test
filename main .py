import gym
from gym import spaces
import numpy as np
import multiprocessing as mp
from model import Model

env = gym.make('CartPole-v1')

pool_size = 4
epochs = 100
iterations = 512
generation_size = 128
generation = [Model()] * generation_size
spread = 0.1

def next(generation, spread):

    top_model = generation[0]
    record = -1
    generation_size = len(generation)
    for model in generation:
        if model.score > record:
            record = model.score
            top_model = model

    for i in range(generation_size):
        generation[i] = top_model.salt(spread)
        generation[i].score = 0

    return generation, record

def eval(model):

    observation = env.reset()

    for i in range(iterations):
        action = model.forward(observation)
        observation, reward, _, _ = env.step(action)
        model.score += reward
        if abs(observation[0]) > 2:
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
for i in range(epochs):

    if __name__ == '__main__':
        print("epoch : ", i, " top record : ", top_record, " spread : ", spread)
        scored_generation = pool.map(eval, generation)
        generation, record = next(scored_generation, spread)

        if record > top_record:
            spread *= .9
            top_record = record
        else:
            pass

        
view(generation[0])

env.close()