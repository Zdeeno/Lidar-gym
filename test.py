import numpy as np
import gym
import lidar_gym

env = gym.make('lidar-v0')
done = False

episode = 1
env.seed(2)
space = env.action_space

print(env.observation_space)

while True:

    counter = 1
    obv = env.reset()
    print('------------------- Episode number', episode, '-------------------------')
    print('------------------- Iteration number 0 -------------------------')
    print('Observation:\n', obv)

    while not done:
        print('------------------- Episode number', episode, '-------------------------')
        print('------------------- Iteration number ', counter, '-------------------------')

        # obv, reward, done, info = env.step({"rays": myDirections, "map": myMap})
        obv, reward, done, info = env.step(obv)
        print('Observation:\n', len(obv[obv != 0]))
        print('\nreward:\n', reward, '\n')

        env.render()
        counter += 1

    episode += 1
    done = False
