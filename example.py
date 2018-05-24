import numpy as np
import gym
import lidar_gym

env = gym.make('lidar-v1')

done = False
random_action = env.action_space.sample()
episode = 1
env.seed(7)

print(env.action_space)

while True:

    counter = 1
    obv = env.reset()
    print('------------------- Episode number', episode, '-------------------------')
    print('------------------- Iteration number 0 -------------------------')
    print('Observation:\nNext positions:\n', obv['T'], '\nPoints:\n', obv['points'], '\nValues:\n', obv['values'])

    while not done:
        print('------------------- Episode number', episode, '-------------------------')
        print('------------------- Iteration number ', counter, '-------------------------')

        # obv, reward, done, info = env.step({"rays": myDirections, "map": myMap})
        obv, reward, done, info = env.step(random_action)
        print('Observation:\nNext positions:\n', obv['T'], '\nPoints:\n', obv['points'], '\nValues\n', obv['values'])
        print('\nHited ', np.shape(np.where(obv['values'] == 1))[1], ' points!\n')
        print('reward:\n', reward, '\n')

        env.render()
        counter += 1

    episode += 1
    done = False
