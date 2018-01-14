import numpy as np
import gym
import lidar_gym


env = gym.make("lidar-v1")

'''
# stupid input
input_map_size = np.asarray(map_shape) / voxel_size
myMap = np.asarray(np.zeros(input_map_size.astype(int), dtype=int))
borders = [int(len(myMap)*0.3), int(len(myMap)*0.7)]
myMap[:, borders[0]:borders[1], 0] = 1

myDirections = np.zeros(density)
for i in range(density[0]):
    myDirections[i][int(density[1]/2)] = True

print('Testing following ray matrix:\n', myDirections)
'''

done = False
random_action = env.action_space.sample()
episode = 1
env.seed(7)

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
