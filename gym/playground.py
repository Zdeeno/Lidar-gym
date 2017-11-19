import lidar_gym as gym


# create environment
env = gym.LidarGym()
done = False
obv = env.reset()


while not done:
    _, _, _, _ = env.step(None)

