import gym


class Lidar_gym(gym.Env):

    def __init__(self):
        pass

    def _reset(self):
        pass

    def _close(self):
        super(Lidar_gym, self)._close()

    def _step(self, action):
        pass

    def _render(self, mode='human', close=False):
        super(Lidar_gym, self)._render(mode, close)