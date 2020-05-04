import collections
import numpy as np

Trajectory = collections.namedtuple('Trajectory', 'state action reward')


class ReplayBuffer(object):
    def __init__(self, max_size=int(1e6)):
        self.max_size = max_size
        self.size = 0

        self.state = []
        self.action = []
        self.reward = []
        self.not_done = []

    def add(self, trajectory: Trajectory):
        while self.size > self.max_size:
            self.size -= len(self.reward.pop(0))
            self.action.pop(0)
            self.state.pop(0)
            self.not_done.pop(0)

        self.state.append(trajectory.state)
        self.action.append(trajectory.action)
        self.reward.append(trajectory.reward)
        not_done = np.ones_like(trajectory.reward)
        not_done[-1] = 0
        self.not_done.append(not_done)
        self.size += len(trajectory.reward)
