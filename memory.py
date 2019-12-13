import random
from collections import namedtuple


class ReplayMemory(object):

    def __init__(self, capacity=100):
        self.capacity = capacity
        self.memory = {}
        self.position = {}
        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward', 'done'))

    def push(self, state, action, next_state, reward, done, gameName):
        """Saves a transition."""
        if len(self.memory[gameName]) < self.capacity:
            self.memory[gameName].append(None)
        self.memory[gameName][self.position[gameName]] = self.Transition(state, action, next_state, reward, done)
        self.position[gameName] = (self.position[gameName] + 1) % self.capacity

    def sample(self, batch_size, gameName):
        return random.sample(self.memory[gameName], batch_size)

    def len(self, gameName):
        return len(self.memory[gameName])

    def add_env(self, gameName):
        self.memory[gameName] = []
        self.position[gameName] = 0
