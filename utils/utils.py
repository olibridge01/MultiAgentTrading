from collections import deque, namedtuple
import random

class ExperienceReplay(object):
    def __init__(self, buffer_size: int = 10000):
        self.buffer = deque([], maxlen=buffer_size)
        self.transition = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

    def add_experience(self, *args):
        """Adds an experience to the replay buffer."""
        self.buffer.append(self.transition(*args))

    def sample(self, batch_size: int = 32, separate: bool = True) -> tuple:
        """Samples a batch of experiences from the replay buffer."""
        samples = random.sample(self.buffer, k=batch_size)
        if separate:
            return self.transition(*zip(*samples))
        else:
            return samples

    def __len__(self):
        """Gets length of the replay buffer."""
        return len(self.buffer)
    

class Config(object):
    """
    Configuration object for running experiments. 
    
    Edit to add useful features.
    """
    def __init__(self):
        self.environment = None
        self.GPU = False
        self.hyperparameters = None