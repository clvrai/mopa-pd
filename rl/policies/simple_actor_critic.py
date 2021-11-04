import torch.nn as nn

class Actor(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def info(self):
        return {}

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
