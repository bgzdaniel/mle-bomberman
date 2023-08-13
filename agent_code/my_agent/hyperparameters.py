import torch
from collections import namedtuple

Parameters = namedtuple('Parameters', ['batch_size', 
                                       'memory_size', 
                                       'epsilon', 
                                       'gamma', 
                                       'discount', 
                                       'update_frequency',
                                       'feature_shape'])

hp = Parameters(batch_size=32, 
                memory_size=1000,
                epsilon=0.1,
                gamma=0.6,
                discount=0.9,
                update_frequency=2,
                feature_shape = torch.Size([17**2])
                )

