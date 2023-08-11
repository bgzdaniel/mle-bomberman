from collections import namedtuple

Parameters = namedtuple('Parameters', ['batch_size', 'memory_size'])

hp = Parameters(batch_size=32, 
                memory_size=1000)

