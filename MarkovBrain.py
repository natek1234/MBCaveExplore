import numpy as np
from Gates import Gates

class MarkovBrain:
    def __init__(self, num_inputs, num_outputs, num_hidden, gates_active, gates_exclude, gates):

        # Markov Brain structure
        self.inputs = np.zeros(num_inputs)
        self.outputs = np.zeros(num_outputs)
        self.num_hidden = np.zeros(num_hidden)

        # Markov Brain gates
        self.gate_types = gates
        # List of Gates objects
        self.gates = []

        # Fitness
        self.fitness = 0 # starts at neutral point
    
    # Random initialization of gates and states
    def initialize(self):

        return
    
    # Perform brain update, including fitness update
    def brain_update(self):

        return
    
    # Optional: encode as genome
    def encode(self):

        return
    
    # Optional: decode genome
    def decode(self):

        return