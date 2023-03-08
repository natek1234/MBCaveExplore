import numpy as np
from Gates import Gates

class MarkovBrain:
    def __init__(self, num_inputs, num_outputs, num_hidden, num_gates, gates):

        # Markov Brain structure
        # Input states
        self.inputs = np.zeros(num_inputs)
        self.num_inputs = num_inputs
        self.input_ids = list(range(0,num_inputs))

        # Output states
        self.outputs = np.zeros(num_outputs)
        self.num_outputs = num_outputs
        self.output_ids = list(range(num_inputs, num_inputs+num_outputs))

        # Hidden states
        self.hidden = np.zeros(num_hidden)
        self.num_hidden = num_hidden
        self.output_ids = list(range(num_inputs+num_outputs, num_inputs+num_outputs+num_hidden))

        # Markov Brain gates
        self.gate_types = gates
        # List of Gates objects
        self.num_gates = num_gates
        self.gates = []

        self.ids = num_inputs + num_outputs + num_hidden # by definition, make the first x number of IDs for the inputs, outputs, and hidden states

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