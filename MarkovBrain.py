import numpy as np
from Gates import Gates

class MarkovBrain:
    def __init__(self, num_inputs, num_outputs, num_hidden, num_gates, gates, all_gates = True):

        ## MARKOV BRAIN STRUCTURE ##
        # Input states
        self.inputs = np.zeros(num_inputs) # init inputs to 0
        self.num_inputs = num_inputs
        self.input_ids = list(range(0,num_inputs))

        # Output states
        self.outputs = np.zeros(num_outputs)
        self.num_outputs = num_outputs
        self.output_ids = list(range(num_inputs, num_inputs+num_outputs))

        # Hidden states
        self.hidden = np.zeros(num_hidden)
        self.num_hidden = num_hidden
        self.hidden_ids = list(range(num_inputs+num_outputs, num_inputs+num_outputs+num_hidden))

        # Markov Brain gates
        self.gate_types = gates
        # List of Gates objects
        self.num_gates = num_gates
        self.gate_ids = list(range(num_inputs+num_outputs+num_hidden, num_inputs+num_outputs+num_hidden+num_gates))
        self.gates = []

        # Randomly initialize gates to probabilistic, deterministic, or special
        gate_type = np.random.choice(['probabilistic','deterministic'], self.num_gates) # 0 is probabilistic, 1 is deterministic, 2 is special (to be added later)
        for count, gate in enumerate(gate_type):
            self.gates = self.gates + [Gates(gate, np.random.choice(self.gate_types), self.gate_ids[count], num_inputs+num_hidden, self.input_ids, self.output_ids, self.hidden_ids)] # Add Gates objects to MB
            # Need to add separate logic for special gates

        self.ids = self.input_ids + self.output_ids + self.hidden_ids + self.gate_ids # by definition, make the first x number of IDs for the inputs, outputs, hidden states, and gates

        ## SIMULATION PROPERTIES ##
        # Fitness
        self.fitness = 0 # starts at neutral point
        # Location
        self.location = None

    # Perform brain update, including fitness update
    def brain_update(self):

        return
    
    # Optional: encode as genome
    def encode(self):

        return
    
    # Optional: decode genome
    def decode(self):

        return