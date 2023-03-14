import numpy as np
from itertools import product

N_INPUT_GATES = ['AND', 'OR', 'NAND', 'NOR']
TWO_INPUT_GATES = ['XOR', 'XNOR']
ONE_INPUT_GATES = ['NOT']

class Gates:

    def __init__(self, gate_type, gate_name, id, total_inputs, input_ids, output_ids, hidden_ids):
        # Gate type = Probablistic, Deterministic, or Special
        # Gate name = type of gate
        
        ## GATE SETUP ##
        self.gate_name = gate_name # XOR, OR, etc.
        self.gate_type = gate_type # probabilistic, deterministic, special
        self.id = id

        # initialize number of inputs
        if self.gate_name in N_INPUT_GATES:
            self.num_inputs = np.random.randint(2, total_inputs)
        elif self.gate_name in TWO_INPUT_GATES:
            self.num_inputs = 2
        else:
            self.num_inputs = 1

        ## PROBABILISTIC GATE SETUP ##

        # initialize probabilistic gate
        if self.gate_type == 'probabilistic':
            self.input_prob = np.array([]) # unused for now - can implement later

            self.truth_table = np.array(list(product([False, True], repeat=self.num_inputs))) # contains all permutations of the inputs

            # Need to initialize these based on specific gates
            ## DO NEXT ##
            self.output_prob = np.array([]) # contains all the output probabilities, ordered according to the truth table

        ## CONNECTIONS SETUP ##
        self.connections = [] # IDs of connections

    # Inputs are assumed to be 1 to n input entries (depending on gate type)
    def evaluate(self, inputs):

        ## Probabilistic Gates ##
        if self.gate_type == 'probabilistic':

            ind = np.where((self.truth_table == inputs).all(axis=1)) # find location in truth table
            rand = np.random.rand() # generate a random number between 0 and 1

            return rand <= self.output_prob[ind] # if random number is smaller than the output probability, output is 1

        ## Determinstic Gates ## 
        elif self.gate_type == 'deterministic':
            if self.gate_type == 'AND': # n inputs
                return np.all(inputs)
            elif self.gate_type == 'OR': # n inputs
                return np.any(inputs)
            elif self.gate_type == 'NOT': # 1 input
                return not inputs[0]
            elif self.gate_type == 'NAND': # n inputs
                return np.logical_not(np.all(inputs))
            elif self.gate_type == 'NOR': # n inputs
                return np.logical_not(np.any(inputs))
            elif self.gate_type == 'XOR': # 2 inputs
                return np.logical_xor(inputs[0],inputs[1])
            elif self.gate_type == 'XNOR': # 2 inputs
                return np.logical_not(np.logical_xor(inputs[0],inputs[1]))

        ## Special Gates ##
        elif self.gate_type == 'special':

            pass

        else:
            print('Invalid gate type: ' + str(self.gate_type))