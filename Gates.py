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

        # Prob gates have higher than 0.5 probability of firing if output is supposed to be 1 and less than 0.5 if it should be 0

        # initialize probabilistic gate
        if self.gate_type == 'probabilistic':
            self.input_prob = np.array([]) # unused for now - can implement later

            self.truth_table = np.array(list(product([False, True], repeat=self.num_inputs))) # contains all permutations of the inputs

            # Randomly initialize probabilities
            probs_high = np.random.uniform(0.5, 1, self.truth_table.shape[0])
            probs_low = np.random.uniform(0, 0.5, self.truth_table.shape[0])

            # Need to initialize these based on specific gates

            if self.gate_name == 'AND': # n inputs
                self.output_prob = np.where(np.sum(self.truth_table, axis=1) == self.num_inputs, probs_high, probs_low)
            elif self.gate_name == 'OR': # n inputs
                self.output_prob = np.where(np.sum(self.truth_table, axis=1) > 0, probs_high, probs_low)
            elif self.gate_name == 'NOT': # 1 input
                self.output_prob = np.where(self.truth_table[:,0] == False, probs_high, probs_low)
            elif self.gate_name == 'NAND': # n inputs
                self.output_prob = np.where(np.sum(self.truth_table, axis=1) != self.num_inputs, probs_high, probs_low)
            elif self.gate_name == 'NOR': # n inputs
                self.output_prob = np.where(np.sum(self.truth_table, axis=1) == 0, probs_high, probs_low)
            elif self.gate_name == 'XOR': # 2 inputs
                self.output_prob = np.where(np.logical_xor(self.truth_table[:,0], self.truth_table[:,1]), probs_high, probs_low)
            elif self.gate_name == 'XNOR': # 2 inputs
                self.output_prob = np.where(np.logical_not(np.logical_xor(self.truth_table[:,0], self.truth_table[:,1])), probs_high, probs_low)

        ## DETERMINISTIC GATE SETUP ## (special case of probabilistic)

        elif self.gate_type == 'deterministic':
            self.input_prob = np.array([]) # unused for now - can implement later

            self.truth_table = np.array(list(product([False, True], repeat=self.num_inputs))) # contains all permutations of the inputs

            # Need to initialize these based on specific gates

            if self.gate_name == 'AND': # n inputs
                self.output_prob = np.where(np.sum(self.truth_table, axis=1) == self.num_inputs, 1, 0)
            elif self.gate_name == 'OR': # n inputs
                self.output_prob = np.where(np.sum(self.truth_table, axis=1) > 0, 1, 0)
            elif self.gate_name == 'NOT': # 1 input
                self.output_prob = np.where(self.truth_table[:,0] == False, 1, 0)
            elif self.gate_name == 'NAND': # n inputs
                self.output_prob = np.where(np.sum(self.truth_table, axis=1) != self.num_inputs, 1, 0)
            elif self.gate_name == 'NOR': # n inputs
                self.output_prob = np.where(np.sum(self.truth_table, axis=1) == 0, 1, 0)
            elif self.gate_name == 'XOR': # 2 inputs
                self.output_prob = np.where(np.logical_xor(self.truth_table[:,0], self.truth_table[:,1]), 1, 0)
            elif self.gate_name == 'XNOR': # 2 inputs
                self.output_prob = np.where(np.logical_not(np.logical_xor(self.truth_table[:,0], self.truth_table[:,1])), 1, 0)

        else:

            print('Illegal gate type!')
            exit()


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
            return self.output_prob[np.where((self.truth_table == inputs).all(axis=1))]

        ## Special Gates ##
        elif self.gate_type == 'special':

            pass

        else:
            print('Invalid gate type: ' + str(self.gate_type))