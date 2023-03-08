import numpy as np

class Gates:

    def __init__(self, gate_type, gate_name):
        
        self.gate_name = gate_name # XOR, OR, etc.
        self.gate_type = gate_type # probabilistic, deterministic, special
        self.prob_table = np.array([])
        self.connections = [] # IDs of connections

    def evaluate(self, inputs):
        output = None

        ## Probabilistic Gates ##
        if self.gate_type == 'probabilistic':

            pass

        ## Determinstic Gates ## 
        elif self.gate_type == 'deterministic':

            pass

        ## Special Gates ##
        elif self.gate_type == 'special':

            pass

        else:
            print('Invalid gate type: ' + str(self.gate_type))

        return output