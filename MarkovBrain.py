import numpy as np
from Gates import Gates
import math

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

    # NOTE: This particular brain update is specific to this simulation. 
    # Input 1: Local temperature
    # Input 2: Distance to closest agent

    # Output 1-4: up, down, left, and right movement (in that order)
    def brain_update(self, cave_map, cave_map_grad, other_agents, this_agent, time_of_day = 'DAY'):

        ## INPUT LAYER ##

        # Local temperature
        self.inputs[0] = cave_map[this_agent]

        # Distance to closest agent
        self.inputs[0] = None

        ## LOGIC LAYER ##


        ## OUTPUT LAYER ##

        # This section enforces movement conditions, including map limits
        
        curr_y = this_agent[0]
        curr_x = this_agent[1]

        # Move up
        if self.outputs[0] >= 1 and this_agent[0] != 0:
            curr_y = curr_y - 1

        # Move down
        if self.outputs[1] >= 1 and this_agent[0] != cave_map.shape[0] - 1:
            curr_y = curr_y + 1

        # Move left 
        if self.outputs[2] >= 1 and this_agent[1] != 0:
            curr_x = curr_x - 1
        
        # Move right
        if self.outputs[3] >= 1 and this_agent[1] != cave_map.shape[1] - 1:
            curr_x = curr_x + 1

        self.location = (curr_y, curr_x)

        ## UPDATE FITNESS ##

        # Fitness rule : promote temperature gradients
        if time_of_day == 'DAY':
            # Promote negative gradients (lower temperatures in caves)
            self.fitness = self.fitness - cave_map_grad[0][this_agent] - cave_map_grad[1][this_agent] # y and x derivatives
        else:
            # Promote positive gradients (higher temperatures in caves)
            self.fitness = self.fitness + cave_map_grad[0][this_agent] + cave_map_grad[1][this_agent] # y and x derivatives

        # Fitness rule : promote against agents crashing (maintain separation)
        for agent in other_agents:
            d = math.dist(agent, this_agent)
            if d < 2: # covers all locations directly around the current location
                self.fitness = self.fitness - 30 # harsh penalty if there is a crash
        
        return self.location, self.fitness # return the new position of the agent
    
    # Optional: encode as genome
    def encode(self):

        return
    
    # Optional: decode genome
    def decode(self):

        return