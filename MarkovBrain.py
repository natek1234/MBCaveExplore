import numpy as np
from Gates import Gates
import math

CAVES = [[62,458],[232,318],[193,271],[451,333],[422,47],[429,382]] # x, y
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

        self.ids = self.input_ids + self.output_ids + self.hidden_ids # by definition, make the first x number of IDs for the inputs, outputs, hidden states

        ## SIMULATION PROPERTIES ##
        # Fitness
        self.fitness = 0 # starts at neutral point
        # Location
        self.location = None
        self.start_loc = None

    # Perform brain update, including fitness update

    # NOTE: This particular brain update is specific to this simulation. 
    # Input 1-4: Local temperature gradient (up, down, left, right)
    # Input 5: Distance to closest agent

    # Output 1-4: up, down, left, and right movement (in that order)
    def brain_update(self, cave_map, cave_map_grad, other_agents, this_agent, time_of_day = 'DAY', first_iter = False, last_iter = False):

        ## INPUT LAYER ##

        # Local temperature gradient information (4 pixels in each direction)
        # IDEA: local average of gradient in x and y direction gives some directional information of where the gradient is leading

        y, x = this_agent # extract coordinates
        #y,x = (280,170) # for testing (negative y)
        #y,x = (310,170) # for testing (positive y)
        #y,x = (300,165) # for testing (negative x and positive y)

        # Determine the size of the array
        height, width = cave_map.shape

        # Create a mask of the same size as the map array
        local_mask = np.zeros((height, width), dtype=np.bool)

        # Consider x and y indices within 2 pixels of the agent
        x_min = max(0, x - 2)
        x_max = min(width, x + 3)
        y_min = max(0, y - 2)
        y_max = min(height, y + 3)

        # Set the values within the range to True in the mask
        local_mask[y_min:y_max, x_min:x_max] = True

        local_y_grad = np.mean(cave_map_grad[0][local_mask])
        local_x_grad = np.mean(cave_map_grad[1][local_mask])
        
        # INPUT 1 and 2: Look up and down
        if local_y_grad > 0.1:
            self.inputs[0] = 1
            self.inputs[1] = 0
        elif local_y_grad < -0.1:
            self.inputs[0] = 0
            self.inputs[1] = 1
        else:            
            self.inputs[0] = 0
            self.inputs[1] = 0

        # INPUT 3 and 4: Look left and right
        if local_x_grad > 0.1:
            self.inputs[2] = 1
            self.inputs[3] = 0
        elif local_x_grad < -0.1:
            self.inputs[2] = 0
            self.inputs[3] = 1 
        else:  
            self.inputs[2] = 0
            self.inputs[3] = 0 

        # Distance to closest agent - 1 if the distance is less than 3 pixels, 0 otherwise
        closest_agent = [math.dist(agent, this_agent) for agent in other_agents]
        
        self.inputs[4] = 1 if min(closest_agent) <= 3 else 0

        ## LOGIC LAYER ##
        gate_outputs = np.zeros(self.num_gates) # initialize a set of gate outputs
        curr_state = np.concatenate((self.inputs, self.outputs, self.hidden)) # extract current state (for easy gate evaluation)
        self.outputs = np.zeros(self.num_outputs) # reset outputs (blank slate)

        # Loop over every gate and evaluate
        for i in range(self.num_gates):

            input_ids = np.in1d(self.ids, self.gates[i].input_connections) # identify input IDs
            inputs = curr_state[input_ids] # extract input values
            gate_outputs[i] = self.gates[i].evaluate(inputs) # evaluate based on inputs

            output_ids = np.in1d(self.ids, self.gates[i].output_connections)[self.num_inputs:self.num_inputs+self.num_outputs] # identify output IDs
            self.outputs[output_ids] = np.logical_or(self.outputs[output_ids], gate_outputs[i])

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
            self.fitness = self.fitness + (np.abs(local_x_grad) + np.abs(local_y_grad)) # y and x derivatives
        else:
            # Promote positive gradients (higher temperatures in caves)
            self.fitness = self.fitness + cave_map_grad[0][this_agent] + cave_map_grad[1][this_agent] # y and x derivatives

        # Fitness rule : promote against agents crashing (maintain separation)
        for d in closest_agent:
            if d < 2: # covers all locations directly around the current location
                self.fitness = self.fitness - 30 # harsh penalty if there is a crash

        if first_iter == True:
            self.start_loc = this_agent

        # Fitness rule : promote ending close to the nearest cave (within acceptable threshold)
        if last_iter:
            if this_agent == self.start_loc: # completely reject all agents that don't move at all
                self.fitness = -np.inf
                
            # Distance to closest cave 
            #closest_agent = [math.dist(cave, this_agent) for cave in CAVES]
            
            #if min(closest_agent) >= 5: # Only activate penalty if further than 5 pixels (10m)
            #    self.fitness = self.fitness - 10*min(closest_agent)
        

        
        return self.location, self.fitness # return the new position of the agent
    
    # Optional: encode as genome
    def encode(self):

        return
    
    # Optional: decode genome
    def decode(self):

        return