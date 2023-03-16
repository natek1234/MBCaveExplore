import yaml
import numpy as np
from MarkovBrain import MarkovBrain
import matplotlib.pyplot as plt
import copy

VISUALIZE = False # Visualization option to be implemented

# Perform mutation on a brain
def mutate(brain):

    return True

# Perform crossover on a brain
def crossover(brain_1, brain_2):

    return True

def selection(brains):

    best_brains = None

    return best_brains

if __name__ == '__main__':

    ## Open Configuration File ##

    stream = open("mb_config.yaml", 'r')
    params = yaml.safe_load(stream) # all config parameters saved in params

    for key, value in params.items():
        print (key + " : " + str(value))

    ## SETUP SIM ENVIRONMENT ##

    # Create map
    cave_map = np.loadtxt(params['maps'][0])

    ## GENERATE FIRST GENERATION ##

    agents = []
    # Initialize first pool of Markov Brains
    for agent in range(0, params['pool_size']):
        agents = agents + [MarkovBrain(params['num_inputs'], params['num_outputs'], params['num_hidden'], params['num_gates'], params['gate_types'])]

    print(agents[0].ids)
    for i in range(0, agents[0].num_gates):
        print(agents[0].gates[i].gate_type)
        print(agents[0].gates[i].gate_name)
        print(agents[0].gates[i].id)
        print(agents[0].gates[i].num_inputs)
        print(agents[0].gates[i].truth_table)
        print(agents[0].gates[i].output_prob)
        print(agents[0].gates[i].input_connections)
        print(agents[0].gates[i].output_connections)

    ## EVOLUTION PROCESS ## 

    for evo_step in range(0, params['evolution_steps']):

        # This loop goes over every agent variation for this evolution step
        for agent in agents:

            # Create simulation pool of agents
            sim_agents = [] # stores copies of the simulation agent
            for i in range(0, params['swarm_size']):
                sim_agents = sim_agents + [copy.deepcopy(agent)]
        
            ## SIMULATION PROCESS ##
            for sim_step in range(0, params['time_steps']):

                # Complete the simulation process (brain updates, movement, etc.)

                pass

        # After all agents have been simulated, compare fitness and evolve

        ## SELECTION ## 

        # Select best performing brains as parents

        ## GENERATE ##

        # Create next generation of brains

        # Crossover

        # Mutation

        ## RESET ##

        # Reset map environment 