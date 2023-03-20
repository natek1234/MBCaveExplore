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

def simulate_step(brain, brain_locations, map):
    
    return True

if __name__ == '__main__':

    ## Open Configuration File ##

    stream = open("mb_config.yaml", 'r')
    params = yaml.safe_load(stream) # all config parameters saved in params

    for key, value in params.items():
        print (key + " : " + str(value))

    ## SETUP SIM ENVIRONMENT ##

    # Create map
    cave_map = np.loadtxt(params['maps'][0])
    center = tuple(i//2 for i in cave_map.shape) # extract center coordinate
    cave_map_grad = np.gradient(cave_map) # pre-compute gradient of cave_map

    ## GENERATE FIRST GENERATION ##

    agents = []
    fitness = np.zeros(params['pool_size'])
    # Initialize first pool of Markov Brains
    for agent in range(0, params['pool_size']):
        agents = agents + [MarkovBrain(params['num_inputs'], params['num_outputs'], params['num_hidden'], params['num_gates'], params['gate_types'])]

    '''
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
    '''

    ## EVOLUTION PROCESS ## 

    for evo_step in range(0, params['evolution_steps']):

        # This loop goes over every agent variation for this evolution step
        for agent_index, agent in enumerate(agents):

            # Create simulation pool of agents
            sim_agents = [] # stores copies of the simulation agent
            agent_locations = []
            for i in range(0, params['swarm_size']):
                sim_agents = sim_agents + [copy.deepcopy(agent)]
                loc = None # will be tuple containing location of agent on map


            # Initialize locations in a circle around center of map
            rad = len(sim_agents) // 2 # define radius as half the number of agents to allow for some margin

            # initialize agent locations equidistant around a circle
            for index in range(len(sim_agents)):
                agent_locations.append((center[0] + int(np.round(rad*np.cos((index*2*np.pi)/len(sim_agents)))), center[1] + int(np.round(rad*np.sin((index*2*np.pi)/len(sim_agents))))))          
        
            ## SIMULATION PROCESS ##

            sim_fitness = np.zeros(params['swarm_size']) # Store fitness of each agent

            # Simulate over a certain number of time steps
            for sim_step in range(0, params['time_steps']):
                new_locs = []

                # Each simulation agent must be updated, all locations are only updated after all updates (parallel updates)
                for i, sim_agent in enumerate(sim_agents):
                    
                    # Identify the locations of other agents
                    other_agents = agent_locations[:i] + agent_locations[i+1:]

                    this_agent = agent_locations[i] # extract this agent's location


                    # Simulates 1 update step and returns the new location for the simulation agent
                    loc, fit = sim_agent.brain_update(cave_map, cave_map_grad, other_agents, this_agent)
                    print(sim_agent.location)
                    new_locs = new_locs + [loc]  
                    sim_fitness[i] = fit

                # Update agent locations
                print(agent_locations)
                agent_locations = list(new_locs)

                print(new_locs)
                exit()
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