import yaml
import numpy as np
from MarkovBrain import MarkovBrain
from Gates import Gates
import matplotlib.pyplot as plt
import copy
import pickle
import imageio

VISUALIZE_GRAD_MAP = False
VISUALIZE = False # Visualization option to be implemented
PLOT_OUTPUT = True # Plot and save the output fitness and statistics
OUTPUT_PATH = './stats/'
SAVE_STATS = True # Saves line of descent, avg fitness, and fitness to file

MUT_PROB = 0.01 # Mutation probability

# Perform mutation on a brain
def mutate(brain):
    for gate in range(0,brain.num_gates):
        # Mutate the gate with probability MUT_PROB - Re-create the gate randomly
        if np.random.rand() < MUT_PROB:
            # Randomly initialize gates to probabilistic, deterministic, or special
            gate_type = np.random.choice(['probabilistic','deterministic']) # 0 is probabilistic, 1 is deterministic, 2 is special (to be added later)
            brain.gates[gate] = Gates(gate_type, np.random.choice(brain.gate_types), brain.gate_ids[gate], brain.num_inputs+brain.num_hidden, brain.input_ids, brain.output_ids, brain.hidden_ids)
            #print('MUTATION')
            #exit()
   
    return True

# Perform crossover on a brain
# When a gate is swapped, all its inputs, outputs, and other properties are also swapped
def crossover(brain_1, brain_2):

    new_brain = copy.deepcopy(brain_1) # start the new brain as a duplicate of one of the input brains

    # Randomly keep gates or select gates from the other brain
    for gate_ind in range(0, new_brain.num_gates):
       
        # Only change half the gates 
        if np.random.rand() < 0.5: # 50% prob of changing the gate
            new_brain.gates[gate_ind] = copy.deepcopy(brain_2.gates[gate_ind]) # make the gate a copy of the gate from the second brain

    return new_brain

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

    if VISUALIZE_GRAD_MAP:
        fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True)

        # plot the first subplot
        im1 = ax1.imshow(cave_map_grad[0], cmap='coolwarm', vmin=np.min(cave_map_grad)*0.5, vmax=np.max(cave_map_grad)*0.5)
        ax1.set_title('Cave map gradient (y direction)')
        ax1.set_xlabel('x-axis (2m per pixel)')
        ax1.set_ylabel('y-axis (2m per pixel)')

        # plot the second subplot
        im2 = ax2.imshow(cave_map_grad[1], cmap='coolwarm', vmin=np.min(cave_map_grad)*0.5, vmax=np.max(cave_map_grad)*0.5)
        ax2.set_title('Cave map gradient (x direction)')
        ax2.set_xlabel('x-axis (2m per pixel)')
        ax2.set_ylabel('y-axis (2m per pixel)')

        # create an axes object for the colorbar
        cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])

        # add the shared colorbar
        cbar = plt.colorbar(im2, cax=cbar_ax)

        plt.show()

    ## GENERATE FIRST GENERATION ##

    agents = []
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

    ## DATA COLLECTION ##
    all_parents =  [] # will store all the selected parents objects for saving
    all_fitness = [] # will store all the fitness of all the selected parents
    all_fitness_avg = [] # stores average of selected parents' fitness
    best_brain = None # stores the best performing brain

    ## EVOLUTION PROCESS ## 

    for evo_step in range(0, params['evolution_steps']):

        print('\n\nEVOLUTION STEP: {}/{}'.format(evo_step, params['evolution_steps']))
        print('---------------------------------------------------------------')

        fitness = np.zeros(params['pool_size']) # initialize fitness pool for all agents

        # This loop goes over every agent variation for this evolution step
        for agent_index, agent in enumerate(agents):

            print(f'\n\nMB CANDIDATE: {agent_index}')
            print('---------------------------------------------------------------')

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

            ## VISUALIZATION ##
            if VISUALIZE:

                fig, ax = plt.subplots(1, 1)

                # create an initial plot with the cave_map
                im = ax.imshow(cave_map, cmap='coolwarm', vmin=np.min(cave_map), vmax=np.max(cave_map)*1.25)
                ax.set_title('Agent Locations')
                plt.colorbar(im)

                scatter = ax.scatter([loc[1] for loc in agent_locations], [loc[0] for loc in agent_locations], c='r', s=2)

                # show the plot
                plt.show(block=False)

            # Simulate over a certain number of time steps
            for sim_step in range(0, params['time_steps']):
                print('\rSimulation step: {}/{}'.format(sim_step, params['time_steps']), end='', flush=True)
                new_locs = []

                # Each simulation agent must be updated, all locations are only updated after all updates (parallel updates)
                for i, sim_agent in enumerate(sim_agents):
                    
                    # Identify the locations of other agents
                    other_agents = agent_locations[:i] + agent_locations[i+1:]

                    this_agent = agent_locations[i] # extract this agent's location


                    # Simulates 1 update step and returns the new location for the simulation agent
                    loc, fit = sim_agent.brain_update(cave_map, cave_map_grad, other_agents, this_agent)
                    new_locs = new_locs + [loc]  
                    sim_fitness[i] = fit

                # Update agent locations
                #print(agent_locations)
                agent_locations = list(new_locs)

                ## UPDATE VISUALIZATION ##
                if VISUALIZE:
                    # Update the dots which represents the agents using the new agent locations on the previously made plot

                    # draw the plot
                    scatter.remove()
                    scatter = ax.scatter([loc[1] for loc in agent_locations], [loc[0] for loc in agent_locations], c='r', s=2)
                    plt.draw()
                    plt.pause(0.001) # essentially controls simulation speed

            # Close figure
            if VISUALIZE:
                plt.close()  

            # Update the agent-level fitness
            fitness[agent_index] = np.mean(sim_fitness)

        # After all agents have been simulated, compare fitness and evolve

        ## SELECTION ## 

        # Select two highest performing brains as parents

        idx = np.argpartition(fitness, -2)[-2:] # get indices of top two elements

        print(f'\n\nEnd of generation \nParent 1 fitness: {fitness[idx[0]]} \nParent 2 fitness: {fitness[idx[1]]}')

        # Collect Data
        agents[idx[0]].fitness = fitness[idx[0]]
        agents[idx[1]].fitness = fitness[idx[1]]
        all_parents = all_parents + [[agents[idx[0]], agents[idx[1]]]]
        all_fitness = all_fitness + [[fitness[idx[0]], fitness[idx[1]]]]
        all_fitness_avg = all_fitness_avg + [np.mean([fitness[idx[0]], fitness[idx[1]]])]

        # Update best brain
        if best_brain == None:
            best_brain = agents[idx[1]]
        elif fitness[idx[1]] > best_brain.fitness:
            best_brain = agents[idx[1]]

        '''
        # TEST - List properties of parent brains
        print('Parent 1')
        print('------------------------------------------')
        for i in range(0, 5):
            print(agents[idx[0]].gates[i].gate_type)
            print(agents[idx[0]].gates[i].gate_name)
            print(agents[idx[0]].gates[i].id)
            print(agents[idx[0]].gates[i].num_inputs)
            #print(agents[idx[0]].gates[i].truth_table)
            #print(agents[idx[0]].gates[i].output_prob)
            print(agents[idx[0]].gates[i].input_connections)
            print(agents[idx[0]].gates[i].output_connections)

        print('Parent 2')
        print('------------------------------------------')
        for i in range(0, 5):
            print(agents[idx[1]].gates[i].gate_type)
            print(agents[idx[1]].gates[i].gate_name)
            print(agents[idx[1]].gates[i].id)
            print(agents[idx[1]].gates[i].num_inputs)
            #print(agents[idx[1]].gates[i].truth_table)
            #print(agents[idx[1]].gates[i].output_prob)
            print(agents[idx[1]].gates[i].input_connections)
            print(agents[idx[1]].gates[i].output_connections)
        '''

        ## GENERATE ##

        # Create next generation of brains
        new_agents = []
        for agent in range(0, params['pool_size']):
            #print(agent)

            # Crossover: combine the two best performing agents randomly
            new_agent = crossover(agents[idx[0]], agents[idx[1]])

            # Mutation: mutate the new agent
            mutate(new_agent)

            '''
            print('New agent gates: ')
            print('------------------------------------------')
            for i in range(0, 5):
                print(new_agent.gates[i].gate_type)
                print(new_agent.gates[i].gate_name)
                print(new_agent.gates[i].id)
                print(new_agent.gates[i].num_inputs)
                #print(new_agent.gates[i].truth_table)
                #print(new_agent.gates[i].output_prob)
                print(new_agent.gates[i].input_connections)
                print(new_agent.gates[i].output_connections)
            
            '''

            # Add the new agent to the pool
            new_agents = new_agents + [new_agent]

        agents = list(new_agents)

    
    ## POST PROCESSING ##

    # Plot and save the output statistics
    if PLOT_OUTPUT:
        
        iterations = np.arange(params['evolution_steps'])

        # Create the plot
        plt.plot(iterations, all_fitness_avg)

        # Set the axis labels and title
        plt.xlabel('Evolution Step')
        plt.ylabel('Fitness')
        plt.title('Agent Fitness vs. Evolution Step')

        # Save the output
        plt.savefig(OUTPUT_PATH + 'avg_fitness_test.png')

        # Display the plot
        plt.show()

    if SAVE_STATS:
        with open(OUTPUT_PATH + 'parents_LOD.pkl', 'wb') as f:
            # Use the pickle module to dump the list of objects into the file
            pickle.dump(all_parents, f)

        with open(OUTPUT_PATH + 'avg_fitness.pkl', 'wb') as f:
            # Use the pickle module to dump the list of objects into the file
            pickle.dump(all_fitness_avg, f)

        with open(OUTPUT_PATH + 'fitness.pkl', 'wb') as f:
            # Use the pickle module to dump the list of objects into the file
            pickle.dump(all_fitness, f)

        '''

        # Create simulation pool of agents
        sim_agents = [] # stores copies of the simulation agent
        agent_locations = []
        for i in range(0, params['swarm_size']):
            sim_agents = sim_agents + [copy.deepcopy(best_brain)]
            loc = None # will be tuple containing location of agent on map

        
        # Initialize locations in a circle around center of map
        rad = len(sim_agents) // 2 # define radius as half the number of agents to allow for some margin

        # initialize agent locations equidistant around a circle
        for index in range(len(sim_agents)):
            agent_locations.append((center[0] + int(np.round(rad*np.cos((index*2*np.pi)/len(sim_agents)))), center[1] + int(np.round(rad*np.sin((index*2*np.pi)/len(sim_agents))))))          
    
        ## SIMULATION PROCESS ##

        sim_fitness = np.zeros(params['swarm_size']) # Store fitness of each agent

        ## VISUALIZATION ##
        if VISUALIZE:

            fig, ax = plt.subplots(1, 1)

            # create an initial plot with the cave_map
            im = ax.imshow(cave_map, cmap='coolwarm', vmin=np.min(cave_map), vmax=np.max(cave_map)*1.25)
            ax.set_title('Agent Locations')
            plt.colorbar(im)

            scatter = ax.scatter([loc[1] for loc in agent_locations], [loc[0] for loc in agent_locations], c='r', s=2)

            # show the plot
            plt.show(block=False)

        # Simulate over a certain number of time steps
        for sim_step in range(0, params['time_steps']):
            print('\rSimulation step: {}/{}'.format(sim_step, params['time_steps']), end='', flush=True)
            new_locs = []

            # Each simulation agent must be updated, all locations are only updated after all updates (parallel updates)
            for i, sim_agent in enumerate(sim_agents):
                
                # Identify the locations of other agents
                other_agents = agent_locations[:i] + agent_locations[i+1:]

                this_agent = agent_locations[i] # extract this agent's location


                # Simulates 1 update step and returns the new location for the simulation agent
                loc, fit = sim_agent.brain_update(cave_map, cave_map_grad, other_agents, this_agent)
                new_locs = new_locs + [loc]  
                sim_fitness[i] = fit

            # Update agent locations
            #print(agent_locations)
            agent_locations = list(new_locs)

            ## UPDATE VISUALIZATION ##
            if VISUALIZE:
                # Update the dots which represents the agents using the new agent locations on the previously made plot

                # draw the plot
                scatter.remove()
                scatter = ax.scatter([loc[1] for loc in agent_locations], [loc[0] for loc in agent_locations], c='r', s=2)
                plt.draw()
                plt.pause(0.001) # essentially controls simulation speed
        '''