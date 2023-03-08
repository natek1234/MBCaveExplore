import yaml
import numpy as np
from MarkovBrain import MarkovBrain

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
    print(cave_map)


    ## EVOLUTION PROCESS ## 
    for evo_step in range(0, params['evolution_steps']):

        ## SIMULATION PROCESS ##
        for sim_step in range(0, params['time_steps']):

            # Complete the simulation process (brain updates, movement, etc.)

            pass

        ## SELECTION ## 

        # Select best performing brains as parents

        ## GENERATE ##

        # Create next generation of brains

        # Crossover

        # Mutation

        ## RESET ##

        # Reset map environment 