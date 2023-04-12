import pickle
import numpy as np
import yaml
import matplotlib.pyplot as plt
import copy
import imageio.v3 as iio
import sys

# Read arguments and set up paths
config_path = sys.argv[1]

stream = open(config_path, 'r')
params = yaml.safe_load(stream) # all config parameters saved in params

out_path = params['out_path']

DIR = out_path
BRAIN_PATH = DIR + 'parents_LOD.pkl'
TEST = DIR + 'fitness.pkl'
GIF_PATH = out_path + 'gif_imgs/'
SAVE_SIM_GIF = True
VISUALIZE_GRAD_MAP = False


# Create map
cave_map = np.loadtxt(params['maps'][1])
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


# Open the file in read binary mode
with open(TEST, 'rb') as f:
    # Load the contents of the file using pickle.load()
    data = pickle.load(f)

# Open the file in read binary mode
with open(BRAIN_PATH, 'rb') as f:
    # Load the contents of the file using pickle.load()
    brains = pickle.load(f)

best_brain = None
best_brain_fitness = -np.inf

for i in range(0, len(brains)):

    if brains[i][1].fitness > best_brain_fitness:
        best_brain_fitness = brains[i][1].fitness
        best_brain = brains[i][1]

# SIMULATE

gif_figs = []

# Create simulation pool of agents
sim_agents = [] # stores copies of the simulation agent
agent_locations = []
for i in range(0, params['swarm_size']):
    sim_agents = sim_agents + [copy.deepcopy(best_brain)]
    sim_agents[-1].fitness = 0.0 # ensure fitness of each simulation starts at 0 (avoid copying errors)
    loc = None # will be tuple containing location of agent on map


# Initialize locations in a circle around center of map
rad = len(sim_agents) # define radius as half the number of agents to allow for some margin

# initialize agent locations equidistant around a circle
for index in range(len(sim_agents)):
    agent_locations.append((center[0] + int(np.round(rad*np.cos((index*2*np.pi)/len(sim_agents)))), center[1] + int(np.round(rad*np.sin((index*2*np.pi)/len(sim_agents))))))          

## SIMULATION PROCESS ##

sim_fitness = np.zeros(params['swarm_size']) # Store fitness of each agent

## VISUALIZATION ##

fig, ax = plt.subplots(1, 1)

# create an initial plot with the cave_map
im = ax.imshow(cave_map, cmap='coolwarm', vmin=np.min(cave_map), vmax=np.max(cave_map)*1.25)
ax.set_title('Agent Locations')
plt.colorbar(im)

scatter = ax.scatter([loc[1] for loc in agent_locations], [loc[0] for loc in agent_locations], c='r', s=2)

# show the plot
plt.show(block=False)

if SAVE_SIM_GIF:

    fig2, ax2 = plt.subplots(1, 1)

    # create an initial plot with the cave_map
    im2 = ax2.imshow(cave_map, cmap='coolwarm', vmin=np.min(cave_map), vmax=np.max(cave_map)*1.25)
    ax2.set_title('Agent Locations')
    plt.colorbar(im2)

    scatter2 = ax2.scatter([loc[1] for loc in agent_locations], [loc[0] for loc in agent_locations], c='r', s=2)
    #gif_figs = gif_figs + [fig2]
    plt.close(fig2)

# Simulate over a certain number of time steps
for sim_step in range(0, params['time_steps']):
    if sim_step % 100 == 0:
        print('\rSimulation step: {}/{}'.format(sim_step, params['time_steps']), end='', flush=True)
    new_locs = []

    # Each simulation agent must be updated, all locations are only updated after all updates (parallel updates)
    for i, sim_agent in enumerate(sim_agents):
        
        # Identify the locations of other agents
        other_agents = agent_locations[:i] + agent_locations[i+1:]

        this_agent = agent_locations[i] # extract this agent's location


        # Simulates 1 update step and returns the new location for the simulation agent
        if sim_step == params['time_steps']-1:
            loc, fit = sim_agent.brain_update(cave_map, cave_map_grad, other_agents, this_agent, last_iter=True) # last iteration takes into account distance to nearest cave
        else:
            loc, fit = sim_agent.brain_update(cave_map, cave_map_grad, other_agents, this_agent)
        new_locs = new_locs + [loc]  
        sim_fitness[i] = fit

    # Update agent locations
    agent_locations = list(new_locs)

    ## UPDATE VISUALIZATION ##
    # Update the dots which represents the agents using the new agent locations on the previously made plot

    # draw the plot
    scatter.remove()
    scatter = ax.scatter([loc[1] for loc in agent_locations], [loc[0] for loc in agent_locations], c='r', s=2)
    plt.draw()
    plt.pause(0.001) # essentially controls simulation speed

    if SAVE_SIM_GIF and sim_step % 10 == 0: # add every 10 steps to the gif

        fig2, ax2 = plt.subplots(1, 1)

        # create an initial plot with the cave_map
        im2 = ax2.imshow(cave_map, cmap='coolwarm', vmin=np.min(cave_map), vmax=np.max(cave_map)*1.25)
        ax2.set_title(f'Agent Locations - sim step {sim_step}')
        plt.colorbar(im2)

        scatter2 = ax2.scatter([loc[1] for loc in agent_locations], [loc[0] for loc in agent_locations], c='r', s=2)
        gif_figs = gif_figs + [GIF_PATH + f"sim step-{sim_step}.jpg"]
        fig2.savefig(GIF_PATH + f"sim step-{sim_step}.jpg")
        plt.close(fig2)

# Close figure
plt.close()  

if SAVE_SIM_GIF:
    frames = np.stack([iio.imread(fig) for fig in gif_figs], axis=0)

    iio.imwrite(DIR + 'best_brain.gif', frames, duration=0.5)