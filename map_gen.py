import numpy as np

# Approximately MARS according to NASA
DAY_AVG = 22 # Average temperature during the day (C)
NIGHT_AVG = -100 # Average temperature during the night (C)

# Temperature difference from average exactly at cave location
MIN_GRAD = 5 
MAX_GRAD = 10

# Generates a map given cave locations and writes to disk
# Experiment: 1km^2, 2m resolution means 500 by 500 map (can be changed)
def map_gen(desired_caves, output_loc, map_size = (500,500), time_of_day = 'Day'):

    # Initialize map depending on time of day
    if time_of_day == 'Day':
        map_out = np.ones(map_size)*DAY_AVG
        max_grad = -MAX_GRAD # during the day, the cave will be cooler than the average
        min_grad = -MIN_GRAD
    else:
        map_out = np.ones(map_size)*NIGHT_AVG
        max_grad = MAX_GRAD # during the night, the cave will be warmer than the average
        min_grad = MIN_GRAD
    
    # Place caves in map, implement Gaussian distribution around cave location (is this good?)
    cave_locs_x = np.random.randint(10, map_size[0], desired_caves)
    cave_locs_y = np.random.randint(10, map_size[1], desired_caves)

    cave_locs = (cave_locs_x, cave_locs_y) # randomly generated cave locations
    
    grad = np.random.uniform(min_grad, max_grad, desired_caves) # randomly assign gradient at cave

    map_out[cave_locs] = map_out[cave_locs] + grad # modify map to include caves

    ## ADD GAUSSIAN DROPOFF AROUND THE CAVE LOCATIONS IN TERMS OF INTENSITY ##

    # Apply Gaussian dropoff around the cave locations
    dropoff_sigma = 15
    for i in range(desired_caves):
        mask = np.zeros(map_size, dtype=bool)
        mask[cave_locs_x[i],cave_locs_y[i]] = True
        dropoff = grad[i] * np.exp(-((np.arange(map_size[0]) - cave_locs_x[i])**2 + (np.arange(map_size[1])[:,None] - cave_locs_y[i])**2) / (2 * dropoff_sigma**2))
        dropoff[mask] = 0
        map_out += dropoff

    # Write to disk
    np.savetxt(output_loc, map_out)

    return True

if __name__ == '__main__':

    map_gen(6, './maps_6_caves/map_3_r15.txt')