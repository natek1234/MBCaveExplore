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

    ## TO DO

    # Write to disk
    np.savetxt(output_loc, map_out)

    return True

if __name__ == '__main__':

    map_gen(6, './map_test.txt')