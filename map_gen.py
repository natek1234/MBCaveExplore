import numpy as np

DAY_AVG = 20 # Average temperature during the day (TBD)
NIGHT_AVG = -20 # Average temperature during the night (TBD)
MAX_GRAD = 10 # Temperature difference from average exactly at cave location

# Generates a map given cave locations and writes to disk
def map_gen(desired_caves, output_loc, map_size = (50,50), time_of_day = 'Day'):

    # Initialize map depending on time of day
    if time_of_day == 'Day':
        map_out = np.ones(map_size)*DAY_AVG
        max_grad = -MAX_GRAD # during the day, the cave will be cooler than the average
    else:
        map_out = np.ones(map_size)*NIGHT_AVG
        max_grad = MAX_GRAD # during the night, the cave will be warmer than the average
    
    # Place caves in map, implement Gaussian distribution around cave location (is this good?)

    ## TO DO

    # Write to disk
    np.savetxt(output_loc, map_out)

    return True

if __name__ == '__main__':

    map_gen(None, './map_test.txt')