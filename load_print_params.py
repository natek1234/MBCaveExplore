import pickle

TEST = "./stats/Archive/all_maps_8_inputs_add_distance/5_gates_100_iter/sim_params.pkl"


# Open the file in read binary mode
with open(TEST, 'rb') as f:
    # Load the contents of the file using pickle.load()
    data = pickle.load(f)

for entry in data:
    print(entry, ': ', data[entry])