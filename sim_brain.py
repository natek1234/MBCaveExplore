import pickle

BRAIN_PATH = './stats/parents_LOD.pkl'
TEST = './stats/fitness.pkl'


# Open the file in read binary mode
with open(TEST, 'rb') as f:
    # Load the contents of the file using pickle.load()
    data = pickle.load(f)

print(data)

# Open the file in read binary mode
with open(BRAIN_PATH, 'rb') as f:
    # Load the contents of the file using pickle.load()
    brains = pickle.load(f)

print(brains)

for i in range(0, len(brains)):
    print('Brain fitness: ', brains[i][0].fitness)
    print('Brain location: ', brains[i][0].location)
    print('Extracted fitness: ', data[i][0])

    print('Brain fitness: ', brains[i][1].fitness)
    print('Extracted fitness: ', data[i][1])