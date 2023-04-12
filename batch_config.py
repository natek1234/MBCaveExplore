import yaml
import sys

# Set up variables and files
CONFIG_PATH = sys.argv[1]
NEW_CONFIG_PATH = sys.argv[2]
OUT_PATH = sys.argv[3]

# Set up simulation configuration
NUM_INPUTS = 8
NUM_OUTPUTS = 4
NUM_HIDDEN = 8
NUM_GATES = 10

POOL_SIZE = 20
SWARM_SIZE = 10
TIME_STEPS = 100
EVOLUTION_STEPS = 100

MUT_PROB = 0.01

print("Simulation variables set...")

# Modify config file
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

config["num_inputs"] = NUM_INPUTS
config["num_outputs"] = NUM_OUTPUTS
config["num_hidden"] = NUM_HIDDEN
config["num_gates"] = NUM_GATES
config["pool_size"] = POOL_SIZE
config["swarm_size"] = SWARM_SIZE
config["time_steps"] = TIME_STEPS
config["evolution_steps"] = EVOLUTION_STEPS
config["mut_prob"] = MUT_PROB
config["out_path"] = OUT_PATH

with open(NEW_CONFIG_PATH, "w") as f:
    yaml.dump(config, f)