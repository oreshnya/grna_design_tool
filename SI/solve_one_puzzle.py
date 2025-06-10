import deepchem as dc
from rnalib import *
import sys
import time

# Attempt to solve a single puzzle.  The target structure in dot bracket notation
# should be supplied as a command line argument.

puzzle = sys.argv[1]
env = RNAEnvironment([puzzle], -1)
policy = RNAPolicy(env.length)
a3c = dc.rl.A3C(env, policy, model_dir='best_model')
restore(a3c)
env.reset()
steps = 0
start_time = time.time()
end_time = start_time + 24*60*60
while not env.terminated and time.time() < end_time:
    env.step(a3c.select_action(env.state))
    steps += 1
if env.terminated:
    print('SUCCESS')
    print(sequence_to_string(env.sequence))
else:
    print('FAILED')
print('time:', time.time()-start_time)
print('steps:', steps)
