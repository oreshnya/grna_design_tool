import deepchem as dc
import tensorflow as tf
from rnalib import *
import sys
import time

# Attempt to solve a single puzzle. The target structure in dot bracket notation
# should be supplied as a command line argument.

puzzle = sys.argv[1]
env = RNAEnvironment([puzzle], -1)

policy = RNAPolicy(env.length)

a3c = dc.rl.A3C(env, policy, model_dir='best_model')
restore(a3c)
env.reset()
steps = 0
start_time = time.time()
end_time = start_time + 10*60  # максимум 1 минута

print(f"Starting solve loop for: {puzzle}")

# while not env.terminated and time.time() < end_time:
while not env.terminated:
    # if steps % 500 == 0:
    #     print(f"Step {steps}: {sequence_to_string(env.sequence)}")
    #     # print(type(env.sequence), env.sequence)
    if steps % 10 == 0:
        seq = env.sequence
        struct = sequence_to_bracket(seq)
        print(f"=================== Step {steps} ===================")
        print(f"{sequence_to_string(seq)} ← current sequence")
        print(f"{struct} ← current structure")
        print(f"{puzzle} ← target structure")
        print('')
    env.step(a3c.select_action(env.state))
    steps += 1

if env.terminated:
    print('SUCCESS')
    print(sequence_to_string(env.sequence))
else:
    print('FAILED')

print('time:', time.time() - start_time)
print('steps:', steps)