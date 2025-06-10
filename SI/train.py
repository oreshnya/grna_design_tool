import deepchem as dc
import deepchem.models.tensorgraph.optimizers as optimizers
import numpy as np
from multiprocessing.dummy import Pool
from rnalib import *

# Load in the training puzzles.

num_validation = 500
easy_puzzles = []
hard_puzzles = []
with open('puzzles32.txt') as infile:
    for line in infile:
        puzzle, count = line.split()
        if int(count) >= 3:
            easy_puzzles.append(puzzle)
        else:
            hard_puzzles.append(puzzle)
validation_puzzles = hard_puzzles[-num_validation:]

def compute_validation_score(a3c, puzzles):
    """Compute the total number of steps needed to solve the validation puzzles."""

    def eval_puzzle(puzzle):
        count = 0
        env = RNAEnvironment([puzzle], 10000)
        env.reset()
        while not env.terminated:
            env.step(a3c.select_action(env.state))
            count += 1
        return count

    pool = Pool()
    score = sum(pool.map(eval_puzzle, puzzles)) / len(puzzles)
    print('validation score:', score)
    return score

# For the first stage, train for 500,000 steps on the easy puzzles.

print('Begin training stage 1')
env = RNAEnvironment(easy_puzzles, 10000)
policy = RNAPolicy(env.length)
decay = 0.8
learning_rate = optimizers.ExponentialDecay(1e-5, decay, 100000)
a3c = dc.rl.A3C(env, policy, max_rollout_length=50, model_dir='model', entropy_weight=0.1,
                optimizer=optimizers.Adam(learning_rate=learning_rate))
a3c.fit(500000)

# For the second stage, train on both easy and hard puzzles.  Every 100,000
# steps compute the validation score.  If it's the best we've seen so far, save
# the model to disk.

print('Begin training stage 2')
env = RNAEnvironment(easy_puzzles+hard_puzzles[:-num_validation], 10000)
policy = RNAPolicy(env.length)
best_score = 10000
for i in range(10):
    learning_rate = 1e-5*(decay**(i+5))
    a3c = dc.rl.A3C(env, policy, max_rollout_length=50, model_dir='model', entropy_weight=0.1,
                    optimizer=optimizers.Adam(learning_rate=learning_rate))
    a3c.fit(100000, restore=True)
    score = compute_validation_score(a3c, validation_puzzles)
    if score < best_score:
        best_score = score
        with a3c._graph._get_tf("Graph").as_default():
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global')
            saver = tf.train.Saver(variables)
            saver.save(a3c._session, 'best_model/model')
print('best validation score:', best_score)
