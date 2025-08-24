These files contain the source code and data needed to reproduce the results from the paper.
Before running them, you first must install certain prerequisites:

- DeepChem 1.3.1
- Tensorflow 1.3
- ViennaRNA 2.3.5 (including the Python API)

It is possible they will also work with newer versions of those libraries, but those are
the only versions they have been tested with.

To have the RL agent solve a puzzle, run the `solve_one_puzzle.py` script, passing the
puzzle in dot bracket notation as a command line argument.  For example:

    python solve_one_puzzle_patch.py "(((((((.(.(.(.(((((((....)))))))))))))))))"

The `best_model` directory contains the saved model parameters in the form of Tensorflow
checkpoint files.  These are the parameters that were used when running the tests in the
paper.  Alternatively, you can run the `train.py` script to retrain the model yourself.
This should take a few hours on a typical laptop.  Note that it will overwrite the
`best_model` directory.
