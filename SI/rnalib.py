import deepchem as dc
import deepchem.models.tensorgraph.layers as layers
import numpy as np
import tensorflow as tf
# import RNA
import random
import subprocess

# This file defines the classes for the environment and the policy network, as well
# as various useful functions.

width = 80 # Number of channels for internal layers
pairs = {0:(3,), 1:(2,), 2:(1,3), 3:(0,2)} # Which other types of bases each type can pair with

def bracket_to_bonds(structure):
    """Given a structure in dot bracket notation, compute the list of bonds."""
    bonds = [None]*len(structure)
    opening = []
    for i,c in enumerate(structure):
        if c == '(':
            opening.append(i)
        elif c == ')':
            j = opening.pop()
            bonds[i] = j
            bonds[j] = i
    return bonds

def sequence_to_string(sequence):
    """Convert a one hot encoded sequence to a string."""
    bases = ['A', 'C', 'G', 'U']
    return ''.join(bases[i] for i in sequence)

# def sequence_to_bracket(sequence):
#     """Compute the native structure (in dot bracket notation) of a one hot encoded sequence."""
#     structure, energy = RNA.fold(sequence_to_string(sequence))
#     return structure

def sequence_to_bracket(sequence):
    seq_str = sequence_to_string(sequence)
    result = subprocess.run(
        ['RNAfold'], input=seq_str.encode(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    lines = result.stdout.decode().strip().split('\n')
    if len(lines) >= 2:
        struct_line = lines[1].strip().split()[0]
        return struct_line
    else:
        raise RuntimeError(f"RNAfold failed on input: {seq_str}")


class RNAEnvironment(dc.rl.Environment):
    """This class implements the environment our agent will interact with."""

    def __init__(self, puzzles, max_steps):
        """Create a new RNAEnvironment.

        Parameters
        ----------
        puzzles: list
            the list of training puzzles.  A random one is selected for each episode.
        max_steps: int
            the maximum number of steps for any episode.  If the puzzle has not been
            solved after this many steps, it will give up.  Pass -1 to not set a limit.
        """
        self.puzzles = puzzles
        self.max_steps = max_steps
        self.length = len(puzzles[0])
        super(RNAEnvironment, self).__init__([(self.length, 4), (7*self.length, 1)], self.length*4, [np.float32, np.int32])

    def step(self, action):
        """Perform one action on the environment."""
        index = action//4
        base = action%4
        self.count += 1
        if self.count == self.max_steps:
            self._terminated = True # Give up.
            reward = 0
        if self.sequence[index] == base:
            # This action doesn't change anything.
            reward = 0
        elif not self._terminated:
            self.sequence[index] = base
            pair_index = self.target_bonds[index]
            if pair_index is not None:
                if self.sequence[pair_index] not in pairs[base]:
                    self.sequence[pair_index] = pairs[base][0]
            self._update_state()
            reward = 1 if self._terminated else 0
        return reward

    def reset(self):
        """Reset the environment and begin a new episode."""
        self.goal = random.choice(self.puzzles)
        self.target_bonds = bracket_to_bonds(self.goal)
        self.count = 0
        while True:
            self.sequence = [random.randint(0, 3) for i in range(self.length)]
            self._update_state()
            if not self._terminated:
                break

    def _update_state(self):
        """Update the state vectors encoding the current sequence and list of bonds."""
        bracket = sequence_to_bracket(self.sequence)
        bonds = bracket_to_bonds(bracket)
        self._terminated = (bracket == self.goal)

        # Compute the state.

        state1 = np.zeros((self.length, 4))
        state1[np.arange(self.length), self.sequence] = 1
        state2 = []
        for i in range(self.length):
            state2.append(i-2)
            state2.append(i-1)
            state2.append(i)
            state2.append(i+1)
            state2.append(i+2)
            if bonds[i] is None:
                state2.append(self.length)
            else:
                state2.append(bonds[i])
            if self.target_bonds[i] is None:
                state2.append(self.length)
            else:
                state2.append(self.target_bonds[i])
        for i, s in enumerate(state2):
            if s < 0 or s > self.length:
                state2[i] = self.length
        self._state = [state1, np.expand_dims(np.array(state2, np.int32), 1)]


class GatherBonds(layers.Layer):
    """This layer assembles the inputs for a conv7 operation."""

    def __init__(self, length, **kwargs):
        super(GatherBonds, self).__init__(**kwargs)
        self.length = length
        try:
            input_width = self.in_layers[0].shape[-1]
            self._shape = (None, length*7*input_width, 1)
        except:
            pass

    def create_tensor(self, **kwargs):
        inputs = self.in_layers[0].out_tensor
        indices = self.in_layers[1].out_tensor
        input_width = int(inputs.get_shape()[-1])
        padded = tf.pad(inputs, [[0,0], [0,1], [0,0]])
        padded_shape = tf.shape(padded)
        batch_offset = tf.reshape(tf.range(0, padded_shape[0]) * padded_shape[1], [-1, 1, 1])
        flattened_indices = tf.reshape(indices+batch_offset, tf.concat([[-1], tf.shape(indices)[2:]], 0))
        flattened_inputs = tf.reshape(padded, tf.concat([[-1], padded_shape[2:]], 0))
        gathered = tf.gather(flattened_inputs, flattened_indices)
        self.out_tensor = tf.reshape(gathered, [-1, self.length*7*input_width, 1])
        return self.out_tensor


def create_conv7(parent, length, indices, n_outputs):
    """Create a conv7 operation."""
    gather = GatherBonds(length, in_layers=[parent, indices])
    w = gather.shape[1]//length
    return layers.Conv1D(
        width=w, stride=w,
        out_channels=n_outputs,
        activation_fn=tf.nn.relu,  # ← добавили
        in_layers=gather
    )


def create_residual(parent, length, indices, n_outputs):
    """Create a residual block."""
    conv = create_conv7(parent, length, indices, n_outputs)
    flattened = layers.Flatten(conv)
    return parent + layers.Conv1D(
        width=n_outputs, stride=n_outputs,
        out_channels=n_outputs,
        activation_fn=tf.nn.relu,  # ← исправили
        weights_initializer=tf.zeros_initializer,
        biases_initializer=tf.zeros_initializer,
        in_layers=flattened
    )


class RNAPolicy(dc.rl.Policy):
    """This class implements the policy network."""

    def __init__(self, length):
        """Create an RNAPolicy.

        Parameters
        ----------
        length: int
            the length of the training sequences
        """
        self.length = length

    def create_layers(self, state, **kwargs):
        indices = state[1]
        conv1 = create_conv7(state[0], self.length, indices, width)
        conv2 = create_residual(conv1, self.length, indices, width)
        conv3 = create_residual(conv2, self.length, indices, width)
        conv4 = create_residual(conv3, self.length, indices, width)
        flattened_conv = layers.Flatten(in_layers=conv4)
        action = layers.Flatten(layers.Conv1D(
                                                width=width, stride=width, out_channels=4,
                                                biases_initializer=tf.zeros_initializer,
                                                activation_fn=tf.nn.relu,  # ← ✅ теперь всё ок
                                                in_layers=flattened_conv
                                            ))
        masked = layers.Add(in_layers=[action, layers.Flatten(in_layers=state[0])], weights=[1, -1e6])
        action_prob = layers.SoftMax(in_layers=masked)
        value = layers.Dense(out_channels=1, in_layers=flattened_conv)
        return {'action_prob': action_prob, 'value': value}

def restore(a3c, dir=None):
    """Restore the model from the most recent checkpoint file.

    Normally we would just call a3c.restore(), but that will fail if the checkpoint
    was created for sequences of a different length.  This function loads all
    variables *except* the ones for the single dense layer.
    """
    if dir is None:
        dir = a3c._graph.model_dir
    last_checkpoint = tf.train.latest_checkpoint(dir)
    with a3c._graph._get_tf("Graph").as_default():
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global')
        variables = [v for v in variables if 'Dense' not in v.name]
        saver = tf.train.Saver(variables)
        saver.restore(a3c._session, last_checkpoint)
