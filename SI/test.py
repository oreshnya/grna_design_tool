from rnalib import *

# seq_str = "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAAA"
seq_str = "AUUUAU"
base_to_index = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
seq_encoded = [base_to_index[b] for b in seq_str]

struct = sequence_to_bracket(seq_encoded)
print(struct)