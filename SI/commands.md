## Quick start

conda activate rna-rl

cd /mnt/c/projects/a3c/SI

python solve_one_puzzle_patch.py "(((((((.(.(.(.(((((((....)))))))))))))))))"

## Output

Starting solve loop for: ((((((((((((((...)))))))))))))).
=================== Step 0 ===================
CGCGUUCUGAUAUAACCUAGCUACCAGGUUCU ← current sequence
.............(((((.......))))).. ← current structure
((((((((((((((...)))))))))))))). ← target structure

SUCCESS
UUUUAAAAAAAAAAAUCUUUUUUUUUUAAAAU
time: 0.6400511264801025
steps: 42

Starting solve loop for: (((((.....)))))
=================== Step 0 ===================
AAGAGGUGAGUUAGC ← current sequence
............... ← current structure
(((((.....))))) ← target structure

SUCCESS
ACUUUUUUUUAAAGU
time: 0.631406307220459
steps: 39

## Install conda

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

bash ~/miniconda.sh

export PATH="$HOME/miniconda3/bin:$PATH"

source ~/.bashrc    # или source ~/.zshrc

conda --version

conda init


## Create venv

conda create -n rna-rl python=3.6.15 --no-channel-priority -c conda-forge -c anaconda

conda activate rna-rl

export CONDA_SUBDIR=linux-64

mkdir -p $CONDA_PREFIX/etc/conda/activate.d

echo "export CONDA_SUBDIR=linux-64" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

cat $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh


conda config --env --add channels deepchem
conda config --env --add channels omnia
conda config --env --add channels rdkit
conda config --env --add channels bioconda
conda config --env --add channels conda-forge


conda install -c rdkit rdkit=2017.03.1
conda install -c bioconda viennarna=2.3.5
conda install scipy=0.18.1
conda install -c deepchem deepchem=1.3.1
conda install -c conda-forge/label/cf201901 tensorflow=1.3.0


## Check

### TensorFlow check
python -c "
import tensorflow as tf
print('✅ TensorFlow:', tf.__version__)
"

✅ TensorFlow: 1.3.0

### DeepChem check
pip show deepchem

Name: deepchem
Version: 1.3.0

### RDKit check
python -c "
import rdkit
print('✅ RDKit:', rdkit.__version__)
"

✅ RDKit: 2017.03.1


### ViennaRNA check
python -c "
import RNA
print('✅ ViennaRNA:', RNA.pf_fold('(....)'))
"

✅ ViennaRNA: ['......', 1.3685070606957362e-16]

### Run
cd /mnt/c/projects/a3c/SI

python solve_one_puzzle.py "(((((((.(.(.(.(((((((....)))))))))))))))))"