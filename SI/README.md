These files contain the source code and data needed to reproduce the results from the paper.
Before running them, you first must install certain prerequisites:

- DeepChem 1.3.1
- Tensorflow 1.3
- ViennaRNA 2.3.5 (including the Python API)



It is possible they will also work with newer versions of those libraries, but those are
the only versions they have been tested with.

To have the RL agent solve a puzzle, run the `solve_one_puzzle.py` script, passing the
puzzle in dot bracket notation as a command line argument.  For example:

    python solve_one_puzzle.py "(((((((.(.(.(.(((((((....)))))))))))))))))"

The `best_model` directory contains the saved model parameters in the form of Tensorflow
checkpoint files.  These are the parameters that were used when running the tests in the
paper.  Alternatively, you can run the `train.py` script to retrain the model yourself.
This should take a few hours on a typical laptop.  Note that it will overwrite the
`best_model` directory.

-------------------------------------------------------------------


curl "CanonicalGroupLimited.Ubuntu20.04onWindows_2004.2022.8.0_neutral_~_79rhkp1fndgsc.appxbundle" -o ubuntu.appxbundle

curl "http://tlu.dl.delivery.mp.microsoft.com/filestreamingservice/files/9e127247-690c-441f-a9b6-95ec28d9104c?P1=1750291509&P2=404&P3=2&P4=QkGnUQGI4VguddWWb5XmwI%2bBgJ%2fJ71giZAWpg9DEsKY4Vd5iyEEYWXBHsUeQu8KqvaPFLHiXoEszW4udAtMUag%3d%3d.appxbundle" -o ubuntu.appxbundle

C:\Users\Oreshnya\AppData\Local\Programs\Python\Python36\python.exe -m venv rl_2018   



conda create -n rna-rl python=3.6.15 mamba
conda activate rna-rl

# conda-forge for general pkgs, bioconda for ViennaRNA,
# deepchem & omnia for the ancient deepchem build,
# cf201901 label for the tar.bz2 TensorFlow 1.3 build
conda config --env --add channels conda-forge
conda config --env --add channels bioconda
conda config --env --add channels deepchem
conda config --env --add channels omnia
conda config --env --add channels conda-forge/label/cf201901
conda config --env --add channels rdkit


# Core stack (≈200 MB download, a few minutes on WSL2)
mamba install \
  tensorflow=1.3.0 \
  deepchem=1.3.1 \
  viennarna=2.3.5 \
  numpy=1.15  scipy=1.1  pandas=0.23  scikit-learn=0.19

  mamba install --strict-channel-priority \
  tensorflow=1.3.0 \
  deepchem=1.3.1 \
  viennarna=2.3.5 \
  rdkit=2017.03.1 \
  numpy=1.15 scipy=1.1 pandas=0.23 scikit-learn=0.19

# Check
python -c "import tensorflow as tf, deepchem, RNA, rdkit; print('TF', tf.__version__, 'DC', deepchem.__version__, 'RNA OK', RNA.pf_fold('(.)'), 'RDKit OK')"

---- try 2

conda create --no-default-packages --no-channel-priority -n rna-rl python=3.6.15

conda activate rna-rl

export CONDA_SUBDIR=linux-64

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo "export CONDA_SUBDIR=linux-64" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

conda config --env --add channels deepchem
conda config --env --add channels omnia
conda config --env --add channels rdkit
conda config --env --add channels bioconda
conda config --env --add channels conda-forge
conda config --env --add channels defaults
