# Averaging Neural Networks in Parameter Space

## Topic Overview

This is an implementation of my Master's Thesis concerning a new neural network ensembling method. Neural Networks are most commonly ensembled in function space, i.e. by averaging over the M models in the ensemble. These ensembles are often referred to as "DeepEnsembles". Unfortunately, DeepEnsembles require storing M models and M forward passes at inference time and for this reason their use is not widespread, despite their large accuracy increases.

One established way of overcoming these computational constraints is known as Knowledge Distillation. We leave the details of Knowledge Distillation to the demo, but mention it as it is an important benchmark for our novel approach.

We propose a new ensemble method named "PermAVG" to overcome the computational constraints of DeepEnsembles. Rather than averaging in function space, the PermAVG method averages the M models in parameter space, such that an ensemble of M models can be reduced to a single model. In this way we are able to overcome the computational constraints of DeepEnsembles, while hopefully maintaining accuracy gains. The "PermAVG" method seeks to learn the optimal permutations such that the average over the weights of M models is optimal.

## Creating the Conda Environment

In order to run the demo.ipynb you must first create the conda environment. You can do this simply by running:

    conda env create -f freeze.yml

Then we need to add the project to the PYTHONPATH. You can do this by running conda develop on the repository as follows.:

    cd ..
    conda develop ensembling_in_parameter_space

In case you experience the following error:

    CommandNotFoundError: To use 'conda develop', install conda-build.

You need to install the conda-build tools. You can do this by running:

    conda install conda-build

## Running the demo

In order to run the demo, we must run the conda environment with the conda environement activate. Hence, do the following:

1) Activate the conda environment

    conda activate perm_avg

2) Run Jupyter notebook in the active environment

    jupyter notebook


