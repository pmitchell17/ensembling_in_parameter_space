# Averaging Neural Networks in Parameter Space

## Topic Overview

This is an implementation of my Master's Thesis concerning a new neural network ensembling method. Neural Networks are most commonly ensembled in function space, i.e. by averaging over the M models in the ensemble. These ensembles are often referred to as "DeepEnsembles". Unfortunately, DeepEnsembles require storing M models and M forward passes at inference time and for this reason their use is not widespread, despite their large accuracy increases.

One established way of overcoming these computational constraints is known as Knowledge Distillation. We leave the details of Knowledge Distillation to the demo, but mention it as it is an important benchmark for our novel approach.

We propose a new ensemble method named "PermAVG" to overcome the computational constraints of DeepEnsembles. Rather than averaging in function space, the PermAVG method averages the M models in parameter space, such that an ensemble of M models can be reduced to a single model. In this way we are able to overcome the computational constraints of DeepEnsembles, while hopefully maintaining accuracy gains. The "PermAVG" method seeks to learn the optimal permutations such that the average over the weights of M models is optimal.

## Running the Demo

In order to run the demo.ipynb you must first create the conda environment. You can do this simply by running:

    conda env create -f environment.yml

In order to have the conda environment active in jupyter notebook:

1) Activate the conda environment

    conda activate perm_avg

2) Run Jupyter notebook in the active environment

    jupyter notebook


