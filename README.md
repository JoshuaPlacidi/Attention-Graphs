# Feature and Multi-Label Propagation in Graph Neural Networks

Joshua Placidi, The University of Edinburgh in collaboration with Amazon UK

File description:
|
|-> config.py: contains variable that need to be configured for the system
|-> data.py: function for loading/downloading dataset from Open Graph Benchmark
|-> logger.py: helper class for evaluating model performance, generates evalaution metrics and plots
|-> run.py: file used to specify and run experiments
|-> run_baselines.py: runs all baseline models for a particular experiment
|-> run_hyperparameter_search.py: runs a hyperparameter search
|-> training.py: core implementation for training graphs is here, contains all the functions for running, training, evaluation, and testing.
|-> models
     |-> gnn.py: class for implementing graph neural networks
     |-> mlp.py: implementation of an mlp
     |-> multiprop.py: implementation of our proposed multiprop model
