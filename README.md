# MultiProp
### A framework for propagating multiple sources of information in Graph Neural Networks

  [Joshua Placidi](https://www.linkedin.com/in/joshua-placidi/), [John Pate](https://www.linkedin.com/in/john-pate-07451526/), [Tania Bakhos](https://www.linkedin.com/in/tania-bakhos-745b13172/), [Pawel Pomorski](https://www.linkedin.com/in/pawel-pomorski-90754910b/)

This work was conducted as the thesis component of my (Joshua Placidi) Masters degree in Artifical Intelligence at The University of Edinburgh in collaboration with Amazon UK, the project supervisors were John Pate, Tania, Bakhos, and Pawel Pamorski.

***


Supervised learning with Graph Neural Networks (GNNs) typically involves propagating feature information to predict labels, with MultiProp we introduce an architecture for jointly spreading both feature and label information in an inductive GNN framework.





if labels are also propagated the size of training data can be expanded without increasing the size of the dataset, potentially improving model performance. We present an end-to-end multi-propagation GNN that performs feature and label message passing in an inductive framework that generalises to unseen data.

Graph Neural Networks (GNNs) iteratively propagate information through nodes in a [graph](https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)), for a comprehenive introduction to GNNs I recommend reading [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/), an interactive article written by researchers at Google.
A typical GNN approach to a supervised graph task is to spread feature information through nodes via their edges, with MultiProp we experiment with spreading feature informations and label information at the node level.







by Joshua Placidi.
under the supervision of John Pate, Tania Bahkos, and Pawel Pomorski, all of Amazon UK.

The University of Edinburgh in collaboration with Amazon UK

File description:  

config.py: contains variable that need to be configured for the system. 

data.py: function for loading/downloading dataset from Open Graph Benchmark. 

logger.py: helper class for evaluating model performance, generates evalaution metrics and plots. 

run.py: file used to specify and run experiments. 

run_baselines.py: runs all baseline models for a particular experiment. 

run_hyperparameter_search.py: runs a hyperparameter search. 

training.py: core implementation for training graphs is here, contains all the functions for running, training, evaluation, and testing.


/models. 

gnn.py: class for implementing graph neural networks. 

mlp.py: implementation of an mlp. 

multiprop.py: implementation of our proposed multiprop model. 




The dataset used in this project is Open Graph Benchmark (https://ogb.stanford.edu/) protein dataset. 

run.py contains some commented out lines that can be used to run experiments.
