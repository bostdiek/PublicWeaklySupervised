# On the robustness of Weakly Supervised Neural Networks

This page contains the code and data used for the paper found at ...

## Abstract

## Notes
 * WeakSupervisionDemo.ipynb is a jupyter notebook that contains a quick demo on how to train weakly supervised networks with Keras. It also shows the toy model distributions used for our paper.

* The 'Programs' directory holds a python script to generate data for the section of the paper on mislabeled data. There is also a jupyter notebook that covers all of the BSM section of the paper.

* 'Functions' contains a sample generator for our bi-modal distributions. An example of how to use this function can be found in the WeakSupervisionDemo.ipynb notebook. The explicit definition of the weak cost function is also in this directory.

* Lastly, the 'Data' folder contains both the BSM Monte Carlo data, saved Keras model weights, and the results of the mis-labeled data sets.
