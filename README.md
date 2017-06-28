## (Machine) Learning to Do More with Less
***

 * Timothy Cohen
 * Marat Freytsis
 * Bryan Ostdiek

### Abstract
Determining the best method for training a machine learning algorithm is critical to maximizing its ability to classify data. In this paper, we compare the standard "fully supervised" approach (that relies on knowledge of event-by-event truth-level labels) with a recent proposal that instead utilizes class ratios as the only discriminating information provided during training.  This so-called "weakly supervised" technique has access to less information than the fully supervised method and yet is still able to yield impressive discriminating power.  In addition, weak supervision seems particularly well suited to particle physics since quantum mechanics is incompatible with the notion of mapping an individual event onto any single Feynman diagram. We examine the technique in detail -- both analytically and numerically --  with a focus on the robustness to issues of mischaracterizing the training samples.  Weakly supervised networks turn out to be remarkably insensitive to systematic mismodeling. Furthermore, we demonstrate that the event level outputs for weakly versus fully supervised networks are probing different kinematics, even though the numerical quality metrics are essentially identical. This implies that it should be possible to improve the overall classification ability by combining the output from the two types of networks. For concreteness, we apply this technology to a signature of beyond the Standard Model physics to demonstrate that all these impressive features continue to hold in a scenario of relevance to the LHC.

### Notes
 * WeakSupervisionDemo.ipynb is a jupyter notebook that contains a quick demo on how to train weakly supervised networks with Keras. It also shows the toy model distributions used for our paper.

* The 'Programs' directory holds a python script to generate data for the section of the paper on mislabeled data. There is also a jupyter notebook that covers all of the BSM section of the paper.

* 'Functions' contains a sample generator for our bi-modal distributions. An example of how to use this function can be found in the WeakSupervisionDemo.ipynb notebook. The explicit definition of the weak cost function is also in this directory.

* Lastly, the 'Data' folder contains both the BSM Monte Carlo data, saved Keras model weights, and the results of the mis-labeled data sets.
