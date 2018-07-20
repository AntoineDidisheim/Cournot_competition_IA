This projects proposes simulaiton of a Cournot economy were various numbers of IAs are participaticipating.

Each IA can be chosen to follows one of many à la mode algorithm (from deep net to random forrest)

The IAs can recieve as input only their own production or also t-1 productions and profits of other competititors

The number of competitors can be defined as anything from one to as much as your computer can handle.

The nnet IA have been codedon Tensorflow, hence this application can easily be extended to large CPU networks.

the wip.py file calls a simple simulation with graphical illustration of the results

the sim_sim.py calls a lot of simulation to show the convergence of the results.

The cournot file contain a class defining the parameters of the game or economy

The player file contains sub-classes of potential players with different techniques of optimisation. 