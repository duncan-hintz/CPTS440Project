# CPTS440Project

Demo For Final submission:


It is important to note that due to the nature of this project running a node.js server, it is impossible to run on a single CoLab notebook. 
Instructions for setting up a local version are in this document, and a video demonstration of the working code is at the following link:
 
https://youtu.be/IT2O8cigtFQ

Finally, due to Github file size restraints, we are unable to submit some weights from recent training on larger networks (100MB max, most weights are 300+ on recent versions)
If this causes an issue of one module expecting a different node like in the AlgoPlayer demo, simply run cleanRL_implementation once with mode = 0, and the resulting weights will be saved locally.


Dependencies:

node/npm

python 3.12+ (working in 3.12)


Setup:

First, run:

pip install poke-env

in the main directory.

Before running a game, do the following:
1. change the cwd to ./pokemon-showdown/
2. run:  node pokemon-showdown start --no-security
This will start a local showdown server that doesn't authenticate login, allowing our bots.
3. Now in a separate terminal, run the file desired.
For an example game, run ./env/testing.py
This will simulate one game with random input that can be viewed in the browser.
This is good for debugging the customEnv class as it steps through moves in a battle.

To run a batch of training, run /cleanRL_implementation.py

Recommended use of parameters in that file are:

fixed=True

Mode = 0


For an example of how to setup a poke-env player, check ./neural-net/NN_AlgoPlayer.py \[DEPRECATED\], or look at the poke-env docs @ https://poke-env.readthedocs.io/en/stable/

To run a demo against a human player, run NNAlgoPlayer_demo.py with the chosen username that is logged in to the local server.