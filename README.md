# CPTS440Project

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

For an example of how to setup learning, check ./neural-net/cleanRL_implementation.py

For an example of how to setup a poke-env player, check ./neural-net/NN_AlgoPlayer.py, or look at the poke-env docs.
