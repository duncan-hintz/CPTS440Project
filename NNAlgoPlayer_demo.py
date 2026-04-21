from NN_AlgoPlayer import AlgoPlayer
import asyncio

state_path= "./models/testing"
myPlayer = AlgoPlayer(state_path=state_path)
asyncio.run(myPlayer.send_challenges("daisy_hearts1",n_challenges=1))