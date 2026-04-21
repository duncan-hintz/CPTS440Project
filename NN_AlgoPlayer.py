#Needed for playing
from poke_env.player import Player
from customEnv import CustomEnv
from poke_env.environment import DoublesEnv

#Needed for neural net specifically, could be replaced
import torch
from cleanRL_implementation import Agent


class AlgoPlayer(Player):
    def __init__(self,state_path,num_actions=11449,informat="gen9randomdoublesbattle",inteam=None):

        #NN Agent:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent=Agent(num_actions=num_actions,mode=1).to(self.device)
        self.agent.load_state_dict(torch.load(f"{state_path}/agent.pt",map_location=self.device))
        self.agent.eval()

        #Used for parsing battle info
        self.function_env=CustomEnv()
        #Setup as a player for Poke_env
        super().__init__(battle_format=informat,team=inteam)

    #This is the function poke_env calls when a move is needed
    def choose_move(self, battle):

        #Get the observations for the current state of the battle using CustomEnv.embed_battle
        #Change embed_battle for different reasoning
        #obs is {"observations":, "action_mask":}
        obs = self.function_env.embed_battle(battle=battle)

        #The following section parses the observation into a format used by the neural net
        #Alter for your own parsing as needed

        """Replacement for batchify_obs :"""
        # convert to list of np arrays
        mask = obs["action_mask"]
        obs = obs["observations"]
        #obs = np.stack([obs[a] for a in obs], axis=0)
        #mask = np.stack([mask[a] for a in mask], axis=0)

        # convert to torch
        mask = torch.tensor(mask).to(self.device)
        obs = torch.tensor(obs).to(self.device)
        obs = (obs,mask)
        """End replacement of batchify_obs"""
        
        #print(obs[0])

        #Get action using the agent, Replace with other logic as needed
        #the agent is a neural net in this case
        #Action is a sum in the form of: action_1*107 + action_2
        actions, logprobs, _, values = self.agent.get_action_and_value(obs)
        actions=actions.cpu().numpy()

        #Parse single action into two actions
        #i.e. from 21087 to (62,103)   [math in this comment is not accurate, just the format]
        actions=(actions//107,actions%107)

        #print(mask.nonzero())
        #print(actions)

        #Turn tuple of numbers into words processed by the env
        return DoublesEnv.action_to_order(action=actions,battle=battle)