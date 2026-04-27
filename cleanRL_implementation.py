import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from customEnv import CustomEnv

from poke_env.player import RandomPlayer
from poke_env.player.player import Player
from poke_env.environment.doubles_env import DoublesEnv
from poke_env import AccountConfiguration

from tabulate import tabulate
import asyncio

#import bad_viz
#import os
#os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

fixed = False
if(fixed):
    format="gen9doublesou"
    #Define the teams
    team1="""Charizard||LifeOrb|Blaze|hurricane,heatwave,scorchingsands,protect||85,,85,85,85,85|M|,0,,,,||82|,,,,,Fire]Malamar||SitrusBerry|Contrary|protect,trickroom,knockoff,superpower||85,85,85,85,85,|F|,,,,,0||80|,,,,,Fighting]Hydrapple||ChoiceSpecs|Regenerator|earthpower,leafstorm,gigadrain,dracometeor||85,,85,85,85,85|F|,0,,,,||85|,,,,,Fire]Tornadus||SitrusBerry|Prankster|knockoff,bleakwindstorm,tailwind,heatwave||85,85,85,85,85,85|M|||77|,,,,,Steel]Barraskewda||LifeOrb|PropellerTail|psychicfangs,waterfall,protect,closecombat||85,85,85,85,85,85|F|||85|,,,,,Fighting]Arcanine||HeavyDutyBoots|Intimidate|willowisp,flareblitz,closecombat,morningsun||85,85,85,85,85,85|M|||82|,,,,,Fighting"""
    team2 = """
    """
    state_path="./models/testing/fixed_teams/"
else:
    format="gen9randomdoublesbattle"
    team1=None
    team2=None
    state_path="./models/testing/random_teams/"

observation_num=len(CustomEnv.embed_battle(None,None))

class Agent(nn.Module):
    def __init__(self, num_actions,mode=0,player=1):
        super().__init__()
        self.divisor=1
        self.network = nn.Sequential(
            self._layer_init(nn.Linear(observation_num, observation_num*4)),
            nn.ReLU(),
            self._layer_init(nn.Linear(observation_num*4, observation_num*64)),
            nn.ReLU(),
            self._layer_init(nn.Linear(observation_num*64, observation_num*16)),
            nn.ReLU(),
            self._layer_init(nn.Linear(observation_num*16, 36)),
            nn.ReLU(),
        )
        self.actor = self._layer_init(nn.Linear(36, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(36, 1))
        self.mode=mode
        """Modes:
        The mode changes the handling of the env input. In some modes, only one player is handled in the env.
        Playing on a validation set versus a real person also runs into some issues, so that requires another switch.
        Mode 0: Random p2, training set
        Mode 1: vs. player validation mode
        """

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(self.network(x / self.divisor))

    def get_action_and_value(self, x, action=None):
        #x = (obs,mask)
        #unless checking, then x is just obs
        if(action is None):
            hidden = self.network(x[0]/ self.divisor)
        else:
            hidden = self.network(x / self.divisor)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        
        #probsCopy=Categorical(logits=logits)
        if(action is None):
            #mask probs here
            if(not probs.probs[x[1]==1].any()):
                probs.probs=x[1]
                probs.probs=probs.probs/x[1].sum()
            else:
                probs.probs=probs.probs*x[1]
            probs.probs=torch.nan_to_num(probs.probs)
            #Normalize back to sum to 1, needed to reapply to Categorical
            if(self.mode==0 or self.mode==1):
                probs.probs=probs.probs/probs.probs.sum()
            else:
                probs.probs[0] = probs.probs[0]/probs.probs[0].sum()
                probs.probs[1] = probs.probs[1]/probs.probs[1].sum()
            probs.probs=torch.nan_to_num(probs.probs)

            #probs = Categorical(probs=probs.probs)
            
            #Need to filter for when turn is "wait"
            
            if(self.mode!=1):
                if((player==1 and env.battle1._wait) or (player==2 and env.battle2._wait)):
                    if(self.mode==0):
                        probs.probs[0]=1
                    else:
                        probs.probs[0][0]=1 
                if(self.mode!=0 and env.battle2._wait):
                    probs.probs[1][0]=1
                    
                if(not probs.probs.any()):
                    if(self.mode==0):
                        probs.probs[0]=1
                    else:
                        probs.probs[0][0]=1

        if action is None:
             action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def batchify_obs(obs, device,mode=0):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    mask = {a:obs[a]["action_mask"] for a in obs}
    obs = remove_mask(obs)
    #Dont actually remove mask, save for later
    obs = np.stack([obs[a] for a in obs], axis=0)
    mask = np.stack([mask[a] for a in mask], axis=0)

    # Our observations are agent:obs
    #Keep this, no need for transpose
    #obs = obs.transpose(0, -1, 1, 2)

    # convert to torch
    mask = torch.tensor(mask).to(device)
    obs = torch.tensor(obs).to(device)

    
    if(mode==0):
        #Random p2
        #Sets obs to remove p2s observations and mask
        obs = obs[0]
        mask = mask[0]


    return obs,mask

def remove_mask(obs):
    obs = {a:obs[a]["observations"] for a in obs}
    return obs
def batchify(x, device,mode=0):
    """Converts PZ style returns to batch of torch arrays."""
    if(mode==0):
        x=x[env.possible_agents[0]]
    elif(mode==1):
        y=x[env.possible_agents[1]]
        x=x[env.possible_agents[0]]
        x=torch.tensor(x).to(device)
        y=torch.tensor(y).to(device)
        return(x,y)
    else:
        # convert to list of np arrays
        x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a:x[i] for i, a in enumerate(env.possible_agents)}

    return x

class AlgoPlayer(Player):
    def __init__(self,state_path,num_actions=11449,informat="gen9randomdoublesbattle",inteam=None, filename="agent.pt"):

        #NN Agent:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent=Agent(num_actions=num_actions).to(self.device)
        self.agent.load_state_dict(torch.load(f"{state_path}/{filename}",map_location=self.device))
        self.agent.eval()


        self.function_env=CustomEnv()
        super().__init__(battle_format=informat,team=inteam)

    def choose_move(self, battle):
        #obs is {"observations":, "action_mask":}
        obs = self.function_env.embed_battle(battle=battle)

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

        #Get action using the agent, Replace with other logic if needed
        actions, logprobs, _, values = self.agent.get_action_and_value(obs)
        actions=actions.cpu().numpy()

        #Parse single action into two actions
        actions=(actions//107,actions%107)

        #print(mask.nonzero())
        #print(actions)

        #Turn tuple of numbers into words processed by the env
        return DoublesEnv.action_to_order(action=actions,battle=battle)

async def validation_games(p1,p2):
    await p1.battle_against(p2, n_battles=1)

if __name__ == "__main__":
    """
    Modes:
    0=random p2
    1=equal p2
    """
    mode = 0
    
    
    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    batch_size = 32
    lr=0.00001
    
    #stack_size = 4
    #frame_size = (64, 64)
    
    max_cycles = 125
    total_episodes = 1024

    """ ENV SETUP """
    if(fixed):
        env=CustomEnv(battle_format=format,team=team1)
    else:
        env=CustomEnv()
    
    if(mode==0):
        num_agents=1
    else:
        num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n
    observation_size = np.zeros(observation_num).shape

    """ LEARNER SETUP """
    agent = Agent(num_actions=num_actions).to(device)
    if(mode==1):
        agent.load_state_dict(torch.load(f"{state_path}agent_last.pt",map_location=device))
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    if(mode==1):
        agent2 = Agent(num_actions=num_actions,player=2).to(device)
        agent2.load_state_dict(torch.load(f"{state_path}agent_last.pt",map_location=device))
        agent2.eval()

    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0

    #rb_obs = torch.zeros((max_cycles, num_agents, stack_size, *frame_size)).to(device)
    rb_obs=torch.zeros((max_cycles, num_agents,observation_num)).to(device)

    rb_actions = torch.zeros((max_cycles, num_agents)).to(device)
    rb_logprobs = torch.zeros((max_cycles, num_agents)).to(device)
    rb_rewards = torch.zeros((max_cycles, num_agents)).to(device)
    rb_terms = torch.zeros((max_cycles, num_agents)).to(device)
    rb_values = torch.zeros((max_cycles, num_agents)).to(device)

    wins=0
    last10=[]
    last50=[]

    """ TRAINING LOGIC """
    # train for n number of episodes
    for episode in range(total_episodes):
        # collect an episode
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors
            if(fixed):
                next_obs, info = env.reset(seed=42)
            else:
                next_obs, info = env.reset(seed=None)

            # reset the episodic return
            total_episodic_return = 0

            # each episode has num_steps
            for step in range(0, max_cycles):
                for player in next_obs:
                    next_obs[player]=next_obs[player]['observation']
                
                # rollover the observation
                if(mode==0):
                    #get the mask for random sampling
                    p2mask=next_obs[env.possible_agents[1]]["action_mask"]
                    
                obs, mask = batchify_obs(obs=next_obs, device=device, mode=mode)
                if(mode==1):
                    obs2=(obs[1],mask[1])
                    obs=obs[0]
                    mask=mask[0]

                # get action from the agent
                actions, logprobs, _, values = agent.get_action_and_value((obs,mask))
                if(mode==1):
                    actions2, logprobs2, _2, values2 = agent2.get_action_and_value(obs2)
                if(mode==0):
                    actions=torch.tensor((actions,env.action_space(env.possible_agents[1]).sample(mask=p2mask))).to(device)
                elif(mode==1):
                    actions=torch.tensor((actions,actions2)).to(device)

                # execute the environment and log data
                next_obs, rewards, terms, truncs, infos = env.step(
                    unbatchify(actions, env)
                )

                # add to episode storage
                rb_obs[step] = obs[0]
                rb_rewards[step] = batchify(rewards, device)
                rb_terms[step] = batchify(terms, device)
                rb_actions[step]=actions[0]
                rb_logprobs[step] = logprobs
                rb_values[step] = values.flatten()

                # compute episodic return
                total_episodic_return += rb_rewards[step].cpu().numpy()

                # if we reach termination or truncation, end
                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    end_step = step
                    break

        # bootstrap value if not done
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            for t in reversed(range(end_step)):
                delta = (
                    rb_rewards[t]
                    + gamma * rb_values[t + 1] * rb_terms[t + 1]
                    - rb_values[t]
                )
                rb_advantages[t] = delta + gamma * gamma * rb_advantages[t + 1]
            rb_returns = rb_advantages + rb_values

        # convert our episodes to batch of individual transitions
        b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
        b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
        b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
        b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
        b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

        # Optimizing the policy and value network
        b_index = np.arange(len(b_obs))
        clip_fracs = []
        for repeat in range(3):
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)
            for start in range(0, len(b_obs), batch_size):
                # select the indices we want to train on
                end = start + batch_size
                batch_index = b_index[start:end]

                _, newlogprob, entropy, value = agent.get_action_and_value(
                    b_obs[batch_index], b_actions.long()[batch_index]
                )
                logratio = newlogprob - b_logprobs[batch_index]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                # normalize advantages
                advantages = b_advantages[batch_index]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Policy loss
                pg_loss1 = -b_advantages[batch_index] * ratio
                pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value = value.flatten()
                v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                v_clipped = b_values[batch_index] + torch.clamp(
                    value - b_values[batch_index],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                #get_dot=bad_viz.register_hooks(loss)
                loss.backward()
                #dot=get_dot()
                #dot.save('tmp.dot')
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        print(f"Training episode {episode+1}")
        show_loss = False
        if(show_loss):
            print(f"Episodic Return: {np.mean(total_episodic_return)}")
            print(f"Episode Length: {end_step}")
            print("")
            print(f"Value Loss: {v_loss.item()}")
            print(f"Policy Loss: {pg_loss.item()}")
            print(f"Old Approx KL: {old_approx_kl.item()}")
            print(f"Approx KL: {approx_kl.item()}")
            print(f"Clip Fraction: {np.mean(clip_fracs)}")
            print(f"Explained Variance: {explained_var.item()}")
            print("\n-------------------------------------------\n")
        if(len(last10)==10):
            last10.pop(0)
        last10.append(env.battle1.won)
        last10wr=len([w for w in last10 if w])/10.0
        if(len(last50)==50):
            last50.pop(0)
        last50.append(env.battle1.won)
        last50wr=len([w for w in last50 if w])/50.0
        if(env.battle1.won):
            print("Win")
            wins=wins+1
        else:
            print("Loss")
        wr=wins/(episode+1)   
        print(f"Current WR: {wr*100}%")
        print(f"Last 10 WR: {last10wr*100}%\n")
        print(f"Last 50 WR: {last50wr*100}%\n")



        


    #Save model
    torch.save(agent.state_dict(), f"{state_path}/agent.pt")

    """ RENDER THE POLICY """
    render=True
    if(render):
        if(fixed):
            env = CustomEnv(account_configuration1=AccountConfiguration("Learning", None),account_configuration2=AccountConfiguration("Bot", None),battle_format=format,team=team1,render_mode="human")
        else:
            env = CustomEnv(render_mode="human")

        agent.eval()

    with torch.no_grad():
        # render 1 episodes out
        num_render_episodes =1
        if(render):
            for episode in range(num_render_episodes):
                if(fixed):
                    obs, infos = env.reset(seed=42)
                else:
                    obs, infos = env.reset(seed=None)
                
                terms = [False]
                truncs = [False]
                while not any(terms) and not any(truncs):
                    for player in obs:
                        obs[player]=obs[player]['observation']

                    if(mode==0):
                        p2mask=obs[env.possible_agents[1]]["action_mask"]
                    obs = batchify_obs(obs, device,mode=mode)
                    if(mode==1):
                        obs2=(obs[0][1],obs[1][1])
                        obs=(obs[0][0],obs[1][0])
                    actions, logprobs, _, values = agent.get_action_and_value(obs)
                    if(mode==1):
                        actions2, logprobs2, _2, values2 = agent2.get_action_and_value(obs)
                    if(mode==0):
                        actions=torch.tensor((actions,env.action_space(env.possible_agents[1]).sample(mask=p2mask))).to(device)
                    elif(mode==1):
                        actions=torch.tensor((actions,actions2)).to(device)
                    obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
                    
                    terms = [terms[a] for a in terms]
                    truncs = [truncs[a] for a in truncs]
            
        
        #Create a random agent to simulate against
        if(mode==0):
            bot=RandomPlayer(battle_format=format,team=team1)
        elif(mode==1):
            bot=AlgoPlayer(state_path=state_path, informat=format,inteam=team1, filename="agent_last.pt")
        myPlayer = AlgoPlayer(state_path=state_path,informat=format,inteam=team1)
        players=[bot,myPlayer]

        num_challenges=1000

        cross_evaluation = {
            p1.username: {p2.username: None for p2 in players} for p1 in players
        }
        for i in range(num_challenges):
            asyncio.run(validation_games(bot,myPlayer))
            print(f"Validation game {i} complete")
        
        cross_evaluation[bot.username][myPlayer.username]=bot.win_rate
        cross_evaluation[myPlayer.username][bot.username]=myPlayer.win_rate

        table = [["-"] + [p.username for p in players]]
        for p_1, results in cross_evaluation.items():
            table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])

        print(tabulate(table))

        print(
        f"Player {myPlayer.username} won {myPlayer.n_won_battles} out of {myPlayer.n_finished_battles} played")

