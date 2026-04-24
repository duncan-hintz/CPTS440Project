import asyncio

from poke_env.environment.doubles_env import DoublesEnv

from typing import Optional, Union

import numpy as np
from gymnasium.spaces import Discrete

from poke_env.battle import AbstractBattle
from poke_env.battle.double_battle import DoubleBattle
#from poke_env.environment.env import ObsType

from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status

from poke_env.ps_client import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    ServerConfiguration,
)
from poke_env.teambuilder import Teambuilder

from poke_env.data import GenData #poke-env update

import webbrowser

from time import sleep

import pickle

from collections import defaultdict

class CustomEnv(DoublesEnv):
    metadata={}

    def __init__(
        self,
        account_configuration1: Optional[AccountConfiguration] = None,
        account_configuration2: Optional[AccountConfiguration] = None,
        avatar: Optional[int] = None,
        battle_format: str = "gen9randomdoublesbattle",
        log_level: Optional[int] = None,
        save_replays: Union[bool, str] = False,
        server_configuration: Optional[
            ServerConfiguration
        ] = LocalhostServerConfiguration,
        accept_open_team_sheet: Optional[bool] = False,
        start_timer_on_battle_start: bool = False,
        start_listening: bool = True,
        open_timeout: Optional[float] = 10.0,
        ping_interval: Optional[float] = 20.0,
        ping_timeout: Optional[float] = 20.0,
        challenge_timeout: Optional[float] = 60.0,
        team: Optional[Union[str, Teambuilder]] = None,
        fake: bool = False,
        strict: bool = False,
        render_mode: Optional[str] = None,
    ):
        super().__init__(
            account_configuration1=account_configuration1,
            account_configuration2=account_configuration2,
            avatar=avatar,
            battle_format=battle_format,
            log_level=log_level,
            save_replays=save_replays,
            server_configuration=server_configuration,
            accept_open_team_sheet=accept_open_team_sheet,
            start_timer_on_battle_start=start_timer_on_battle_start,
            start_listening=start_listening,
            open_timeout=open_timeout,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            challenge_timeout=challenge_timeout,
            team=team,
            fake=fake,
            strict=strict,
        )
        self.gen_data = GenData.from_format(battle_format) #poke-env update
        self.action_spaces = {
            agent: Discrete(107*107) for agent in self.possible_agents
        }
        self.render_browser_open=False
        self.render_mode=render_mode

        #Load in dicts:
        """Generic to copy:

        self.*REPLACE*MapPath="mappings/*REPLACE*Dict.txt"
        try:
            with open(self.*REPLACE*MapPath, "rb") as *REPLACE*File:
                self.*REPLACE*_dict = pickle.load(*REPLACE*File)
                self.*REPLACE*_dict = defaultdict(str,self.*REPLACE*_dict)
        except:
            self.*REPLACE*_dict=defaultdict(str)
            with open(self.*REPLACE*MapPath,"wb") as *REPLACE*File:
                pickle.dump(self.*REPLACE*_dict, *REPLACE*File)
        
        self.*REPLACE*_dict_index=len(self.*REPLACE*_dict)-1
        """


        #Item Dict
        self.itemMapPath="mappings/itemDict.txt"
        try:
            with open(self.itemMapPath, "rb") as itemFile:
                self.item_dict = pickle.load(itemFile)
                self.item_dict = defaultdict(str,self.item_dict)
        except:
            self.item_dict=defaultdict(str)
            self.item_dict['unknown_item']=-1
            self.item_dict['']=0
            with open(self.itemMapPath,"wb") as itemFile:
                pickle.dump(self.item_dict, itemFile)
        
        self.item_dict_index=len(self.item_dict)-1

        
        #Ability Dict
        self.abilityMapPath="mappings/abilityDict.txt"
        try:
            with open(self.abiilityMapPath, "rb") as abilityFile:
                self.ability_dict = pickle.load(abilityFile)
                self.ability_dict = defaultdict(str,self.ability_dict)
        except:
            self.ability_dict=defaultdict(str)
            with open(self.abilityMapPath,"wb") as abilityFile:
                pickle.dump(self.ability_dict, abilityFile)
        
        self.ability_dict_index=len(self.ability_dict)-1
        

        
    
    def reset(self,seed=None,options=None):
        self.render_browser_open = False
        toReturn= super().reset(seed=seed,options=options)
        """if(not seed is None):
            if hasattr(self.agent1.ps_client, "websocket"):
                asyncio.run(self.agent1.ps_client.send_message(f"ebat reseed 00000,00000,00000,000{seed}"))"""
        return toReturn

    def get_mask(self, battle: AbstractBattle):
        #Initial action masking for gen 9, removing other gimmicks
        action_mask=[0,]*107
        action_mask2=[0,]*107

        #orders is in the form of [[Orders],[Orders]] where each nested list is the valid orders for each slot
        #it does not account for invalid moves together (like double tera)
        for i in range(2):
            for order in battle.valid_orders[i]:
                orderNum=DoublesEnv._order_to_action_individual(order=order,battle=battle,fake=False,pos=i)
                if(i==0):
                    action_mask[orderNum]=1
                else:
                    action_mask2[orderNum]=1

        #DoubleBattleOrder.join_orders()

        #Converting the masks to the proper type
        action_mask=np.array(action_mask,dtype=np.int8)
        action_mask2=np.array(action_mask2,dtype=np.int8)
        #Combine into a single dimensional array of all combinations
        action_mask_combined=np.sum(np.array(np.meshgrid(action_mask, action_mask2),dtype=np.int8).T.reshape(-1, 2),axis=1)
        #If only one choice was valid, mask the combination
        action_mask_combined[action_mask_combined <2]=0
        action_mask_combined[action_mask_combined==2]=1
        #Now mask illegal combined operations:
        #To get combination index of [x,y], do: [107*x + y]
        #To undo: num//107 = x, num%107=y
        #Both pass:
        action_mask_combined[0]=0
        #Both switch to same target:
        for i in range(1,7):
            action_mask_combined[108*i]=0
        #Both tera:
        for i in range(87,107):
            for j in range(87,107):
                action_mask_combined[107*i + j]=0
        action_mask_combined=np.array(action_mask_combined,dtype=np.int8)
        return action_mask_combined



    def embed_pokemon(self, pokemon: Pokemon):
        #current number of tracked observations, used for unknown pokemon
        
        if(pokemon==None):
            #Default returns
            return [
                    0, #not active
                    1.0, #full hp
                    -1.0, #unknown_item
                    PokemonType.THREE_QUESTION_MARKS.value / 20, #type1 unknown
                    PokemonType.THREE_QUESTION_MARKS.value / 20, #type2 unknown
                    -1, #base stats unknown
                    -1, #" "
                    -1, #" "
                    -1, #" "
                    -1, #" "
                    -1, #" "
                    0, #No status
                    0, #No boosts
                    0, #" "
                    0, #" "
                    0, #" "
                    0, #" "
                    0, #" "
                    -1, #level unknown
                    -1, #stats unknown
                    -1, #" "
                    -1, #" "
                    -1, #" "
                    -1, #" "
                    -1, #" "
                    -1, #Max hp unknown
                    -1, #Current hp unknown
                    -1, #Unknown ability
            ]
        
        #Embed mappings and check if they need to be updated to files:

        #Item:
        if(pokemon.item in self.item_dict):
            item=self.item_dict[pokemon.item]
        else: #new item
            item=self.item_dict_index
            #add to dict
            self.item_dict[pokemon.item]=self.item_dict_index
            #increase index
            self.item_dict_index=self.item_dict_index+1
            #save to file now, avoids rewriting later if this run is terminated early
            #Will be slow in first iterations, but cost nothing later once most items are added
            with open(self.itemMapPath,"wb") as itemFile:
                pickle.dump(self.item_dict, itemFile)

        if(not item == -1 or not item==0):
            item = item/(self.item_dict_index-1)

        #Ability:
        if(pokemon.ability is None):
            ability=-1
        elif(pokemon.ability in self.ability_dict):
            ability=self.ability_dict[pokemon.ability] / (self.ability_dict_index-1)
        else:
            ability=1
            self.ability_dict[pokemon.ability]=self.ability_dict_index
            self.ability_dict_index=self.ability_dict_index+1
            with open(self.abilityMapPath,"wb") as abilityFile:
                pickle.dump(self.ability_dict, abilityFile)

        base_stat_out = [value/255 if not value is None else -1 for value in pokemon.base_stats.values()]
        boosts_out  = [boost/8 if not boost is None else 0 for boost in pokemon.boosts.values()]
        stats_out = [value/255 if not value is None else -1 for value in pokemon.stats.values()]

        toReturn = [
            pokemon.active,
            pokemon.current_hp_fraction,
            item,
            pokemon.type_1.value /19,
            (pokemon.type_2.value if not pokemon.type_2 is None else PokemonType.THREE_QUESTION_MARKS.value) / 19,
            base_stat_out[0],
            base_stat_out[1],
            base_stat_out[2],
            base_stat_out[3],
            base_stat_out[4],
            base_stat_out[5],
            (pokemon.status.value if not pokemon.status is None else 0)/6,
            boosts_out[0],
            boosts_out[1],
            boosts_out[2],
            boosts_out[3],
            boosts_out[4],
            boosts_out[5],
            pokemon.level/100,
            stats_out[0],
            stats_out[1],
            stats_out[2],
            stats_out[3],
            stats_out[4],
            stats_out[5],
            pokemon.max_hp/714,
            pokemon.current_hp/714,
            ability,
        ]

        return toReturn
    
    def embed_battle(self, battle: AbstractBattle):# -> tuple[ObsType,dict[int:int]]:
        """
        Returns the embedding of the current battle state in a format compatible with
        the Gymnasium API.

        :param battle: The current battle state.
        :type battle: DoubleBattle

        :return: The embedding of the current battle state.
        """
        if(battle==None):
            #Used for len returns of size of embed
            return np.zeros(18+(12*(len(CustomEnv.embed_pokemon(None,None)))))

        assert isinstance(battle, DoubleBattle)

        if(battle.finished):
            return {"observations":None,"action_mask":None}

        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = np.zeros(8)
        moves_dmg_multiplier = np.ones(8)
        #For each available move per active pokemon on the team, 
        #embed base power and each damage multiplier

        for i, move in enumerate(battle.available_moves[0]):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if battle.opponent_active_pokemon[0] is not None:
                for active_pokemon in battle.opponent_active_pokemon:
                    if active_pokemon is not None:
                        moves_dmg_multiplier[i] = move.type.damage_multiplier(
                            active_pokemon.type_1,
                            active_pokemon.type_2,
                            type_chart=self.gen_data.type_chart, #new Poke-env updated version
                        )
        for j, move in enumerate(battle.available_moves[1]):
            moves_base_power[3+j] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if battle.opponent_active_pokemon[0] is not None:
                for active_pokemon in battle.opponent_active_pokemon:
                    if active_pokemon is not None:
                        moves_dmg_multiplier[3+j] = move.type.damage_multiplier(
                            active_pokemon.type_1,
                            active_pokemon.type_2,
                            type_chart=self.gen_data.type_chart, #new Poke-env updated version
                        )
        

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        
        team_mons=[]
        for self_mon in battle.team.values():
            team_mons.append(self.embed_pokemon(self_mon))

        team_mons=np.concatenate(team_mons)

        opponent_mons=[]
        for opp_mon in battle.opponent_team.values():
            opponent_mons.append(self.embed_pokemon(opp_mon))

        #if unknown on other mons:
        for i in range(6-len(opponent_mons)):
            opponent_mons.append(self.embed_pokemon(None))

        opponent_mons=np.concatenate(opponent_mons)

        # Final vector with n components
        final_vector = np.concatenate(
            [
                moves_base_power, #The eight available moves
                moves_dmg_multiplier, #For each available move, the damage multiplier against the active pokemon
                [fainted_mon_team, fainted_mon_opponent], #The fainted pokemon on each team
                team_mons,
                opponent_mons,
            ]
        )

        action_mask=self.get_mask(battle)

        if self.render_mode == "human":
            self.render()

        toReturn={"observations":np.float32(final_vector),"action_mask":action_mask}

        return toReturn



    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=20.0, hp_value=10.0, victory_value=100.0,status_value=2.0
        )
    
    def step(self,actions):
        #Convert single nums to double
        actions={a:(actions[a]//107,actions[a]%107) for a in self.agents}
        return super().step(actions)
    
    def render(self):        
        if self.battle1 is not None:
            if(not self.render_browser_open):
                url = "https://localhost.psim.us/" + self.battle1.battle_tag
                webbrowser.open(url, new=0, autoraise=True)
                self.render_browser_open = True
                sleep(3)

            print(
                "  Turn %4d. | [%s][%3d/%3dhp] %10.10s, [%3d/%3dhp] %10.10s - %10.10s [%3d%%hp], %10.10s[%3d%%hp][%s]"
                % (
                    self.battle1.turn,
                    "".join(
                        [
                            "⦻" if mon.fainted else "●"
                            for mon in self.battle1.team.values()
                        ]
                    ),
                    0 if self.battle1.active_pokemon[0]==None else self.battle1.active_pokemon[0].current_hp,
                    0 if self.battle1.active_pokemon[0]==None else self.battle1.active_pokemon[0].max_hp,
                    "None" if self.battle1.active_pokemon[0]==None else self.battle1.active_pokemon[0].species,
                    0 if self.battle1.active_pokemon[1]==None else self.battle1.active_pokemon[1].current_hp,
                    0 if self.battle1.active_pokemon[1]==None else self.battle1.active_pokemon[1].max_hp,
                    "None" if self.battle1.active_pokemon[1]==None else self.battle1.active_pokemon[1].species,
                    "None" if self.battle1.opponent_active_pokemon[0]==None else self.battle1.opponent_active_pokemon[0].species,
                    0 if self.battle1.opponent_active_pokemon[0]==None else self.battle1.opponent_active_pokemon[0].current_hp,
                    "None" if self.battle1.opponent_active_pokemon[1]==None else self.battle1.opponent_active_pokemon[1].species,
                    0 if self.battle1.opponent_active_pokemon[1]==None else self.battle1.opponent_active_pokemon[1].current_hp,
                    "".join(
                        [
                            "⦻" if mon.fainted else "●"
                            for mon in self.battle1.opponent_team.values()
                        ]
                    ),
                ),
                end="\n",
                # end="\n" if self.battle1.finished else "\r",
            )
