import asyncio

import cProfile
from pstats import Stats

from functools import lru_cache

from poke_env.environment.doubles_env import DoublesEnv

from typing import Optional, Union, List

import numpy as np
from gymnasium.spaces import Discrete

from poke_env.battle import AbstractBattle
from poke_env.battle.double_battle import DoubleBattle
#from poke_env.environment.env import ObsType

from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status

#The below are used to replace double_battle.valid_orders
from poke_env.battle.target import Target
from poke_env.battle.effect import Effect
from poke_env.battle.move import SPECIAL_MOVES, Move, _PROTECT_MOVES
from poke_env.battle.move_category import MoveCategory
from poke_env.player.battle_order import (
    DefaultBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)

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
        fake: bool = True,
        strict: bool = True,
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
            with open(self.abilityMapPath, "rb") as abilityFile:
                self.ability_dict = pickle.load(abilityFile)
                self.ability_dict = defaultdict(str,self.ability_dict)
        except:
            self.ability_dict=defaultdict(str)
            with open(self.abilityMapPath,"wb") as abilityFile:
                pickle.dump(self.ability_dict, abilityFile)

                
        self.ability_dict_index=len(self.ability_dict)-1

        #Move Dict
        self.moveMapPath="mappings/moveDict.txt"
        try:
            with open(self.moveMapPath, "rb") as moveFile:
                self.move_dict = pickle.load(moveFile)
                self.move_dict = defaultdict(str,self.move_dict)
        except:
            self.move_dict=defaultdict(str)
            with open(self.moveMapPath,"wb") as moveFile:
                pickle.dump(self.move_dict, moveFile)

                
        self.move_dict_index=len(self.move_dict)-1
        

        self.targetDictKeyList=[
                Target.from_showdown_message("adjacentAlly"),
                Target.from_showdown_message("adjacentAllyOrSelf"),
                Target.from_showdown_message("adjacentFoe"),
                Target.from_showdown_message("all"),
                Target.from_showdown_message("allAdjacent"),
                Target.from_showdown_message("allAdjacentFoes"),
                Target.from_showdown_message("allies"),
                Target.from_showdown_message("allySide"),
                Target.from_showdown_message("allyTeam"),
                Target.from_showdown_message("any"),
                Target.from_showdown_message("foeSide"),
                Target.from_showdown_message("normal"),
                Target.from_showdown_message("randomNormal"),
                Target.from_showdown_message("scripted"),
                Target.from_showdown_message("self"),
                0,
                None,
        ]
        self.targetDict={
            self.targetDictKeyList[2]:[1,2],#adjacentFoe
            self.targetDictKeyList[3]:[0],
            self.targetDictKeyList[4]:[0],
            self.targetDictKeyList[5]:[0],
            self.targetDictKeyList[6]:[0],
            self.targetDictKeyList[7]:[0],
            self.targetDictKeyList[8]:[0],
            self.targetDictKeyList[10]:[0],
            self.targetDictKeyList[12]:[0],
            self.targetDictKeyList[13]:[0],
            self.targetDictKeyList[14]:[0],
            self.targetDictKeyList[15]:[0],
            self.targetDictKeyList[16]:[1,2],
        }
        #self.pr = cProfile.Profile()
        

        
    
    def reset(self,seed=None,options=None):
        self.render_browser_open = False
        toReturn= super().reset(seed=seed,options=options)
        """if(not seed is None):
            if hasattr(self.agent1.ps_client, "websocket"):
                asyncio.run(self.agent1.ps_client.send_message(f"ebat reseed 00000,00000,00000,000{seed}"))"""
        return toReturn


    @lru_cache(maxsize=1024)
    def from_showdown_message(self, message: str):
        message = message.replace("move: ", "").translate(str.maketrans({" ": "_", "-": "_"}))
        
        # manual CamelCase split (faster than regex)
        tokens = []
        current = []

        for c in message:
            if c.isupper() and current:
                tokens.append("".join(current))
                current = [c]
            else:
                current.append(c)

        if current:
            tokens.append("".join(current))

        return Target["_".join(tokens).upper()]

    def deduce_move_target(self,move,entry):
        #Redo target
        req_targ=self.from_showdown_message(entry["target"])
        if "target" in entry:
            targ=req_targ
        else:
            targ=None

        if move.id in SPECIAL_MOVES:
            return targ
        elif req_targ:
            return req_targ
        elif targ == "randomNormal":
            return req_targ
        return targ

    def get_possible_showdown_targets(
        self,battle, move, pokemon: Pokemon, dynamax: bool = False
    ):
        """
        Given move of an ALLY Pokemon, returns a list of possible Pokemon Showdown
        targets for it. This is smart enough so that it figures whether the Pokemon
        is already dynamaxed.

        :param move: Move instance for which possible targets should be returned
        :type move: Move
        :param pokemon: The ally using the move.
        :type pokemon: Pokemon
        :param dynamax: whether given move also STARTS dynamax for its user
        :return: a list of integers indicating Pokemon Showdown targets:
            -1, -2, 1, 2 or self.EMPTY_TARGET_POSITION that indicates "no target"
        :rtype: List[int]
        """
        #self.pr.enable()
        if move._id in SPECIAL_MOVES:
            #self.pr.disable()
            return [0]

        pokemon_1, pokemon_2 = battle._active_pokemon[f"{battle._player_role}a"], battle._active_pokemon[f"{battle._player_role}b"]
        if pokemon_1 is None or not pokemon_1.active or pokemon_1.fainted:
            pokemon_1 = None
        if pokemon_2 is None or not pokemon_2.active or pokemon_2.fainted:
            pokemon_2 = None
        if pokemon == pokemon_1 and move._id in [m._id for m in battle.available_moves[0]]:
            self_position = -1
            ally_position = -2
        elif pokemon == pokemon_2 and move._id in [
            m._id for m in battle.available_moves[1]
        ]:
            self_position = -2
            ally_position = -1
        else:
            raise Exception(
                f"Selected move {move._id} is not owned by any active ally Pokemon "
                f"that is currently battling"
            )
        entry= {"pp": 1, "type": "normal", "category": "Special", "accuracy": 1} if move._id in {"recharge", "fight"} else GenData.from_gen(move._gen).moves[move._id]
        """if dynamax or pokemon.is_dynamaxed:
            if MoveCategory[entry["category"].upper()] == MoveCategory.STATUS:
                targets = [0]
            else:
                targets = [1, 2]
        el"""
        if "nonGhostTarget" in entry and (
            PokemonType.GHOST not in pokemon.types
        ):  # fixing target for Curse
            targets = [0]
        elif move._id == "pollenpuff" and Effect.HEAL_BLOCK in pokemon.effects:
            targets = [1, 2]
        elif (
            move._id == "terastarstorm"
            and not pokemon.fainted
            and pokemon.is_terastallized
            and pokemon.tera_type == PokemonType.STELLAR
        ):
            targets = [0]
        else:
            self.targetDict[self.targetDictKeyList[0]]=[ally_position]
            self.targetDict[self.targetDictKeyList[1]]=[ally_position,self_position]
            self.targetDict[self.targetDictKeyList[9]]=[ally_position,1,2]
            self.targetDict[self.targetDictKeyList[11]]=[ally_position,1,2]
            targets=self.targetDict[self.deduce_move_target(move,entry)]

        pokemon_ids = set(battle._opponent_active_pokemon.keys())
        pokemon_ids.update(battle._active_pokemon.keys())
        if battle._player_role == "p1":
            opp_role= "p2"
        else:
            opp_role= "p1"
        targets_to_keep = {
            {
                f"{battle._player_role}a": -1,
                f"{battle._player_role}b": -2,
                f"{opp_role}a": 1,
                f"{opp_role}b": 2,
            }[pokemon_identifier]
            for pokemon_identifier in pokemon_ids
        }
        targets_to_keep.add(0)
        targets = [target for target in targets if target in targets_to_keep]
        #self.pr.disable()
        return targets

    def valid_orders(self,battle):
        orders: List[List[SingleBattleOrder]] = [[], []]
        if battle._wait:
            return [[DefaultBattleOrder()], [DefaultBattleOrder()]]
        active_mon1 = battle._active_pokemon[f"{battle._player_role}a"]
        active_mon2 = battle._active_pokemon[f"{battle._player_role}b"]
        if active_mon1 is None or not active_mon1.active or active_mon1.fainted:
                active_mon1 = None
        if active_mon2 is None or not active_mon2.active or active_mon2.fainted:
            active_mon2 = None
        for i in range(2):
            if any(battle.force_switch) and not battle.force_switch[i]:
                orders[i] += [PassBattleOrder()]
                continue
            if not battle.trapped[i]:
                orders[i] += [
                    SingleBattleOrder(mon) for mon in battle.available_switches[i]
                ]
            if all(battle.force_switch) and len(battle.available_switches[0]) == 1:
                orders[i] += [PassBattleOrder()]
                continue
            active_mon=[active_mon1,active_mon2][i]
            if active_mon is not None and not battle.force_switch[i]:
                orders[i] += [
                    SingleBattleOrder(move, move_target=target)
                    for move in battle.available_moves[i]
                    for target in self.get_possible_showdown_targets(battle, move, active_mon)
                ]
                if battle.can_tera[i]:
                    orders[i] += [
                            SingleBattleOrder(move, move_target=target, terastallize=True)
                            for move in battle.available_moves[i]
                            for target in self.get_possible_showdown_targets(
                                battle, move, active_mon
                            )
                        ]
            if not orders[i]:
                orders[i] += [PassBattleOrder()]
        return orders

    def get_action_mask_individual(battle: DoubleBattle, pos: int):
        return None    

    def get_action_mask(self, battle: AbstractBattle):
        if(battle.won):
            return
        #Initial action masking for gen 9, removing other gimmicks
        action_mask=[0,]*107
        action_mask2=[0,]*107

        #orders is in the form of [[Orders],[Orders]] where each nested list is the valid orders for each slot
        #it does not account for invalid moves together (like double tera)
        for i in range(2):
            for order in self.valid_orders(battle)[i]:
                orderNum=DoublesEnv._order_to_action_individual(order=order,battle=battle,fake=self._fake,pos=i)
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

    def embed_move(self, move:Move):
        if(move==None):
            return [
                -1,
                -1,
                0,
                1,
                0,
                0,
                PokemonType.THREE_QUESTION_MARKS.value / 20,
            ]
        
        if(move._id in self.move_dict):
            id=self.move_dict[move._id]
        else: #new move
            id=self.move_dict_index
            #add to dict
            self.move_dict[move._id]=self.move_dict_index
            #increase index
            self.move_dict_index=self.move_dict_index+1
            #save to file now, avoids rewriting later if this run is terminated early
            #Will be slow in first iterations, but cost nothing later once most items are added
            with open(self.moveMapPath,"wb") as moveFile:
                pickle.dump(self.move_dict, moveFile)
        entry = {"pp": 1, "type": "normal", "category": "Special", "accuracy": 1} if move._id in {"recharge", "fight"} else GenData.from_gen(move._gen).moves[move._id]
        return[
            id,
            1.0 if entry["accuracy"] is True else entry["accuracy"]/100,
            move._base_power_override if move._base_power_override is not None else entry.get("basePower", 0),
            move.expected_hits,
            int(move._id in _PROTECT_MOVES),
            entry["priority"]/5,
            PokemonType.from_name(entry["type"]).value/20,
        ]

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
            ]+[
                -1,
                -1,
                0,
                1,
                0,
                0,
                PokemonType.THREE_QUESTION_MARKS.value / 20,
            ]+[
                -1,
                -1,
                0,
                1,
                0,
                0,
                PokemonType.THREE_QUESTION_MARKS.value / 20,
            ]+[
                -1,
                -1,
                0,
                1,
                0,
                0,
                PokemonType.THREE_QUESTION_MARKS.value / 20,
            ]+[
                -1,
                -1,
                0,
                1,
                0,
                0,
                PokemonType.THREE_QUESTION_MARKS.value / 20,
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
        moveList=list(pokemon.base_moves.values())
        for i in range(4):
            if(i<len(moveList)):
                toReturn = toReturn+self.embed_move(moveList[i])
            else:
                toReturn=toReturn+self.embed_move(None)

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

        if(battle._player_role=="p1"):
            opp_role="p2"
        else:
            opp_role="p1"
        opp_active_mon=battle._opponent_active_pokemon[f"{opp_role}a"]
        if opp_active_mon is None or not opp_active_mon.active or opp_active_mon.fainted:
            opp_active_mon = None
        opp_active_mon2=battle._opponent_active_pokemon[f"{opp_role}b"]
        if opp_active_mon2 is None or not opp_active_mon2.active or opp_active_mon2.fainted:
            opp_active_mon2 = None
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
            
            
            
            if opp_active_mon is not None:
                for active_pokemon in [opp_active_mon,opp_active_mon2]:
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
            if opp_active_mon is not None:
                for active_pokemon in [opp_active_mon,opp_active_mon2]:
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

        action_mask=self.get_action_mask(battle)

        if self.render_mode == "human":
            self.render()

        wait=False
        if(battle.wait):
            wait=True

        toReturn={"observations":np.float32(final_vector),"action_mask":action_mask, "wait":wait}

        return toReturn



    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=15.0, hp_value=10, victory_value=100.0,status_value=5
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
