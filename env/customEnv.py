from poke_env.environment.doubles_env import DoublesEnv

from typing import Optional, Union

import numpy as np
from gymnasium.spaces import Discrete

from poke_env.battle import AbstractBattle
from poke_env.battle.double_battle import DoubleBattle
from poke_env.environment.env import ObsType
from poke_env.data import GenData

from poke_env.ps_client import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    ServerConfiguration,
)
from poke_env.teambuilder import Teambuilder

import webbrowser
import hashlib

from time import sleep

# --- Observation encoding helpers ---

_WEATHER_MAP = {
    "SUNNYDAY": 1/7, "RAINDANCE": 2/7, "SANDSTORM": 3/7,
    "HAIL": 4/7, "SNOW": 4/7, "SNOWSTORM": 4/7,
    "DESOLATELAND": 5/7, "PRIMORDIALSEA": 6/7, "DELTASTREAM": 7/7,
}

_TERRAIN_MAP = {
    "ELECTRIC_TERRAIN": 1/4, "GRASSY_TERRAIN": 2/4,
    "MISTY_TERRAIN": 3/4, "PSYCHIC_TERRAIN": 4/4,
}

_STATUS_MAP = {
    "BRN": 1/6, "FRZ": 2/6, "PAR": 3/6,
    "PSN": 4/6, "SLP": 5/6, "TOX": 6/6,
}

_HASH_NORM = 4096


def _hp_fraction(mon) -> float:
    """Returns current HP fraction (0.0 if fainted or not present)."""
    if mon is None or mon.fainted:
        return 0.0
    return float(mon.current_hp_fraction)


def _encode_status(mon) -> float:
    """Encodes a Pokémon's status condition as a normalized float in [0, 1]."""
    if mon is None or mon.status is None:
        return 0.0
    return _STATUS_MAP.get(mon.status.name, 0.0)


def _encode_str(s) -> float:
    """Deterministically hash-encodes a string (e.g. ability or item name) to [0, 1]."""
    if not s:
        return 0.0
    h = int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % _HASH_NORM
    return h / _HASH_NORM


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
        self.action_spaces = {
            agent: Discrete(107*107) for agent in self.possible_agents
        }
        self.render_browser_open=False
        self.render_mode=render_mode
        self._type_chart = GenData.from_format(battle_format).type_chart
        self._last_rendered_turn = -1
    
    def reset(self,seed=None,options=None):
        self.render_browser_open = False
        self._last_rendered_turn = -1
        return super().reset(seed=seed,options=options)

    def get_mask(self, battle: AbstractBattle):
        #Initial action masking for gen 9, removing other gimmicks
        action_mask=[0,]*107
        action_mask2=[0,]*107

        #orders is in the form of [[Orders],[Orders]] where each nested list is the valid orders for each slot
        #it does not account for invalid moves together (like double tera)
        for i in range(2):
            for order in battle.valid_orders[i]:
                try:
                    orderNum=DoublesEnv._order_to_action_individual(order=order,battle=battle,fake=False,pos=i)
                except ValueError:
                    continue
                if(i==0):
                    action_mask[orderNum]=1
                else:
                    action_mask2[orderNum]=1

        #DoubleBattleOrder.join_orders()

        #Converting the masks to the proper type
        action_mask=np.array(action_mask,dtype=np.int8)
        action_mask2=np.array(action_mask2,dtype=np.int8)
        #Combine into a single dimensional array of all combinations
        action_mask_combined=np.sum(np.array(np.meshgrid(action_mask, action_mask2)).T.reshape(-1, 2),axis=1)
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

    
    def embed_battle(self, battle: AbstractBattle) -> tuple[ObsType,dict[int:int]]:
        """
        Returns the embedding of the current battle state in a format compatible with
        the Gymnasium API.

        :param battle: The current battle state.
        :type battle: DoubleBattle

        :return: The embedding of the current battle state.
        """
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
                    if(active_pokemon !=None):
                        moves_dmg_multiplier[i] = move.type.damage_multiplier(
                            active_pokemon.type_1,
                            active_pokemon.type_2,
                            type_chart=self._type_chart,
                        )
        for j, move in enumerate(battle.available_moves[1]):
            moves_base_power[4+j] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if battle.opponent_active_pokemon[0] is not None:
                for active_pokemon in battle.opponent_active_pokemon:
                    if(active_pokemon!=None):
                        moves_dmg_multiplier[4+j] = move.type.damage_multiplier(
                            active_pokemon.type_1,
                            active_pokemon.type_2,
                            type_chart=self._type_chart,
                        )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # --- damage on each team ---
        # HP fraction for each active slot (own0, own1, opp0, opp1)
        hp_fractions = np.array([
            _hp_fraction(battle.active_pokemon[0]),
            _hp_fraction(battle.active_pokemon[1]),
            _hp_fraction(battle.opponent_active_pokemon[0]),
            _hp_fraction(battle.opponent_active_pokemon[1]),
        ], dtype=np.float32)

        # --- field status (weather, terrain, entry hazards) ---
        weather_val = 0.0
        for w in battle.weather:
            weather_val = _WEATHER_MAP.get(w.name, 0.0)
            break

        terrain_val = 0.0
        for f in battle.fields:
            v = _TERRAIN_MAP.get(f.name, 0.0)
            if v > 0.0:
                terrain_val = v
                break

        own_sr, own_spikes = 0.0, 0.0
        for sc, count in battle.side_conditions.items():
            if sc.name == "STEALTH_ROCK":
                own_sr = 1.0
            elif sc.name == "SPIKES":
                own_spikes = min(count, 3) / 3.0

        opp_sr, opp_spikes = 0.0, 0.0
        for sc, count in battle.opponent_side_conditions.items():
            if sc.name == "STEALTH_ROCK":
                opp_sr = 1.0
            elif sc.name == "SPIKES":
                opp_spikes = min(count, 3) / 3.0

        field_status = np.array(
            [weather_val, terrain_val, own_sr, own_spikes, opp_sr, opp_spikes],
            dtype=np.float32,
        )

        # --- active mon status conditions ---
        active_status = np.array([
            _encode_status(battle.active_pokemon[0]),
            _encode_status(battle.active_pokemon[1]),
            _encode_status(battle.opponent_active_pokemon[0]),
            _encode_status(battle.opponent_active_pokemon[1]),
        ], dtype=np.float32)

        # --- other mon (bench) status conditions ---
        active_ids = {id(m) for m in battle.active_pokemon if m is not None}
        own_bench = [m for m in battle.team.values() if id(m) not in active_ids and not m.fainted]
        opp_active_ids = {id(m) for m in battle.opponent_active_pokemon if m is not None}
        opp_bench = [m for m in battle.opponent_team.values() if id(m) not in opp_active_ids and not m.fainted]

        own_bench_status = [_encode_status(m) for m in own_bench[:4]]
        while len(own_bench_status) < 4:
            own_bench_status.append(0.0)
        opp_bench_status = [_encode_status(m) for m in opp_bench[:4]]
        while len(opp_bench_status) < 4:
            opp_bench_status.append(0.0)
        bench_status = np.array(own_bench_status + opp_bench_status, dtype=np.float32)

        # --- active mon abilities (hash-encoded) ---
        active_abilities = np.array([
            _encode_str(battle.active_pokemon[0].ability if battle.active_pokemon[0] else None),
            _encode_str(battle.active_pokemon[1].ability if battle.active_pokemon[1] else None),
            _encode_str(battle.opponent_active_pokemon[0].ability if battle.opponent_active_pokemon[0] else None),
            _encode_str(battle.opponent_active_pokemon[1].ability if battle.opponent_active_pokemon[1] else None),
        ], dtype=np.float32)

        # --- active mon items (hash-encoded) ---
        active_items = np.array([
            _encode_str(battle.active_pokemon[0].item if battle.active_pokemon[0] else None),
            _encode_str(battle.active_pokemon[1].item if battle.active_pokemon[1] else None),
            _encode_str(battle.opponent_active_pokemon[0].item if battle.opponent_active_pokemon[0] else None),
            _encode_str(battle.opponent_active_pokemon[1].item if battle.opponent_active_pokemon[1] else None),
        ], dtype=np.float32)

        # Final vector — 48 components total:
        #   [0:8]   move base powers (slot0: 0-3, slot1: 4-7)
        #   [8:16]  type effectiveness (slot0: 8-11, slot1: 12-15)
        #   [16:18] fainted fractions (own, opp)
        #   [18:22] active HP fractions (own0, own1, opp0, opp1)
        #   [22:28] field status (weather, terrain, own_SR, own_spikes, opp_SR, opp_spikes)
        #   [28:32] active mon status (own0, own1, opp0, opp1)
        #   [32:40] bench status (own x4, opp x4)
        #   [40:44] active mon abilities
        #   [44:48] active mon items
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
                hp_fractions,
                field_status,
                active_status,
                bench_status,
                active_abilities,
                active_items,
            ]
        )

        action_mask=self.get_mask(battle)

        if self.render_mode == "human":
            turn = getattr(battle, 'turn', -1)
            if turn != self._last_rendered_turn:
                self._last_rendered_turn = turn
                self.render()

        return {"observations":np.float32(final_vector),"action_mask":action_mask}



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
                try:
                    webbrowser.open(url, new=0, autoraise=True)
                    sleep(3)
                except Exception:
                    pass
                self.render_browser_open = True

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
                end="\n" if self.battle1.finished else "\r",
            )
