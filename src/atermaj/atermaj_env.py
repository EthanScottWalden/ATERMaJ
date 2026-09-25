"""
TODO:
- Investigate how to log joker usage and/or consumable usage. Find out what the heck is in raw_state and if this contains consumable names or ids.
    - Once this is done, figure out a GOOD way to log it, or a GOOD way to store info on it until the end. This will involve both logging callbacks and internal info management in the env.

    - TODO: Seemingly the "center_key" attribute of a card tells us what kind of card it is, including consumables! We can log this info when a move targets a consumable.


- Mess around with how rewards are computed, if I so desire.
- Ask about what to do if simulator throws an exception due to bugged seed.
- Ask about what to do about the fact that the 500 vector used in gymnasium wrapper doesn't represent all realistic game states.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from jackdaw.env.action_space import (
    ActionMask,
    FactoredAction,
    factored_to_engine_action,
    get_action_mask,
)
from jackdaw.env import BalatroGymnasiumEnv
from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.game_interface import GameAdapter
from jackdaw.env.game_spec import GameActionMask, GameObservation, GameSpec
from jackdaw.env.observation import encode_observation
from jackdaw.engine.card import Card

import numpy as np


class AtermajEnv(BalatroGymnasiumEnv):
    """This is a wrapper class around BalatroGymnasiumEnv created for the sole purpose of overriding methods to define custom behavior.
    
    Overriden methods: (none)"""
    def __init__(
            self,
            adapter_factory: Callable[[], GameAdapter],
            back_keys: list[str] | None = None,
            stakes: list[int] | None = None,
            max_steps: int = 10_000,
            seed_prefix: str = "TRAIN",
            reward_shaping: bool = False,
        ) -> None:
        self._all_jokers_obtained = set()
        self._rounds_spent_with_joker = dict()

        super().__init__(adapter_factory, back_keys, stakes, max_steps, seed_prefix, reward_shaping)

    def _compute_reward(self, info: dict[str, Any], terminated: bool, truncated: bool) -> float:
        """Compute step reward from game state deltas."""
        if not self._reward_shaping:
            if terminated or truncated:
                return 1.0 if self._inner.episode_won else -1.0
            return 0.0

        gs: dict[str, Any] = info.get("raw_state", {})
        phase = gs.get("phase")

        # Step cost — discourages stalling; doubled in shop phase
        reward = -0.002 if phase == "shop" else -0.001

        ante = gs.get("round_resets", {}).get("ante", 1)
        round_num = gs.get("round", 0)
        chips = gs.get("chips", 0)
        ante_scale = ante / 8.0

        # 1. Blind beaten: round increased → +0.15 * ante_scale
        if round_num > self._prev_round:
            reward += 0.15 * ante_scale
            # 2. Boss blind beaten (ante increased) → extra +0.1 * ante_scale
            if ante > self._prev_ante:
                reward += 0.1 * ante_scale
            # 3. Efficient clear: hands remaining bonus
            hands_left = gs.get("current_round", {}).get("hands_left", 0)
            reward += 0.01 * hands_left

        # 4. Score progress within a blind: chips gained toward target
        blind = gs.get("blind")
        blind_target = getattr(blind, "chips", 0) if blind is not None else 0
        if blind_target > 0 and chips > self._prev_chips:
            chip_delta = chips - self._prev_chips
            reward += 0.02 * min(chip_delta / blind_target, 1.0)

        # 5. Terminal
        if terminated or truncated:
            reward += 0.5 if self._inner.episode_won else -0.2

        return reward

    def _update_trackers(self, info) -> None:
        # Update trackers
        gs: dict[str, Any] = info.get("raw_state", {})

        ante = gs.get("round_resets", {}).get("ante", 1)
        round_num = gs.get("round", 0)
        chips = gs.get("chips", 0)
        jokers = gs.get("jokers", [])

        # if (random.randint(1,100) == 1):
        #     green_joker = Card()
        #     green_joker.set_ability(center="j_green_joker")

        #     jokers.append(green_joker)

        self._all_jokers_obtained.update([joker.ability['name'] for joker in jokers])

        if (self._prev_round < round_num and len(jokers) > 0):
            for joker in jokers:
                prev_rounds_spent = self._rounds_spent_with_joker.get(joker.ability['name'], 0)
                self._rounds_spent_with_joker[joker.ability['name']] = prev_rounds_spent + 1

        # print("Obtained:", self._all_jokers_obtained)
        # print("Rounds w/:", self._rounds_spent_with_joker)

        self._prev_round = round_num
        self._prev_ante = ante
        self._prev_chips = chips
        self._episode_max_ante = max(self._episode_max_ante, ante)
        self._episode_max_round = max(self._episode_max_round, round_num)

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        factored = self._action_table[action]
        game_obs, terminated, truncated, game_mask, info = self._inner.step(factored)

        reward = self._compute_reward(info, terminated, truncated)
        self._update_trackers(info)

        # Rebuild action table for next step
        if not (terminated or truncated):
            self._action_table = self._enumerate_actions(game_mask, info)
        else:
            self._action_table = []

        obs = self._build_obs(game_obs)
        step_info: dict[str, Any] = {"action_mask": self.action_masks()}
        if terminated or truncated:
            step_info["balatro/ante_reached"] = self._episode_max_ante
            step_info["balatro/rounds_beaten"] = self._episode_max_round
            step_info["balatro/won"] = self._inner.episode_won
        return obs, reward, terminated, truncated, step_info
