from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from jackdaw.env.action_space import ActionType, FactoredAction
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
        self._prev_consumables: list = list()
        self._prev_pack_cards: list = list()

        # session variables are never reset, and describe the whole training session.
        self._session_rounds_spent_with_joker: dict[str, int] = dict()
        self._session_consumable_usages: dict[str, int] = dict()
        self._session_max_ante: int = 1
        self._session_max_round: int = 0

        super().__init__(adapter_factory, back_keys, stakes, max_steps, seed_prefix, reward_shaping)

    def _compute_reward(self, info: dict[str, Any], terminated: bool, truncated: bool, factored: FactoredAction) -> float:
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
        ante_scale = self._prev_ante / 8.0

        earnings = gs.get("round_earnings")

        # Score progress within a blind: chips gained toward target
        blind = gs.get("blind")
        blind_target = getattr(blind, "chips", 0) if blind is not None else 0
        if blind_target > 0 and chips > self._prev_chips:
            chip_delta = chips - self._prev_chips
            reward += 0.02 * min(chip_delta / blind_target, 1.0)
            # Blind beaten: round increased → +0.15 * ante_scale
            if self._prev_chips < blind_target <= chips:
                reward += 0.15 * ante_scale

        # Boss blind beaten (ante increased) → extra +0.1 * ante_scale
        if ante > self._prev_ante:
            reward += 0.1 * ante_scale

        # Cash out: reward for unused hands and interest
        if earnings is not None and factored.action_type == ActionType.CashOut:
            # Efficient clear: hands remaining bonus. this does not account for the green deck!
            hands_left = earnings.unused_hands_bonus
            reward += 0.005 * hands_left
            # Bonus for each interest dollar.
            interest = earnings.interest
            reward += 0.015 * interest
            
        # Terminal
        if terminated or truncated:
            reward += 0.5 if self._inner.episode_won else -0.2

        return reward

    def _update_trackers(self, info, factored: FactoredAction) -> None:
        # Update trackers
        gs: dict[str, Any] = info.get("raw_state", {})

        def update_consumable_usage(consumable_name: str):
            prev_consumable_usage = self._session_consumable_usages.get(consumable_name, 0)
            self._session_consumable_usages[consumable_name] = prev_consumable_usage + 1

        if factored.action_type == ActionType.UseConsumable:
            consumable_name = self._prev_consumables[factored.entity_target].ability['name']
            update_consumable_usage(consumable_name=consumable_name)
        elif factored.action_type == ActionType.PickPackCard and not gs['pack_type'] in ('Standard', 'Buffoon'):
            consumable_name = self._prev_pack_cards[factored.entity_target].ability['name']
            update_consumable_usage(consumable_name=consumable_name)

        ante = gs.get("round_resets", {}).get("ante", 1)
        round_num = gs.get("round", 0)
        chips = gs.get("chips", 0)
        jokers = gs.get("jokers", [])
        consumables = gs.get("consumables", [])
        pack_cards = gs.get("pack_cards", [])

        # if (random.randint(1,100) == 1):
        #     green_joker = Card()
        #     green_joker.set_ability(center="j_green_joker")

        #     jokers.append(green_joker)

        if (self._prev_round < round_num and len(jokers) > 0):
            for joker in jokers:
                prev_rounds_spent = self._session_rounds_spent_with_joker.get(joker.ability['name'], 0)
                self._session_rounds_spent_with_joker[joker.ability['name']] = prev_rounds_spent + 1

        # print("Obtained:", self._all_jokers_obtained)
        # print("Rounds w/:", self._rounds_spent_with_joker)

        self._prev_round = round_num
        self._prev_ante = ante
        self._prev_chips = chips
        self._prev_consumables = list(consumables)
        self._prev_pack_cards = list(pack_cards)
        self._episode_max_ante = max(self._episode_max_ante, ante)
        self._episode_max_round = max(self._episode_max_round, round_num)

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        factored = self._action_table[action]
        game_obs, terminated, truncated, game_mask, info = self._inner.step(factored)

        reward = self._compute_reward(info, terminated, truncated, factored)
        self._update_trackers(info, factored)

        # Rebuild action table for next step
        if not (terminated or truncated):
            self._action_table = self._enumerate_actions(game_mask, info)
        else:
            self._action_table = []

        obs = self._build_obs(game_obs)
        step_info: dict[str, Any] = {"action_mask": self.action_masks()}
        if terminated or truncated:
            gs: dict[str, Any] = info.get("raw_state", {})

            step_info["episode/ante_reached"] = self._episode_max_ante
            step_info["episode/rounds_beaten"] = self._episode_max_round
            step_info["episode/won"] = self._inner.episode_won
            step_info["episode/deck"] = gs.get("selected_back_key")
            step_info["episode/stake"] = gs.get("stake")

            step_info["session/max_ante_reached"] = self._session_max_ante
            step_info["session/max_rounds_beaten"] = self._session_max_round
            step_info["session/joker_rounds"] = dict(self._session_rounds_spent_with_joker)
            step_info["session/consumable_usages"] = dict(self._session_consumable_usages)
        return obs, reward, terminated, truncated, step_info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = str(seed)

        game_obs, game_mask, info = self._inner.reset(**kwargs)
        self._prev_ante = self._inner.episode_ante
        self._prev_round = 0
        self._prev_chips = 0
        self._prev_pack_cards = list()
        self._prev_consumables = list()
        self._episode_max_ante = 1
        self._episode_max_round = 0
        self._action_table = self._enumerate_actions(game_mask, info)
        obs = self._build_obs(game_obs)
        return obs, {"action_mask": self.action_masks()}