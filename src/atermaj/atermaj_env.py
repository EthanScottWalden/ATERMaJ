from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any
from itertools import combinations

from jackdaw.env.action_space import ActionType, FactoredAction, get_consumable_target_info
from jackdaw.env import BalatroGymnasiumEnv
from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.game_interface import GameAdapter
from jackdaw.env.game_spec import GameActionMask, GameObservation, GameSpec
from jackdaw.env.observation import encode_observation
from jackdaw.engine.card import Card
from jackdaw.engine.blind import Blind

import numpy as np

MAX_ACTIONS: int = 500
CARD_COMBO_BUDGET: int = 200

# Pre-compute entity layout from spec
_SPEC = balatro_game_spec()
_ENTITY_INFO: list[tuple[str, int, int]] = [
    (et.name, et.max_count, et.feature_dim) for et in _SPEC.entity_types
]
# Action types that take no targets at all
_SIMPLE_TYPES: frozenset[int] = frozenset(
    i
    for i, at in enumerate(_SPEC.action_types)
    if not at.needs_entity_target and not at.needs_card_select
)
# Entity-only action types (entity target, no card select) — excludes UseConsumable
_ENTITY_ONLY_TYPES: frozenset[int] = frozenset(
    i
    for i, at in enumerate(_SPEC.action_types)
    if at.needs_entity_target and not at.needs_card_select
)
# Card-only action types (card select, no entity) — PlayHand, Discard
_CARD_ONLY_TYPES: frozenset[int] = frozenset(
    i
    for i, at in enumerate(_SPEC.action_types)
    if at.needs_card_select and not at.needs_entity_target
)

def _subsample(items: list[Any], budget: int, rng: np.random.Generator) -> list[Any]:
    """Return *items* if within budget, else a random subsample."""
    if len(items) <= budget:
        return items
    indices = rng.choice(len(items), size=budget, replace=False)
    return [items[i] for i in sorted(indices)]

def _card_combos(
    legal_cards: np.ndarray,
    min_select: int,
    max_select: int,
) -> list[tuple[int, ...]]:
    """Enumerate all card-index combinations within the selection range."""
    upper = min(len(legal_cards), max_select)
    lower = min(min_select, upper)
    result: list[tuple[int, ...]] = []
    for k in range(lower, upper + 1):
        result.extend(tuple(int(c) for c in combo) for combo in combinations(legal_cards, k))
    return result

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
        self._rounds_spent_with_joker: dict[str, int] = dict()
        self._consumable_usages: dict[str, int] = dict()

        # session variables are never reset, and describe the whole training session.
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
            prev_consumable_usage = self._consumable_usages.get(consumable_name, 0)
            self._consumable_usages[consumable_name] = prev_consumable_usage + 1

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
                prev_rounds_spent = self._rounds_spent_with_joker.get(joker.ability['name'], 0)
                self._rounds_spent_with_joker[joker.ability['name']] = prev_rounds_spent + 1

        # print("Obtained:", self._all_jokers_obtained)
        # print("Rounds w/:", self._rounds_spent_with_joker)

        self._prev_round = round_num
        self._prev_ante = ante
        self._prev_chips = chips
        self._prev_consumables = list(consumables)
        self._prev_pack_cards = list(pack_cards)
        self._episode_max_ante = max(self._episode_max_ante, ante)
        self._episode_max_round = max(self._episode_max_round, round_num)

    def _enumerate_actions(
            self, mask: GameActionMask, info: dict[str, Any]
        ) -> list[FactoredAction]:
        actions: list[FactoredAction] = []
        type_mask = mask.type_mask
        gs: dict[str, Any] = info.get("raw_state", {})
        blind: Blind | None = gs.get("blind")
        hand: list[Card] = gs.get("hand", [])

        for t in range(len(type_mask)):
            if not type_mask[t]:
                continue

            # --- Simple (no targets) ---
            if t in _SIMPLE_TYPES:
                actions.append(FactoredAction(action_type=t))

            # --- Entity-only ---
            elif t in _ENTITY_ONLY_TYPES:
                if t in mask.entity_masks:
                    legal = np.nonzero(mask.entity_masks[t])[0]
                    for idx in legal:
                        actions.append(FactoredAction(action_type=t, entity_target=int(idx)))

            # --- Card-only (PlayHand / Discard) ---
            elif t in _CARD_ONLY_TYPES:
                legal_cards = np.nonzero(mask.card_mask)[0]
                combos = _card_combos(legal_cards, mask.min_card_select, mask.max_card_select)

                if blind is not None and blind.key == "bl_final_bell" and not blind.disabled:
                    forced_indices = [
                        i for i, card in enumerate(hand)
                        if card.ability.get("forced_selection") is True
                    ]

                    if forced_indices:
                        combos = [combo for combo in combos if all(idx in combo for idx in forced_indices)]

                combos = _subsample(combos, CARD_COMBO_BUDGET, self._rng)
                for combo in combos:
                    actions.append(FactoredAction(action_type=t, card_target=combo))

            # --- UseConsumable (entity + optional cards) ---
            elif t == ActionType.UseConsumable:
                if t in mask.entity_masks:
                    legal_entities = np.nonzero(mask.entity_masks[t])[0]
                    raw_consumables = info.get("raw_state", {}).get("consumables", [])
                    for c_idx in legal_entities:
                        c_idx_int = int(c_idx)
                        if c_idx_int < len(raw_consumables):
                            card = raw_consumables[c_idx_int]
                            min_cards, max_cards, needs = get_consumable_target_info(card)
                            if needs:
                                legal_cards = np.nonzero(mask.card_mask)[0]
                                combos = _card_combos(legal_cards, min_cards, max_cards)
                                combos = _subsample(combos, CARD_COMBO_BUDGET, self._rng)
                                for combo in combos:
                                    actions.append(
                                        FactoredAction(
                                            action_type=t,
                                            entity_target=c_idx_int,
                                            card_target=combo,
                                        )
                                    )
                            else:
                                actions.append(
                                    FactoredAction(action_type=t, entity_target=c_idx_int)
                                )

        # Subsample to budget if needed
        if len(actions) > MAX_ACTIONS:
            actions = _subsample(actions, MAX_ACTIONS, self._rng)

        return actions


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
            step_info["episode/joker_rounds"] = dict(self._rounds_spent_with_joker)
            step_info["episode/consumable_usages"] = dict(self._consumable_usages)

            step_info["session/max_ante_reached"] = self._session_max_ante
            step_info["session/max_rounds_beaten"] = self._session_max_round
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
        self._rounds_spent_with_joker: dict[str, int] = dict()
        self._consumable_usages: dict[str, int] = dict()
        self._action_table = self._enumerate_actions(game_mask, info)
        obs = self._build_obs(game_obs)
        return obs, {"action_mask": self.action_masks()}