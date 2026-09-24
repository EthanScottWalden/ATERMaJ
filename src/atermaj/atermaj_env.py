"""
TODO:
- Investigate how to log joker usage and/or consumable usage. Find out what the heck is in raw_state and if this contains consumable names or ids.
    - Once this is done, figure out a GOOD way to log it, or a GOOD way to store info on it until the end. This will involve both logging callbacks and internal info management in the env.
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
          self.printed_times = 0
          super().__init__(adapter_factory, back_keys, stakes, max_steps, seed_prefix, reward_shaping)

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
            obs, reward, terminated, truncated, step_info = super().step(action)
            
            return obs, reward, terminated, truncated, step_info
    