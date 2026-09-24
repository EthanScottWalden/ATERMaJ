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
    