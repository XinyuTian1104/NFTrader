from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.utils import seeding


class Environment(gym.Env):
    def __init__(self, initial_usd=1024*100) -> None:

        # environment variables
        self.usd_wallet = initial_usd  # initial number of cash
        self.nft_wallet = 0  # initial number of nft

        pass

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        return super().step(action)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        return super().reset(seed=seed, options=options)

    def render(self):
        return
