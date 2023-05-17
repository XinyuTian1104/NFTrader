from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import torch
from gymnasium.utils import seeding

from core.environment.data_loader import DataLoader


class Environment(gym.Env):
    def __init__(self, initial_usd=1024*100, train=False) -> None:

        # environment variables
        self.usd_wallet = initial_usd  # initial number of cash
        self.nft_wallet = 0  # initial number of nft

        # initialize data loader
        self.data_loader = DataLoader(train=train)
        self.current_collection_id = 0
        self.current_timestep = 0

        # define observation space
        # observation space is a dictionary with the following keys:
        #   - 'image': image of the collection, with shape (1, 3, 224, 224)
        #   - 'description': description of the collection, with shape (1, text)
        #   - 'ts_feature': time serise (1-16 rows) of the collection, with shape (1, 16, 5)
        #   - 'ts': time series, at the current timestep of the collection, with shape (5, )
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=255, shape=(1, 3, 224, 224), dtype=np.float32),
            'description': gym.spaces.Text(max_length=1024 * 1024),
            'ts_feature': gym.spaces.Box(low=0, high=255, shape=(1, 16, 5), dtype=np.float32),
            'ts': gym.spaces.Box(low=0, high=255, shape=(5, ), dtype=np.float32),
        })

        # define action space
        # each action is a tuple of (action, percentage), where
        #  - action is one of 'buy', 'sell', 'hold'
        #    0 - buy
        #    1 - sell
        #    2 - hold
        #  - percentage is a float between 0 and 1
        self.action_space = gym.spaces.Tuple((
            gym.spaces.Discrete(3),
            gym.spaces.Box(low=0, high=1, shape=(1, ), dtype=np.float32),
        ))

    def _get_obs(self):
        text, image, time_series = self.data_loader.load_collection_features(
            self.current_collection_id)

        pass

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        return super().step(action)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        return super().reset(seed=seed, options=options)

    def render(self):
        return
