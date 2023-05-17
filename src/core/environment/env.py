from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import torch
from gymnasium.utils import seeding

from core.environment.data_loader import DataLoader


class Environment(gym.Env):
    def __init__(self, initial_usd=1024*1000, train=False) -> None:

        # environment variables
        self.usd_wallet = initial_usd  # initial number of cash
        self.nft_wallet = 0  # initial number of nft
        self.initial_usd = initial_usd  # initial number of cash

        # initialize data loader
        self.data_loader = DataLoader(train=train)
        self.current_collection_id = 0
        self.current_price_usd = np.inf

        self.is_train = train

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
            'nft_wallet': gym.spaces.Box(low=0, high=np.inf, shape=(1, ), dtype=np.float32),
            'usd_wallet': gym.spaces.Box(low=0, high=np.inf, shape=(1, ), dtype=np.float32),
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

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        # alter the state of the environment
        #  - action is one of 'buy', 'sell', 'hold'
        #    0 - buy
        #    1 - sell
        #    2 - hold
        if action[0] == 0:
            # print('buy')
            # buy, the fraction of total cash to buy
            buy_cash = self.usd_wallet * action[1]
            nft_amount = np.floor(buy_cash / self.current_price_usd)
            buy_cash = nft_amount * self.current_price_usd
            # print('buy nft amount:', nft_amount)
            self.usd_wallet -= buy_cash
            self.nft_wallet += nft_amount

        if action[0] == 1:
            # print('sell')
            # sell, the fraction of total nft to sell
            sell_nft = np.floor(self.nft_wallet * action[1])

            print("sell nft:", sell_nft)
            print("current price:", self.current_price_usd)
            
            

            sell_cash = sell_nft * self.current_price_usd
            self.usd_wallet += sell_cash
            self.nft_wallet -= sell_nft
            # print('sell nft amount:', sell_nft)

        if action[0] == 2:
            # print('hold')
            # hold, do nothing
            pass

        observation, terminated = self._get_obs()
        info = self._get_info()
        truncated = False
        reward = self._get_reward(terminated=terminated)

        return observation, reward, terminated, truncated, info

    def _get_reward(self, terminated):
        if terminated:
            return self.usd_wallet + self.nft_wallet * self.current_price_usd - self.initial_usd
        return 0

    def _get_info(self):
        payload = {
            'usd_wallet': self.usd_wallet,
            'nft_wallet': self.nft_wallet,
            'current_price_usd': self.current_price_usd,
        }

        return payload

    def _get_obs(self):
        text, image, ts_feature = self.data_loader.load_collection_features(
            self.current_collection_id)

        ts, collection_end = self.data_loader.load_time_series(
            collection_id=self.current_collection_id)

        done = False

        print('collection id:', self.current_collection_id)

        if collection_end:
            done = True
            self.current_collection_id += 1

            # environment variables
            self.usd_wallet = self.initial_usd  # initial number of cash
            self.nft_wallet = 0  # initial number of nft

            if self.current_collection_id >= len(self.data_loader):
                self.current_collection_id = 0

        self.current_price_usd = ts[0, -1, 1].item()

        # print('current price:', self.current_price_usd)

        observation = {
            'image': image,
            'description': text,
            'ts_feature': ts_feature,
            'ts': ts,
            'nft_wallet': self.nft_wallet,
            'usd_wallet': self.usd_wallet,
        }

        return observation, done

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        # environment variables
        self.usd_wallet = self.initial_usd  # initial number of cash
        self.nft_wallet = 0  # initial number of nft

        # initialize data loader
        self.data_loader = DataLoader(train=self.is_train)
        self.current_collection_id = 0
        self.current_price_usd = np.inf

        observation, _ = self._get_obs()
        info = self._get_info()

        return observation, info

    def render(self):
        return
