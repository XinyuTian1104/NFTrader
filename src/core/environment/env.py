from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import torch
from gymnasium.utils import seeding

from core.environment.data_loader import DataLoader


VECTOR_LENGTH = 16 * 5 + 3 * 224 * 224 + 16 * 5 + 1 + 1


class Environment(gym.Env):
    def __init__(self, initial_usd=1024*1000, train=True) -> None:

        # environment variables
        self.usd_wallet = initial_usd  # initial number of cash
        self.nft_wallet = 0  # initial number of nft
        self.initial_usd = initial_usd  # initial number of cash

        # initialize data loader
        self.data_loader = DataLoader(train=train)
        self.current_collection_id = 0
        self.current_price_usd = np.inf
        self.counter = 0

        self.is_train = train

        # define observation space
        # observation space is a dictionary with the following keys:
        #   - 'image': image of the collection, with shape (1, 3, 224, 224)
        #   - 'description': description of the collection, with shape (1, text)
        #   - 'ts_feature': time serise (1-16 rows) of the collection, with shape (1, 16, 5)
        #   - 'ts': time series, at the current timestep of the collection, with shape (5, )
        # self.observation_space = gym.spaces.Dict({
        #     'image': gym.spaces.Box(low=0, high=255, shape=(1, 3, 224, 224), dtype=np.float32),
        #     'description': gym.spaces.Text(max_length=1024 * 1024),
        #     'ts_feature': gym.spaces.Box(low=0, high=255, shape=(1, 16, 5), dtype=np.float32),
        #     'ts': gym.spaces.Box(low=0, high=255, shape=(5, ), dtype=np.float32),
        #     'nft_wallet': gym.spaces.Box(low=0, high=np.inf, shape=(1, ), dtype=np.float32),
        #     'usd_wallet': gym.spaces.Box(low=0, high=np.inf, shape=(1, ), dtype=np.float32),
        # })

        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(VECTOR_LENGTH, ), dtype=np.float32
        )

        # define action space
        # each action is a tuple of (action, percentage), where
        #  - action is one of 'buy', 'sell', 'hold'
        #    0 - buy
        #    1 - sell
        #    2 - hold
        #  - percentage is a float between 0 and 1
        # self.action_space = gym.spaces.Tuple((
        #     gym.spaces.Discrete(3),
        #     gym.spaces.Box(low=0, high=1, shape=(1, ), dtype=np.float32),
        # ))
        self.action_space = gym.spaces.Discrete(3)

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        # alter the state of the environment
        #  - action is one of 'buy', 'sell', 'hold'
        #    0 - buy
        #    1 - sell
        #    2 - hold
        if action == 0:
            # buy nft, one each time, if there is enough cash
            if self.usd_wallet >= self.current_price_usd:
                self.usd_wallet -= self.current_price_usd
                self.nft_wallet += 1
            # print('buy nft amount:', 1)

        if action == 1:
            # sell all nft, if there is any
            if self.nft_wallet > 0:
                self.usd_wallet += self.current_price_usd * self.nft_wallet
                self.nft_wallet = 0
            # print('sell nft amount:', self.nft_wallet)

        if action == 2:
            # print('hold')
            # hold, do nothing
            pass

        observation, terminated = self._get_obs()
        info = self._get_info()
        truncated = False
        reward = self._get_reward(terminated=terminated)

        self.counter += 1

        self.current_collection_id = self.data_loader.current_collection_id

        if self.counter >= 64:
            terminated = True

        return observation, reward, terminated, truncated, info

    def _get_reward(self, terminated):
        # reward if the agent is able to make profit for long term
        current_value = self.usd_wallet + self.nft_wallet * self.current_price_usd
        initial_value = self.initial_usd

        diff = np.float32((current_value - initial_value) / 8)

        return diff

    def _get_info(self):

        payload = {
            'usd_wallet': self.usd_wallet,
            'nft_wallet': self.nft_wallet,
            'current_price_usd': self.current_price_usd,
            'current_collection_id': self.data_loader.current_collection_id,
        }

        return payload

    def _get_obs(self):
        text, image, ts_feature = self.data_loader.load_collection_features(
            self.current_collection_id)

        ts, collection_end = self.data_loader.load_time_series(
            collection_id=self.current_collection_id)

        done = False

        # print('collection id:', self.current_collection_id)

        if collection_end:
            done = True
            self.current_collection_id += 1

            # environment variables
            self.usd_wallet = self.initial_usd  # initial number of cash
            self.nft_wallet = 0  # initial number of nft

            if self.current_collection_id >= len(self.data_loader):
                self.current_collection_id = 0

        self.current_price_usd = ts[0, -1, 1].item()

        # construct single-vector observation
        # shape of x: (1, VECTOR_LENGTH)
        # structure of x:
        #  - ts_feature: (batch_size, 16, 5)
        #  - image_feature: (batch_size, 3, 224, 224)
        #  - ts_data: (batch_size, 16, 5)
        #  - usd_wallet: (batch_size, 1)
        #  - nft_wallet: (batch_size, 1)
        ts_feature = ts_feature.reshape(-1)
        image = image.reshape(-1)
        ts = ts.reshape(-1)
        usd_wallet = np.array([self.usd_wallet]).reshape(-1)
        nft_wallet = np.array([self.nft_wallet]).reshape(-1)

        observation = np.concatenate(
            (ts_feature, image, ts, usd_wallet, nft_wallet), axis=0).astype(np.float32)

        return observation, done

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        # environment variables
        self.usd_wallet = self.initial_usd  # initial number of cash
        self.nft_wallet = 0  # initial number of nft
        self.counter = 0
        self.current_collection_id += 1
        if self.current_collection_id >= len(self.data_loader):
            self.current_collection_id = 0

        self.data_loader.current_collection_id = self.current_collection_id
        self.data_loader.current_timestep = 0
        self.data_loader.reload_flag = True
        self.data_loader._flush(self.current_collection_id)

        # initialize data loader
        # self.data_loader = DataLoader(train=self.is_train)
        # self.current_collection_id = 0
        # self.current_price_usd = np.inf

        observation, _ = self._get_obs()
        info = self._get_info()

        return observation, info

    def render(self):
        return
