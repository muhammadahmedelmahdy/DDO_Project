import gymnasium
import numpy as np
import pandas as pd
from gymnasium import Env, spaces
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from gymnasium.spaces.utils import flatten, flatten_space
from gymnasium.utils import seeding
from sklearn.cluster import KMeans
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_checker import check_env

from data_retrieval import DataGenerationModule


class Environment(Env):
    def __init__(
        self,
        max_couriers=4901,
        initial_timestamp=1666077600,
        time_step=300,
        episode_duration=3600,
    ):
        super(Environment, self).__init__()
        self.reward = 0
        self.observation_space = Dict(
            {
                "orders": Box(
                    low=-np.inf, high=np.inf, shape=(568546, 5), dtype=np.float64
                ),  # Max 1000 orders
                "couriers": Box(
                    low=-np.inf, high=np.inf, shape=(max_couriers, 5), dtype=np.float64
                ),  # Max 500 couriers
                "system": Box(low=0, high=np.inf, shape=(2,), dtype=np.int64),
            }
        )

        self.max_couriers = max_couriers
        self.action_mask = np.ones(max_couriers)
        self.action_space = Discrete(max_couriers)
        self.seed()

        self.done = False
        self.data_module = DataGenerationModule(
            "courier_wave_info_meituan.csv",
            "all_waybill_info_meituan_0322.csv",
            "dispatch_rider_meituan.csv",
            "dispatch_waybill_meituan.csv",
        )
        self.initial_timestamp = initial_timestamp
        self.time_step = time_step
        self.current_timestamp = self.initial_timestamp
        self.episode_duration = episode_duration
        self.episode_end_time = self.initial_timestamp + self.episode_duration
        # self.state = self.get_state(self.current_timestamp)
        self.active_orders = self.data_module.get_orders_in_time_window(
            self.current_timestamp, self.episode_end_time
        )
        # print(self.active_orders)
        self.active_couriers = self.data_module.get_active_couriers(
            self.current_timestamp
        )
        # self.active_couriers.drop("order_ids", axis=1, inplace=True)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # def reset(self, seed=None):
    #     self.reward = 0
    #     # Update to the next timestamp for a new episode
    #     self.current_timestamp += self.time_step

    #     # Retrieve the initial state based on the new timestamp
    #     self.active_orders = self.data_module.get_active_orders(self.current_timestamp)
    #     self.active_couriers = self.data_module.get_active_couriers(
    #         self.current_timestamp
    #     )
    #     self.update_action_mask()
    #     self.state = {
    #         "orders": self.active_orders.to_dict(orient="records"),
    #         "couriers": self.active_couriers.to_dict(orient="records"),
    #         "system": {
    #             "active_orders": len(self.active_orders),
    #             "active_couriers": len(self.active_couriers),
    #         },
    #     }
    #     self.action_mask = np.ones(self.max_couriers)
    #     self.done = False
    #     return self.state, {}

    def step(self, action):
        """
        Execute one step within the environment.
        """
        #print(action)
        #print(len(self.active_orders))
        if action not in range(self.active_couriers.shape[0]):
            self.reward -= 10
            return self.state, self.reward, self.done, False, {}

        if self.active_orders.shape[0] == 0:
            self.done = True
            return self.state, self.reward, self.done, False, {}

        #print(self.reward)
        order = self.active_orders.iloc[0]
        selected_courier = self.active_couriers.iloc[action]
        distance = np.sqrt(
            (selected_courier["grab_lat"] - order["sender_lat"]) ** 2
            + (selected_courier["grab_lng"] - order["sender_lng"]) ** 2
        )
        self.reward += 10 / (distance + 1)
        self.active_couriers.loc[action, "unfulfilled_orders"] += 1
        self.active_orders = self.active_orders.iloc[1:]
        self.update_action_mask()
        self.state = self.get_state()
        if self.active_orders.shape[0] == 0:
            # print("No orders left")
            # print("we are done")
            self.done = True
        return self.state, self.reward, self.done, False, {}

    def action_masks(self):
        return self.action_mask

    def update_action_mask(self):
        self.action_mask = np.zeros(self.max_couriers)

        for i, courier in self.active_couriers.iterrows():
            # print(courier)
            # Mask couriers with valid data only
            if not (courier == [-1, -1, -1, -1, -1]).all():
                self.action_mask[i] = 1

    def reset(self, seed=None):
        self.current_timestamp = self.current_timestamp + self.time_step
        self.episode_end_time = self.current_timestamp + self.episode_duration
        self.reward = 0
        # Retrieve orders for the entire time window
        self.active_orders = self.data_module.get_orders_in_time_window(
            self.current_timestamp, self.episode_end_time
        )
        self.active_couriers = self.data_module.get_active_couriers(
            self.current_timestamp
        )
        self.active_couriers.drop("order_ids", axis=1, inplace=True)

        self.done = False
        self.update_action_mask()
        self.state = self.get_state()
        return self.state, {}

    def get_state(self):

        orders = self.active_orders.values
        couriers = self.active_couriers.values
        # Convert "orders" and "couriers" to NumPy arrays
        # orders = np.array(
        #     [list(order.values()) for order in self.active_orders], dtype=np.float32
        # )
        # couriers = np.array(
        #     [list(courier.values()) for courier in self.active_couriers],
        #     dtype=np.float32,
        # )

        # Pad orders with dummy data if necessary
        max_orders = self.observation_space["orders"].shape[0]
        if len(orders) < max_orders:
            padding = max_orders - len(orders)
            dummy_order = np.array(
                [-1, -1, -1, -1, -1], dtype=np.float32
            )  # Dummy order
            orders = np.vstack([orders, np.tile(dummy_order, (padding, 1))])

        # Pad couriers with dummy data if necessary
        max_couriers = self.observation_space["couriers"].shape[0]
        if len(couriers) < max_couriers:
            padding = max_couriers - len(couriers)
            dummy_courier = np.array(
                [-1, -1, -1, -1, -1], dtype=np.float32
            )  # Dummy courier
            couriers = np.vstack([couriers, np.tile(dummy_courier, (padding, 1))])

        # Return the complete state dictionary
        return {
            "orders": orders,
            "couriers": couriers,
            "system": np.array([len(self.active_orders), len(self.active_couriers)]),
        }


# Create the environment
env = Environment()

# Check the environment
check_env(env)
