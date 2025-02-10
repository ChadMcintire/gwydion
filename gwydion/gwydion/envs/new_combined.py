import csv
import datetime
import logging
import time
from statistics import mean

import gymnasium
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from datetime import datetime

from gwydion.envs.deployment import get_max_cpu, get_max_mem, get_max_traffic, get_redis_deployment_list
from gwydion.envs.util import save_to_csv, get_cost_reward, get_latency_reward_redis, get_num_pods

# Number of Requests - Discrete Event
from gwydion.envs.deployment import get_max_cpu, get_max_mem, get_max_traffic, \
    get_online_boutique_deployment_list
from gwydion.envs.util import save_to_csv, get_num_pods, get_cost_reward, \
    get_latency_reward_online_boutique

# MIN and MAX Replication
MIN_REPLICATION = 1
MAX_REPLICATION = 8

###STEP COUNTER###

MAX_STEPS = 25  # MAX Number of steps per episode

# Possible Actions (Discrete)
ACTION_DO_NOTHING = 0
ACTION_ADD_1_REPLICA = 1
ACTION_ADD_2_REPLICA = 2
ACTION_ADD_3_REPLICA = 3
ACTION_ADD_4_REPLICA = 4
ACTION_ADD_5_REPLICA = 5
ACTION_ADD_6_REPLICA = 6
ACTION_ADD_7_REPLICA = 7
ACTION_TERMINATE_1_REPLICA = 8
ACTION_TERMINATE_2_REPLICA = 9
ACTION_TERMINATE_3_REPLICA = 10
ACTION_TERMINATE_4_REPLICA = 11
ACTION_TERMINATE_5_REPLICA = 12
ACTION_TERMINATE_6_REPLICA = 13
ACTION_TERMINATE_7_REPLICA = 14


# Action Moves
MOVES = ["None", "Add-1", "Add-2", "Add-3", "Add-4", "Add-5", "Add-6", "Add-7",
         "Stop-1", "Stop-2", "Stop-3", "Stop-4", "Stop-5", "Stop-6", "Stop-7"]


# Reward objectives
LATENCY = 'latency'
COST = 'cost'

# IDs
ID_DEPLOYMENTS = 0
ID_MOVES = 1

ID_MASTER = 0
ID_SLAVE = 1

ID_recommendation = 0
ID_product_catalog = 1
ID_cart_service = 2
ID_ad_service = 3
ID_payment_service = 4
ID_shipping_service = 5
ID_currency_service = 6
ID_redis_cart = 7
ID_checkout_service = 8
ID_frontend = 9
ID_email = 10

# Reward objectives
LATENCY = 'latency'
COST = 'cost'

# Deployments
DEPLOYMENTS = ["redis-leader", "redis-follower"]

# Deployments
DEPLOYMENTS = ["recommendationservice", "productcatalogservice", "cartservice", "adservice",
               "paymentservice", "shippingservice", "currencyservice", "redis-cart",
               "checkoutservice", "frontend", "emailservice"]

class Combined(gymnasium.Env):
    """Horizontal Scaling for Redis in Kubernetes - an OpenAI gym environment"""
    metadata = {'render.modes': ['human', 'ansi', 'array']}

    def __init__(self, k8s=False, goal_reward=COST, waiting_period=5):
        # Define action and observation space
        # They must be gym.spaces objects

        # CHECK TODO
        super(Redis, self).__init__()
        super(OnlineBoutique, self).__init__()

        self.k8s = k8s

        # CHECK TODO
        self.name = "redis_gym"
        self.name = "online_boutique_gym"

        self.__version__ = "0.0.1"
        self.seed()
        self.goal_reward = goal_reward
        self.waiting_period = waiting_period  # seconds to wait after action

        logging.info("[Init] Env: {} | K8s: {} | Version {} |".format(self.name, self.k8s, self.__version__))

        # Current Step
        self.current_step = 0

        # Actions identified by integers 0-n -> 15 actions!
        self.num_actions = 15

        # CHECK TODO
        # Multi-Discrete version
        # Deployment: Discrete 2 - Master[0], Slave[1]
        # Action: Discrete 9 - None[0], Add-1[1], Add-2[2], Add-3[3], Add-4[4],
        #                      Stop-1[5], Stop-2[6], Stop-3[7], Stop-4[8]

        # Multi-Discrete
        # Deployment: Discrete 11
        # Action: Discrete 9 - None[0], Add-1[1], Add-2[2], Add-3[3], Add-4[4],
        #                      Stop-1[5], Stop-2[6], Stop-3[7], Stop-4[8]


        self.action_space = spaces.MultiDiscrete([2, self.num_actions])
        self.action_space = spaces.MultiDiscrete([11, self.num_actions])

        # Observations: 22 Metrics! -> 2 * 11 = 22
        # "number_pods"                     -> Number of deployed Pods
        # "cpu_usage_aggregated"            -> via metrics-server
        # "mem_usage_aggregated"            -> via metrics-server
        # "cpu_requests"                    -> via metrics-server/pod
        # "mem_requests"                    -> via metrics-server/pod
        # "cpu_limits"                      -> via metrics-server
        # "mem_limits"                      -> via metrics-server
        # "lstm_cpu_prediction_1_step"      -> via pod annotation
        # "lstm_cpu_prediction_5_step"      -> via pod annotation
        # "average_number of requests"      -> Prometheus metric: sum(rate(http_server_requests_seconds_count[5m]))


        # Observations: 22 Metrics! -> 2 * 11 = 22
        # "number_pods"                     -> Number of deployed Pods
        # "cpu_usage_aggregated"            -> via metrics-server
        # "mem_usage_aggregated"            -> via metrics-server
        # "cpu_requests"                    -> via metrics-server/pod
        # "mem_requests"                    -> via metrics-server/pod
        # "cpu_limits"                      -> via metrics-server
        # "mem_limits"                      -> via metrics-server
        # "lstm_cpu_prediction_1_step"      -> via pod annotation
        # "lstm_cpu_prediction_5_step"      -> via pod annotation
        # "average_number of requests"      -> Prometheus metric: sum(rate(http_server_requests_seconds_count[5m]))



        self.min_pods = MIN_REPLICATION
        self.max_pods = MAX_REPLICATION

        # CHECK TODO
        self.num_apps = 2
        self.num_apps = 11

        # Deployment Data
        self.deploymentList = get_redis_deployment_list(self.k8s, self.min_pods, self.max_pods)

        # Deployment Data
        self.deploymentList = get_online_boutique_deployment_list(self.k8s, self.min_pods, self.max_pods)

        # Logging Deployment
        for d in self.deploymentList:
            d.print_deployment()



        self.observation_space = self.get_observation_space()

        # Action and Observation Space
        logging.info("[Init] Action Spaces: " + str(self.action_space))
        logging.info("[Init] Observation Spaces: " + str(self.observation_space))

        # Info
        self.total_reward = None
        self.avg_pods = []
        self.avg_latency = []

        # episode over
        self.episode_over = False
        self.info = {}

        # Keywords for Reward calculation
        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False
        self.cost_weight = 0  # add here a value to consider cost in the reward function

        self.time_start = 0
        self.execution_time = 0
        self.episode_count = 0
        self.file_results = "results.csv"
        self.obs_csv = self.name + "_observation.csv"
        self.df = pd.read_csv("datasets/real/" + self.deploymentList[0].namespace + "/v1/"
                              + self.name + '_' + 'observation.csv')

        self.none_counter = 0

        #CHECK TODO 
        self.action_stats = [0 for _ in range(self.num_actions)]

        self.traffic = self.simulation_traffic()

    def simulation_traffic(self):
        # CHECK TODO
        self.traffic = self.df['frontend_traffic_in'].tolist()
        self.traffic = self.df['redis-leader_traffic_in'].tolist()
        unique_traffic = []
        seen = set()

        for value in self.traffic:
            if value not in seen:
                unique_traffic.append(value)
                seen.add(value)

        # print(unique_traffic)
        return unique_traffic



    def normalize(self, obs):
        return obs / self.observation_space.high

    
