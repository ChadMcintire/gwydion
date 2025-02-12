from gwydion.envs import Redis, OnlineBoutique
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from sb3_contrib import RecurrentPPO, MaskablePPO


def get_model(alg, env, tensorboard_log):
    model = 0
    if alg == 'ppo':
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log, n_steps=500)
    elif alg == 'recurrent_ppo':
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
    elif alg == 'a2c':
        model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)  # , n_steps=steps
    else:
        logging.info('Invalid algorithm!')

    return model


def get_load_model(alg, tensorboard_log, load_path):
    if alg == 'ppo':
        return PPO.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log, n_steps=500)
    elif alg == 'recurrent_ppo':
        return RecurrentPPO.load(load_path, reset_num_timesteps=False, verbose=1,
                                 tensorboard_log=tensorboard_log)  # n_steps=steps
    elif alg == 'a2c':
        return A2C.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log)
    else:
        logging.info('Invalid algorithm!')


def get_env(use_case, k8s, goal):
    envs = 0
    if use_case == 'redis':
        env = Redis(k8s=k8s, goal_reward=goal)
        # For faster training!
        # otherwise just comment the following lines

        env.reset()
        _, _, _, info = env.step([0, 0])
        info_keywords = tuple(info.keys())
        env = SubprocVecEnv([lambda: Redis(k8s=k8s, goal_reward=goal) for i in range(8)])
        envs = VecMonitor(env, filename="vec_redis_gym_results_", info_keywords=info_keywords)

    elif use_case == 'onlineboutique':
        env = OnlineBoutique(k8s=k8s, goal_reward=goal)
        # For faster training!
        # otherwise just comment the following lines

        env.reset()
        _, _, _, _, info = env.step([0, 0])
        info_keywords = tuple(info.keys())
        env = SubprocVecEnv([lambda: OnlineBoutique(k8s=k8s, goal_reward=goal) for i in range(8)])
        envs = VecMonitor(env, filename="vec_onlineboutique_gym_results_", info_keywords=info_keywords)

    else:
        logging.info('Invalid use_case!')

    return envs

