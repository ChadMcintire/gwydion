import hydra
import logging
from omegaconf import DictConfig
from gwydion.envs import Redis, OnlineBoutique
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback



logging.basicConfig(filename='run.log', filemode='w', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')



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
        _, _, _, info = env.step([0, 0])
        info_keywords = tuple(info.keys())
        env = SubprocVecEnv([lambda: OnlineBoutique(k8s=k8s, goal_reward=goal) for i in range(8)])
        envs = VecMonitor(env, filename="vec_onlineboutique_gym_results_", info_keywords=info_keywords)

    else:
        logging.info('Invalid use_case!')

    return envs


@hydra.main(version_base=None, config_path=".", config_name="config")
def run(cfg: DictConfig):
    print(f"Algorithm: {cfg.alg}")
    print(f"K8s Mode: {cfg.k8s}")
    print(f"Use Case: {cfg.use_case}")
    print(f"Goal: {cfg.goal}")
    print(f"Training: {cfg.training}")
    print(f"Testing: {cfg.testing}")
    print(f"Loading: {cfg.loading}")
    print(f"Load Path: {cfg.load_path}")
    print(f"Test Path: {cfg.test_path}")
    print(f"Steps: {cfg.steps}")
    print(f"Total Steps: {cfg.total_steps}")
    print(f"Available Moves: {cfg.MOVES}")

    # Import and initialize Environment
    logging.info(cfg)

    print("type steps", type(cfg.total_steps))

    env = get_env(cfg.use_case, cfg.k8s, cfg.goal)

    scenario = ''
    if cfg.k8s:
        scenario = 'real'
    else:
        scenario = 'simulated'

    tensorboard_log = "results/" + cfg.use_case + "/" + scenario + "/" + cfg.goal + "/"

    name = cfg.alg + "_env_" + cfg.use_case + "_goal_" + cfg.goal + "_k8s_" + str(cfg.k8s) + "_totalSteps_" + str(cfg.total_steps)

    # callback
    checkpoint_callback = CheckpointCallback(save_freq=cfg.steps, save_path="logs/" + name, name_prefix=name)


    if cfg.training:
        if cfg.loading:  # resume training
            model = get_load_model(alg, tensorboard_log, load_path)
            model.set_env(env)
            model.learn(total_timesteps=cfgtotal_steps, tb_log_name=name + "_run", callback=checkpoint_callback)
        else:
            model = get_model(alg, env, tensorboard_log)
            model.learn(total_timesteps=total_steps, tb_log_name=name + "_run", callback=checkpoint_callback)

        model.save(name)

    if cfg.testing:
        model = get_load_model(alg, tensorboard_log, test_path)
        test_model(model, env, n_episodes=100, n_steps=110, smoothing_window=5, fig_name=name + "_check2.png")





if __name__ == "__main__":
    run()
