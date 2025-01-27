python3 run.py --alg recurrent_ppo --k8s=False --use_case redis --goal latency \
--training=True --testing=False --loading=False \
--load_path logs/a2c_env_onlineboutique_goal_cost_k8s_False_totalSteps_500000/a2c_env_redis_goal_cost_k8s_False_totalSteps_500000.zip \
--test_path logs/a2c_env_onlineboutique_goal_latency_k8s_False_totalSteps_500000/a2c_env_onlineboutique_goal_latency_k8s_False_totalSteps_500000.zip \
--steps 50000 --total_steps 250000
