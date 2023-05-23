python3 medal.py \
env_name=sawyer_peg_small_table \
num_train_frames=4000000 \
method_name=medal \
forward_agent.final_fraction=0.1 \
forward_agent.final_timestep=500000 \
backward_agent.balanced_buffer=true \
backward_agent.final_fraction=0.1 \
backward_agent.final_timestep=500000 \
train_horizon=100000 \
num_demos=50 \
seed=0
