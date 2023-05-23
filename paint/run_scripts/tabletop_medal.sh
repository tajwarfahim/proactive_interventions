python3 medal.py \
env_name=tabletop_manipulation_no_walls \
num_train_frames=3000000 \
method_name=medal \
train_horizon=200000 \
eval_horizon=1000 \
eval_every_frames=10000 \
forward_agent.final_fraction=0.1 \
forward_agent.final_timestep=500000 \
backward_agent.balanced_buffer=true \
backward_agent.final_fraction=0.1 \
backward_agent.final_timestep=500000 \
num_demos=50 \
seed=0
