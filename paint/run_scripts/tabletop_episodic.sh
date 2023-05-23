python3 oracle.py \
env_name=tabletop_manipulation_no_walls \
num_train_frames=3000000 \
method_name=sac_episodic_H1000 \
train_horizon=1000 \
eval_horizon=1000 \
eval_every_frames=10000 \
replay_buffer_size=3000000 \
simple_buffer=false \
balanced_buffer=true \
final_fraction=0.1 \
final_timestep=500000 \
num_demos=50 \
reset_at_success=true \
seed=0
