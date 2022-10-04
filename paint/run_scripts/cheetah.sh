python oracle.py \
env_name=half_cheetah_flip \
num_train_frames=3000500 \
train_horizon=3000500 \
eval_horizon=2000 \
eval_every_frames=20000 \
use_stuck_buffer_for_Q=True \
use_stuck_discrim_for_term=True \
agent.stuck_discrim_unsupervised=False \
num_explore_steps=500 \
early_abort_threshold=0.5 \
method_name=paint \
reward_type=dense \
save_video=false
