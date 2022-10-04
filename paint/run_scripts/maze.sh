python3 oracle.py \
env_name=maze-config0 \
num_train_frames=200000 \
method_name=exp \
use_stuck_buffer_for_Q=true \
use_stuck_discrim_label=true \
use_stuck_discrim_for_term=true \
early_abort_threshold=0.5 \
num_explore_steps=500 \
eval_every_frames=2000 \
seed=0
