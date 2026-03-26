# rollout_steps=1, 15k steps
# torchrun --nproc_per_node 4 scripts/train_cast_wm.py --encoder dino
# rollout_steps=3, +10k steps
# torchrun --nproc_per_node 4 scripts/train_cast_wm.py --encoder dino --resume /mnt/weka/zhougrp/octo_wm_cast/dino_cast_wm_20260228_193050/checkpoint_step15000.pt --rollout_steps 3  --batch_size 8 --num_steps 25000 --lr 5e-5 --skip_optimizer_state
# rollout_steps=7, +15k steps
torchrun --nproc_per_node 4 scripts/train_cast_wm.py --encoder dino --resume /mnt/weka/zhougrp/octo_wm_cast/dino_cast_wm_20260228_193050/checkpoint_step25000.pt --rollout_steps 7  --batch_size 8 --num_steps 40000 --lr 5e-5 --skip_optimizer_state