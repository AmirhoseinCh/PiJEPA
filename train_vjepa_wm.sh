# rollout_steps=1, 15k steps
# torchrun --nproc_per_node 4 scripts/train_cast_wm.py --encoder vjepa

# rollout_steps=3, +10k steps
# torchrun --nproc_per_node 4 scripts/train_cast_wm.py --encoder vjepa --resume /mnt/weka/zhougrp/octo_wm_cast/vjepa_cast_wm_20260228_220505/checkpoint_step15000.pt --rollout_steps 3  --batch_size 8 --num_steps 25000 --lr 5e-5 --skip_optimizer_state

# rename final.pt to checkpoint_step25000.pt in vjepa checkpoint dir before running the next command
# mv /mnt/weka/zhougrp/octo_wm_cast/vjepa_cast_wm_20260228_220505/final.pt /mnt/weka/zhougrp/octo_wm_cast/vjepa_cast_wm_20260228_220505/checkpoint_step25000.pt

# rollout_steps=7, +15k steps
torchrun --nproc_per_node 4 scripts/train_cast_wm.py --encoder vjepa --resume /mnt/weka/zhougrp/octo_wm_cast/vjepa_cast_wm_20260228_220505/checkpoint_step25000.pt --rollout_steps 7  --batch_size 8 --num_steps 40000 --lr 5e-5 --skip_optimizer_state