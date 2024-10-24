python ./openrl_ws/test.py \
    --task go1midlevel \
    --algo mat \
    --sim_device cuda:0 \
    --rl_device cuda:0 \
    --num_envs 1 --checkpoint ./checkpoints/go1midlevel/module.pt \
    # --record_video