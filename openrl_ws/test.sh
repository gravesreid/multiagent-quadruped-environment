python ./openrl_ws/test.py \
    --task go1multiobject \
    --algo mat \
    --sim_device cuda:0 \
    --rl_device cuda:0 \
    --num_envs 1 --checkpoint ./checkpoints/go1multiobject/module.pt \
    # --record_video