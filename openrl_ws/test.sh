python ./openrl_ws/test.py \
    --task go1doubleobject \
    --algo mat \
    --sim_device cuda:0 \
    --rl_device cuda:0 \
    --num_envs 1 --checkpoint /home/reid/Projects/trustworthy_ai/multiagent-quadruped-environment/checkpoints/go1doubleobject/module.pt \
    # --record_video