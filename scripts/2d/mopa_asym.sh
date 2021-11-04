#!/bin/bash -x
gpu=$1
seed=$2

algo='sac'
prefix="Asymmetric-MoPA-SAC"
env="PusherObstacle-v0"
max_episode_step="400"
debug="False"
log_root_dir="./logs"
mopa="True"
reward_scale="0.2"
reuse_data="True"
action_range="1.0"
omega='0.5'
use_smdp_update="True"
stochastic_eval="True"
invalid_target_handling="True"
max_reuse_data='30'
ac_space_type="piecewise"
success_reward="150.0"

# additional hyper-parameters to MoPA-RL
policy="asym"
obs_space="all"
lr_actor='0.00001'
lr_critic='0.00001'
env_image_size='32'
screen_width='32'
screen_height='32'
max_global_step='10000000'
record='False'

# wandb parameters
group='2DPusherObstacle-Asymmetric-MoPA-SAC'

python -m rl.main \
    --log_root_dir $log_root_dir \
    --prefix $prefix \
    --env $env \
    --gpu $gpu \
    --max_episode_step $max_episode_step \
    --debug $debug \
    --algo $algo \
    --seed $seed \
    --mopa $mopa \
    --reward_scale $reward_scale \
    --reuse_data $reuse_data \
    --action_range $action_range \
    --omega $omega \
    --success_reward $success_reward \
    --stochastic_eval $stochastic_eval \
    --invalid_target_handling $invalid_target_handling \
    --max_reuse_data $max_reuse_data \
    --ac_space_type $ac_space_type \
    --use_smdp_update $use_smdp_update \
    --policy $policy \
    --obs_space $obs_space \
    --lr_actor $lr_actor \
    --lr_critic $lr_critic \
    --env_image_size $env_image_size \
    --screen_width $screen_width \
    --screen_height $screen_height \
    --group $group \
    --max_global_step $max_global_step \
    --record $record \
