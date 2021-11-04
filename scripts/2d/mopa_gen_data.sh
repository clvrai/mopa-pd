#!/bin/bash -x
gpu=$1
seed=$2

algo='sac'
prefix="MoPA-SAC"
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

# evaluation parameters
date='01.01'
obs_space='state'
policy='mlp'
is_train='False'
wandb='False'
ckpt_num='3000000'
num_eval='500'
save_rollout='True'
record='False'
eval_noise='True'

# image data
screen_width='32'
screen_height='32'

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
    --date $date \
    --obs_space $obs_space \
    --policy $policy \
    --is_train $is_train \
    --wandb $wandb \
    --ckpt_num $ckpt_num \
    --num_eval $num_eval \
    --save_rollout $save_rollout \
    --record $record \
    --eval_noise $eval_noise \
    --screen_width $screen_width \
    --screen_height $screen_height \
