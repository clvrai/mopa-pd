#!/bin/bash -x
# Please modify 'log_dir' and 'ckpt_num' before running this script.
# 'log_dir' is the directoy containing your checkpoint.
# 'ckpt_num' is the checkpoint number.

gpu=$1
seed=$2

prefix="MoPA-SAC"
env="SawyerLiftObstacle-v0"
algo='sac'
max_episode_step="250"
debug="False"
log_root_dir="./logs"
mopa="True"
reuse_data="True"
action_range="0.5"
omega='0.5'
stochastic_eval="True"
invalid_target_handling="True"
vis_replay="True"
plot_type='3d'
ac_space_type="piecewise"
use_smdp_update="True"
success_reward="150.0"
add_curr_rew="True"
max_reuse_data='15'
reward_scale="0.5"
evaluate_interval="10000"
ckpt_interval='10000'
# timelimit="1.5"

# evaluation parameters
log_dir='out/mopa_rl_sawyer_push_0.2million_checkpoint'
ckpt_num='940000'
date='02.01'
obs_space='state'
policy='mlp'
is_train='False'
wandb='False'
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
    --reuse_data $reuse_data \
    --action_range $action_range \
    --omega $omega \
    --stochastic_eval $stochastic_eval \
    --invalid_target_handling $invalid_target_handling \
    --vis_replay $vis_replay \
    --plot_type $plot_type \
    --use_smdp_update $use_smdp_update \
    --ac_space_type $ac_space_type \
    --success_reward $success_reward \
    --max_reuse_data $max_reuse_data \
    --reward_scale $reward_scale \
    --evaluate_interval $evaluate_interval \
    --ckpt_interval $ckpt_interval \
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
    --log_dir $log_dir \
    --screen_width $screen_width \
    --screen_height $screen_height \
