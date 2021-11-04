#!/bin/bash -x
gpu=$1
seed=$2

prefix="Asymmetric-MoPA-SAC"
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

# additional hyper-parameters to MoPA-RL
policy="asym"
obs_space="all"
lr_actor='0.00001'
lr_critic='0.00001'
env_image_size='32'
screen_width='32'
screen_height='32'
max_global_step='3000000'
record='False'

# wandb parameters
group='SawyerLift-Asymmetric-MoPA-SAC'

python -m rl.main \
    --log_root_dir $log_root_dir \
    --prefix $prefix \
    --env $env \
    --max_global_step $max_global_step \
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
    --policy $policy \
    --obs_space $obs_space \
    --lr_actor $lr_actor \
    --lr_critic $lr_critic \
    --env_image_size $env_image_size \
    --screen_width $screen_width \
    --screen_height $screen_height \
    --group $group \
    --record $record \
