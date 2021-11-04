#!/bin/bash -x
gpu=$1
seed=$2

# Model parameters
prefix="MoPA-RL-RL-SAC"
algo='aac'
policy="asym"
obs_space="all"
lr_actor='0.00001'
lr_critic='0.00001'
pretrain_step='30000'
pretrain_evaluate_interval='5000'
max_global_step='10000000'
max_episode_steps='400'
step_switch_policy_only='200000'
evaluate_interval="10000"
expert_mode="mopa"
expert_num_trajectories="15000"

# machine parameters
mopa_checkpoint="out/mopa_rl_2d_pusher_obstacle_checkpoint/ckpt_03000000.pt"
bc_checkpoint="placeholder"
save_img_to_disk="False"
save_img_folder="out/bc_visual_policy_2d_pusher_32px_checkpoint_30out30_screenresfix-img_folder"
parallel_dataloading="False"
parallel_dataloading_mode="disk"

# data related parameters
env_image_size='32'
screen_width='32'
screen_height='32'

env="PusherObstacle-v0"
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


# wandb parameters
group='2D-Pusher-BC-RL-SAC'


# for evaluation
date='None'
# log_dir='out/mopa_rl_asymmetric_sac_2d_pusher_1234_best_checkpoint' # without pretrained critic
# ckpt_num='6500000' # without pretrained critic
log_dir='out/mopa-rl-asym-sac-pretrained-critic-1234-2dpush' # with pretrained critic
ckpt_num='9000000' # with pretrained critic
is_train='False'
wandb='False'
num_eval='100'
save_rollout='True'
record='True'
three_hundred_eval_five_seeds='True'

python -m rl.main \
    --log_root_dir $log_root_dir \
    --prefix $prefix \
    --env $env \
    --gpu $gpu \
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
    --evaluate_interval $evaluate_interval \
    --lr_actor $lr_actor \
    --lr_critic $lr_critic \
    --bc_checkpoint $bc_checkpoint \
    --save_img_to_disk $save_img_to_disk \
    --save_img_folder $save_img_folder \
    --parallel_dataloading $parallel_dataloading \
    --parallel_dataloading_mode $parallel_dataloading_mode \
    --env_image_size $env_image_size \
    --pretrain_step $pretrain_step \
    --pretrain_evaluate_interval $pretrain_evaluate_interval \
    --max_global_step $max_global_step \
    --max_episode_steps $max_episode_steps \
    --group $group \
    --step_switch_policy_only $step_switch_policy_only \
    --screen_width $screen_width \
    --screen_height $screen_height \
    --expert_mode $expert_mode \
    --mopa_checkpoint $mopa_checkpoint \
    --expert_num_trajectories $expert_num_trajectories \
    --date $date \
    --is_train $is_train \
    --wandb $wandb \
    --ckpt_num $ckpt_num \
    --num_eval $num_eval \
    --save_rollout $save_rollout \
    --record $record \
    --log_dir $log_dir \
    --three_hundred_eval_five_seeds $three_hundred_eval_five_seeds \
