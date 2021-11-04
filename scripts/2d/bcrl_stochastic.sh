#!/bin/bash -x
gpu=$1
seed=$2

# Model parameters
prefix="BC-RL-SAC-Stochastic"
algo='aac'
policy="asym"
obs_space="all"
lr_actor='0.00001'
lr_critic='0.00001'
pretrain_step='180000'
pretrain_evaluate_interval='5000'
max_global_step='10000000'
max_episode_steps='400'
evaluate_interval="10000"
step_switch_policy_only='0'
expert_num_trajectories='0'
initial_policy_num_trajectories='20000'
expert_mode='bc-stochastic'
bc_rl_sampling_mode='one-buffer'

# machine parameters
bc_checkpoint="out/bc_visual_policy_stochastic_2d_push_32px_checkpoint_30out30_screenresfix/epoch_71.pth"
mopa_checkpoint="out/mopa_rl_2d_pusher_obstacle_checkpoint/ckpt_03000000.pt"
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
group='2D-Pusher-BC-RL-SAC-Stochastic'

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
    --expert_num_trajectories $expert_num_trajectories \
    --mopa_checkpoint $mopa_checkpoint \
    --expert_mode $expert_mode \
    --initial_policy_num_trajectories $initial_policy_num_trajectories \
    --bc_rl_sampling_mode $bc_rl_sampling_mode \
