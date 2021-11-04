#!/bin/bash -x
gpu=$1
seed=$2

# Model parameters
prefix="BC-RL-SAC-Stochastic-MoPA"
algo='aac'
policy="asym"
obs_space="all"
lr_actor='0.00001'
lr_critic='0.00001'
pretrain_step='0'
pretrain_evaluate_interval='5000'
max_global_step='2100000'
max_episode_steps='250'
max_episode_step='250'
step_switch_policy_only='2100000'
expert_num_trajectories='20000'
initial_policy_num_trajectories='10000'
expert_mode='bc-stochastic-mopa'
bc_rl_sampling_mode='two-buffers'
log_alpha='-40'

# machine parameters
bc_checkpoint="out/bc_visual_policy_stochastic_sawyer_lift_32px_0.94m_checkpoint_22out30_screenresfix/epoch_13.pth"
mopa_checkpoint="out/mopa_rl_sawyer_lift_0.94m_checkpoint/ckpt_00940000.pt"
save_img_to_disk="False"
save_img_folder="out/bc_visual_policy_sawyer_lift_32px_checkpoint_21out30_screenresfix-img_folder"
parallel_dataloading="False"
parallel_dataloading_mode="disk"

# data related parameters
env_image_size='32'
screen_width='32'
screen_height='32'


env="SawyerLiftObstacle-v0"
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
max_reuse_data='15'
reward_scale="0.5"
evaluate_interval="10000"

# wandb parameters
group='SawyerLift-BC-RL-SAC-Stochastic-MoPA'

python -m rl.main \
    --log_root_dir $log_root_dir \
    --prefix $prefix \
    --env $env \
    --gpu $gpu \
    --max_episode_step $max_episode_step \
    --max_episode_steps $max_episode_steps \
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
    --group $group \
    --step_switch_policy_only $step_switch_policy_only \
    --screen_width $screen_width \
    --screen_height $screen_height \
    --expert_num_trajectories $expert_num_trajectories \
    --mopa_checkpoint $mopa_checkpoint \
    --expert_mode $expert_mode \
    --initial_policy_num_trajectories $initial_policy_num_trajectories \
    --bc_rl_sampling_mode $bc_rl_sampling_mode \
    --log_alpha $log_alpha \
