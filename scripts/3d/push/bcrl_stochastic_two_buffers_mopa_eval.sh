#!/bin/bash -x
# Please modify 'log_dir' and 'ckpt_num' before running this script.
# 'log_dir' is the directoy containing your checkpoint.
# 'ckpt_num' is the checkpoint number.

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
max_global_step='1500000'
max_episode_steps='250'
max_episode_step='250'
step_switch_policy_only='170000'
expert_num_trajectories='20000'
initial_policy_num_trajectories='10000'
expert_mode='bc-stochastic-mopa'
bc_rl_sampling_mode='two-buffers'
log_alpha='-0.93'

# machine parameters
bc_checkpoint="out/bc_visual_policy_stochastic_sawyer_push_32px_checkpoint_30out30_screenresfix/epoch_12.pth" # BC Visual Policy Stochastic checkpoint
mopa_checkpoint="out/mopa_rl_sawyer_push_checkpoint/ckpt_03000000.pt"
save_img_to_disk="False"
save_img_folder="out/bc_visual_policy_sawyer_push_32px_checkpoint_30out30_screenresfix-img_folder"
parallel_dataloading="False"
parallel_dataloading_mode="disk"

# data related parameters
env_image_size='32'
screen_width='32'
screen_height='32'


env="SawyerPushObstacle-v0"
debug="False"
reward_type='sparse'
log_root_dir="./logs"
mopa="True"
reuse_data="True"
action_range="0.5"
omega='0.7'
stochastic_eval="True"
invalid_target_handling="True"
vis_replay="True"
plot_type='3d'
ac_space_type="piecewise"
use_smdp_update="True"
success_reward="150.0"
max_reuse_data='15'
reward_scale="0.8"
evaluate_interval="10000"
timelimit='2.0'

# Resume training from checkpoint
# log_dir='./logs/rl.SawyerPushObstacle-v0.04.29.19.06.BC-RL-SAC.0'
# ckpt_num='5002'

# wandb parameters
group='SawyerPush-BC-RL-SAC-Stochastic-MoPA'

# for evaluation
date='None'
log_dir='logs/rl.SawyerPushObstacle-v0.06.15.20.33.BC-RL-SAC-Stochastic-MoPA.0'
ckpt_num='1200000'
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
    --max_episode_step $max_episode_step \
    --max_episode_steps $max_episode_steps \
    --debug $debug \
    --algo $algo \
    --seed $seed \
    --reward_type $reward_type \
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
    --timelimit $timelimit \
    --policy $policy \
    --obs_space $obs_space \
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
    --date $date \
    --is_train $is_train \
    --wandb $wandb \
    --ckpt_num $ckpt_num \
    --num_eval $num_eval \
    --save_rollout $save_rollout \
    --record $record \
    --log_dir $log_dir \
    --three_hundred_eval_five_seeds $three_hundred_eval_five_seeds \
    # --log_dir $log_dir \
    # --ckpt_num $ckpt_num \
