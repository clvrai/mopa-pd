#<!/bin/bash -x
# Please modify 'log_dir' and 'ckpt_num' before running this script.
# 'log_dir' is the directoy containing your checkpoint.
# 'ckpt_num' is the checkpoint number.

gpu=$1
seed=$2

prefix="Asymmetric-MoPA-SAC"
algo='sac'
env="SawyerAssemblyObstacle-v0"
max_episode_step="250"
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
reward_scale="1.0"

# additional hyper-parameters to MoPA-RL
policy="asym"
obs_space="all"
lr_actor='0.00001'
lr_critic='0.00001'
env_image_size='32'
screen_width='32'
screen_height='32'
max_global_step='1500000'
record='False'

# wandb parameters
group='SawyerAssembly-Asymmetric-MoPA-SAC'

# for evaluation
date='None'
log_dir='out/MoPA_Asym_SAC_SawyerAssembly'
ckpt_num='3000000'
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
    --max_global_step $max_global_step \
    --gpu $gpu \
    --max_episode_step $max_episode_step \
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
    --is_train $is_train \
    --policy $policy \
    --obs_space $obs_space \
    --lr_actor $lr_actor \
    --lr_critic $lr_critic \
    --env_image_size $env_image_size \
    --screen_width $screen_width \
    --screen_height $screen_height \
    --group $group \
    --record $record \
    --date $date \
    --wandb $wandb \
    --ckpt_num $ckpt_num \
    --num_eval $num_eval \
    --save_rollout $save_rollout \
    --record $record \
    --log_dir $log_dir \
    --three_hundred_eval_five_seeds $three_hundred_eval_five_seeds \
